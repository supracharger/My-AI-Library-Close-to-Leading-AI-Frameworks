using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace MiniNet
{
    /// <summary>
    /// First half of inputs are gated against the second half of inputs.
    /// Example: value[i]*wt[i] + value[i+length/2]*(1-wt[i])
    /// </summary>
    public class Gate : Module
    {
        // vars
        readonly int _nInput, _dim;
        readonly float _minWt, _maxWt;
        // Weight & Gradient
        float[] _wt, _wtGrad;
        ConcurrentBag<List<float>[]> _wtGradls;
        // Optimizer
        readonly Adam _optim;

        /// <summary>
        /// Gate Forward
        /// </summary>
        /// <param name="x">Values to be gated</param>
        /// <returns>Shape[batch][x[0].Length / 2]</returns>
        public Tensor<float[][]> Forward(Tensor<float[][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }
        /// <summary>
        /// Inside Gate Forward
        /// </summary>
        /// <param name="x">Values to be gated</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <returns>Shape[batch][x[0].Length / 2]</returns>
        public float[][] Forward(float[][] x, out object ctx)
        {
            if (x[0].Length != _nInput)
                throw new Exception($"Invalid input shape. Should be [batch][{_nInput}] got [{x.Length}][{x[0].Length}].");
            ctx = x;
            var outs = new float[x.Length][];
            for (int i = 0; i < x.Length; i++)
            {
                var xi = x[i];
                var oi = outs[i] = new float[_dim];
                for (int j = 0; j < _dim; j++)
                {
                    var w = _wt[j];
                    oi[j] = w * xi[j] + (1 - w) * xi[_dim + j];
                }
            }
            return outs;
        }

        /// <summary>
        /// Inside Gate Backward
        /// </summary>
        /// <param name="y">Output node of forward pass</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }
        /// <summary>
        /// Inside Gate Backward
        /// </summary>
        /// <param name="grad">Output Gradient</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <param name="requiresInGrad">If In gradient is needed</param>
        /// <returns>Input gradient</returns>
        public float[][] Backward(float[][] grad, object ctx, bool requiresInGrad = true)
        {
            var x = (float[][])ctx;
            // Validate Shape
            if (grad.Length != x.Length || grad[0].Length != _dim)
                throw new Exception($"Invalid shape. Should be [{x.Length}][{_dim}] got [{grad.Length}][{grad[0].Length}]");
            var wtGradls = new bool[_dim].Select(no => new List<float>(x.Length)).ToArray();
            // If input gradient is needed
            float[][] ingrad = null;
            if (requiresInGrad)
            {
                ingrad = new float[x.Length][];
                // Loop batch
                for (int i = 0; i < grad.Length; i++)
                {
                    var gi = grad[i];
                    var xi = x[i];
                    var og = ingrad[i] = new float[_nInput];
                    // Loop half of the input size
                    for (int j = 0; j < _dim; j++)
                    {
                        var w = _wt[j];
                        var g = gi[j];
                        og[j] = w * g;                                      // Input Gradient 1
                        og[_dim + j] = (1 - w) * g;                         // Input Gradient 2
                        wtGradls[j].Add((xi[j] - xi[_dim + j]) * g);        // Weight Gradient
                    }
                }
            }
            // Does Not need input gradient
            else
            {
                // Loop batch
                for (int i = 0; i < grad.Length; i++)
                {
                    var gi = grad[i];
                    var xi = x[i];
                    // Loop half of the input size
                    for (int j = 0; j < _dim; j++)
                        wtGradls[j].Add((xi[j] - xi[_dim + j]) * gi[j]);    // Weight Gradient
                }
            }
            // Multithreaded
            _wtGradls.Add(wtGradls);
            return ingrad;
        }

        /// <summary>
        /// Get Weight Gradient & Use optimizer
        /// </summary>
        /// <param name="len">Smooth len</param>
        /// <returns>Max Magnitude of Gradient (GradNorm)</returns>
        public override float SmoothMax(int len)
        {
            // Validation
            if (_wtGradls.Count == 0)
                throw new Exception("There are no gradients. You need to call Backward() first.");
            // Reduce
            var wgl = _wtGradls.ToList();
            _wtGradls = new ConcurrentBag<List<float>[]>();
            var wtGradls = wgl[0];
            if (wgl.Count > 1)
            {
                var cnt = Enumerable.Range(1, wgl.Count - 1).ToArray();
                for (int i = 0; i < wtGradls.Length; i++)
                    wtGradls[i].AddRange(cnt.SelectMany(j => wgl[j][i]));
            }
            // Weight Gradient
            _wtGrad = wtGradls.Select(g => BestDirection(EMA.RunSeries(g, len))).ToArray();
            // Optimizer
            _wtGrad = _optim.Update(_wt, _wtGrad);
            // Max Magnitude
            return _wtGrad.Select(Math.Abs).Max();
        }

        /// <summary>
        /// Step Parameters from gradient
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="div">Max divisor</param>
        public override void Step(float lr, float div = 1)
        {
            div = 1 / div;
            if (_wtGrad == null)    // Calculate
                SmoothMax(1);
            var wtGrad = _wtGrad;
            _wtGrad = null;
            // Grad Norm
            if (div != 1)
                wtGrad = wtGrad.Select(g => g * div).ToArray();
            // Update
            _wt = _wt.Select((w, i) => Math.Min(Math.Max(w - wtGrad[i] * lr, _minWt), _maxWt)).ToArray();
        }

        /// <summary>
        /// Parameters
        /// </summary>
        /// <returns>Parameters</returns>
        public override object Parameters()
        => _wt.ToArray();       // Deep copy

        /// <summary>
        /// Number of Parameters
        /// </summary>
        /// <returns>Number of Parameters</returns>
        public override long NumParams()
        => _wt.Length;

        /// <summary>
        /// Load State
        /// </summary>
        /// <param name="items">object state</param>
        public override void LoadState(object items)
        {
            var w = (float[])items;
            // Validate
            if (w.Length != _wt.Length)
                throw new Exception($"Not the correct length. Should be [{_wt.Length}] got [{w.Length}].");
            // Deep copy
            _wt = w.ToArray();
        }

        /// <summary>
        /// Load State JSON
        /// </summary>
        /// <param name="items">JSON object</param>
        public override void LoadStateJSON(object items)
        {
            var w = JsonConvert.DeserializeObject<float[]>(items.ToString(), _settings);
            LoadState(w);
        }

        // Special Initialization for Gate Weights
        public static void InitWtsGate(float[] w, float start = 0.5f, float pct = 0.1f)
        {
            var rand = Rand2.Uniform(w.Length).Select(v => pct * (v * 2 - 1) + start).ToList();
            for (int i = 0; i < w.Length; i++)
                w[i] = rand[i];
        }

        /// <summary>
        /// First half of inputs are gated against the second half of inputs.
        /// Example: value[i]*wt[i] + value[i+length/2]*(1-wt[i])
        /// </summary>
        /// <param name="nInput">In Features</param>
        /// <param name="minWt">Minimum weight for gate</param>
        /// <param name="maxWt">Maximum weight for gate</param>
        /// <param name="name0">Parents names</param>
        public Gate(int nInput, double minWt = 0.01, double maxWt = 0.99, string name0 = null)
            : base(name0, $"Gate(nInput:{nInput}, minWt:{minWt}, maxWt:{maxWt})")
        {
            // First half of inputs are gated against the second half of inputs
            if (nInput % 2 != 0) throw new Exception("nInput % 2 != 0");
            // Validation
            if (minWt < 0 || minWt > 1) throw new Exception("minWt < 0 || minWt > 1");
            if (maxWt < 0 || maxWt > 1) throw new Exception("maxWt < 0 || maxWt > 1");
            if (minWt >= maxWt) throw new Exception("minWt >= maxWt");
            _nInput = nInput;
            _minWt = (float)minWt;
            _maxWt = (float)maxWt;
            // First half of inputs are gated against the second half of inputs
            _dim = nInput / 2;
            _wt = new float[_dim];
            InitWtsGate(_wt);
            _optim = new Adam(_wt);
            _wtGradls = new ConcurrentBag<List<float>[]>();
        }
    }
}
