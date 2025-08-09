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
    /// Embedding usually but not always used in Natural Language Processing.
    /// </summary>
    public class Embedding : Module
    {
        // vars
        public readonly int _size, _dim;
        // Weights & Gradients
        float[][] _wt, _wtGrad;
        ConcurrentBag<List<float[]>[]> _wtGradls;
        // Optimizer
        readonly Adam _optim;

        /// <summary>
        /// Forwards Pass
        /// </summary>
        /// <param name="x">List of Embedding indexs.</param>
        /// <returns>Shape Tensor(float[batch][_dim])</returns>
        public Tensor<float[][]> Forward(Tensor<int[]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }
        /// <summary>
        /// Forwards Pass
        /// </summary>
        /// <param name="x">List of Embedding indexs.</param>
        /// <returns>Shape float[batch][_dim]</returns>
        public float[][] Forward(int[] values, out object ctx)
        {
            ctx = values;
            var outs = new float[values.Length][];
            // Get Embedding foreach
            for (int i = 0; i < values.Length; i++)
                outs[i] = _wt[values[i]].ToArray();
            return outs;
        }

        /// <summary>
        /// Inlined Backwards Function
        /// </summary>
        /// <param name="y">Output of forwards pass</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            Backward((float[][])y._grad, y._ctx);
        }

        /// <summary>
        /// Inlined Backwards Function
        /// </summary>
        /// <param name="grad">Out gradient shape float[batch][_dim]</param>
        /// <param name="ctx_">Context for backwards pass</param>
        public void Backward(float[][] grad, object ctx_)
        {
            var ctx = (int[])ctx_;
            // Validate Shape
            if (grad.Length != ctx.Length || grad[0].Length != _dim)
                throw new Exception($"Invalid gradient: got ({grad.Length}, {grad[0].Length}) " +
                        $"it should be ({ctx.Length}, {_dim})");
            // Weight Gradient
            var gradls = _wt.Select(w => new List<float[]>(100)).ToArray();
            for (int i = 0; i < grad.Length; i++)
                gradls[ctx[i]].Add(grad[i]);
            // Multithreaded
            _wtGradls.Add(gradls);
        }

        /// <summary>
        /// Gradient Average & Optimizer
        /// </summary>
        /// <param name="len">Not used</param>
        /// <returns>Max Gradient Magnitude</returns>
        public override float SmoothMax(int len)
        {
            // Validation
            if (_wtGradls.Count == 0)
                throw new Exception("There are no gradients. You need to call Backward() first.");
            // Reduce
            var wgl = _wtGradls.ToList();
            _wtGradls = new ConcurrentBag<List<float[]>[]>();
            var wtGradls = wgl[0];
            if (wgl.Count > 1)
            {
                var cnt = Enumerable.Range(1, wgl.Count - 1).ToArray();
                for (int i = 0; i < wtGradls.Length; i++)
                    wtGradls[i].AddRange(cnt.SelectMany(j => wgl[j][i]));
            }
            // Weight Gradient
            var zeros = new float[_dim];
            _wtGrad = wtGradls.Select(w =>
            {
                if (w.Count == 0)   // Has no values give it zeros
                    return zeros;
                return Utilz.Transpose(w).Select(wi => wi.Average()).ToArray();
            }).ToArray();
            // Optimizer
            _wtGrad = _optim.Update(_wt, _wtGrad);
            // Get Max Magnitude
            return _wtGrad.SelectMany(g => g).Select(Math.Abs).Max();
        }

        /// <summary>
        /// Step Parameters by gradient
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="div">Max div (GradNorm)</param>
        public override void Step(float lr, float div = 1)
        {
            div = 1 / div;
            // Should learn slow & never fast
            lr = Math.Min(lr * 0.1f, 0.01f);
            // May need to calculate
            if (_wtGrad == null)
                SmoothMax(1);
            // Grad Norm
            var wtGrad = _wtGrad;
            _wtGrad = null;
            if (div != 1)
                wtGrad = wtGrad.Select(g => g.Select(v => v * div).ToArray()).ToArray();
            // Step
            for (int i = 0; i < _size; i++)
            {
                var w = _wt[i];
                var g = wtGrad[i];
                // Step
                for (int j = 0; j < _dim; j++)
                    w[j] -= g[j] * lr;
            }
        }

        /// <summary>
        /// Number of Parameters
        /// </summary>
        /// <returns>Number of Parameters</returns>
        public override long NumParams()
            => _size * _dim;

        /// <summary>
        /// Parameters Deep copy
        /// </summary>
        /// <returns>Parameters Deep copy</returns>
        public override object Parameters()
            => _wt.Select(w => w.ToArray()).ToArray();       // Deep Copy

        /// <summary>
        /// Load State
        /// </summary>
        /// <param name="items">object state</param>
        public override void LoadState(object items)
        {
            var w = (float[][])items;
            // validate shape
            if (w.Length != _wt.Length || w[0].Length != _wt[0].Length)
                throw new Exception($"Loaded weight shape is ({w.Length}, {w[0].Length}) " +
                        $"it should be ({_wt.Length}, {_wt[0].Length}).");
            // Deep copy
            _wt = w.Select(s => s.ToArray()).ToArray();
        }

        /// <summary>
        /// Load State JSON
        /// </summary>
        /// <param name="items">JSON object</param>
        public override void LoadStateJSON(object items)
        {
            var w = JsonConvert.DeserializeObject<float[][]>(items.ToString(), _settings);
            LoadState(w);
        }

        /// <summary>
        /// Embedding usually but not always used in Natural Language Processing.
        /// </summary>
        /// <param name="size">The length. maxIndex + 1</param>
        /// <param name="dim">Embedding size</param>
        /// <param name="name0">Parents Names</param>
        public Embedding(int size, int dim, string name0 = null)
            : base(name0, $"Embedding(size:{size}, dim:{dim})")
        {
            if (size <= 0) throw new Exception("size <= 0");
            if (dim <= 0) throw new Exception("dim <= 0");
            _size = size;
            _dim = dim;
            // Initialize Weights
            _wt = new bool[size].Select(no => new float[dim]).ToArray();
            InitWts(_wt);
            // Optimizer
            _optim = new Adam(_wt);
            _wtGradls = new ConcurrentBag<List<float[]>[]>();
        }
    }
}
