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
    /// Linear or Fully Connected Deep Layer
    /// </summary>
    public class Linear : Module
    {
        // Weights & Gradients
        float[][] _wt, _wtGrad;
        float[] _bias, _biasGrad;
        ConcurrentBag<List<float>[][]> _wtGradls;
        ConcurrentBag<List<float>[]> _biasGradls;
        // vars
        public readonly int _inFeatures, _outFeatures;
        readonly int _nInput;
        // Optimizer
        readonly List<Adam> _optim;

        /// <summary>
        /// Forwards Pass of Linear Layer
        /// </summary>
        /// <param name="x">Input into layer. Shape[batch][_inFeatures]</param>
        /// <returns>Shape[batch][_outFeatures]</returns>
        public Tensor<float[][]> Forward(Tensor<float[][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }
        /// <summary>
        /// Forwards Pass of Linear Layer
        /// </summary>
        /// <param name="x">Input into layer. Shape[batch][nSeq][_inFeatures]</param>
        /// <returns>Shape[batch][nSeq][_outFeatures]</returns>
        public Tensor<float[][][]> Forward(Tensor<float[][][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward3d, true);
        }
        /// <summary>
        /// Forwards Pass of Linear Layer
        /// </summary>
        /// <param name="x">Input into layer. Shape[m][_inFeatures]</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <returns>Shape[m][_outFeatures]</returns>
        public float[][] Forward(float[][] x, out object ctx)
        {
            // Validate Shape
            if (x[0].Length != _nInput)
                throw new Exception($"Invalid shape. Should be [batch][{_nInput}] got [{x.Length}][{x[0].Length}]");
            ctx = x;
            // Forward Pass
            var y = Utilz.Matmul(x, Utilz.TransposeArray(_wt));
            // My Average Activation Function
            var div = 1 / (float)Math.Sqrt(_nInput);
            MultiplyInPlace(y, div);
            //y = y.Select(s => s.Select(v => v * div).ToArray()).ToArray();
            // Bias
            if (_bias != null)
                for (int i = 0; i < y.Length; i++)
                {
                    var yi = y[i];
                    for (int j = 0; j < _bias.Length; j++)
                        yi[j] += _bias[j];
                }
            return y;
        }
        /// <summary>
        /// Forwards Pass of Linear Layer float[3d]
        /// </summary>
        /// <param name="x">Input into layer. Shape[batch][nSeq][_inFeatures]</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <returns>Shape[batch][nSeq][_outFeatures]</returns>
        public float[][][] Forward(float[][][] x_, out object ctx)
        {
            // Flatten first 2 dims
            var x = x_.SelectMany(xi => xi).ToArray();
            // Use Forward 2d
            var y = Forward(x, out ctx);
            // Reshape back from flatten
            return Utilz.Reshape(y, x_.Length, x_[0].Length, _outFeatures);
        }

        /// <summary>
        /// Inside Backwards Function
        /// </summary>
        /// <param name="y">Output node from forward pass</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }
        /// <summary>
        /// Inside Backwards Function 3d
        /// </summary>
        /// <param name="y">Output node from forward pass</param>
        void Backward3d(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }
        /// <summary>
        /// Inside Backwards Function
        /// </summary>
        /// <param name="grad">Output Gradient</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <param name="requiresInGrad">If Input gradient is needed</param>
        /// <returns>Input Gradient</returns>
        public float[][] Backward(float[][] grad, object ctx, bool requiresInGrad = true)
        {
            // grad[batch][nOut]
            // Validate shape
            var x = (float[][])ctx;
            if (grad.Length != x.Length || grad[0].Length != _wt.Length)
                throw new Exception($"Invalid shape. Should be [{x.Length}][{_wt.Length}] got [{grad.Length}][{grad[0].Length}]");
            // My Average Activation Function
            var div = 1 / (float)Math.Sqrt(_nInput);
            var gd2 = Multiply(grad, div);
            //var gd2 = grad.Select(g => g.Select(v => v * div).ToArray()).ToArray();
            // Weight Gradient
            var wtGradls = Mul(Utilz.TransposeArray(gd2), x);
            //var wtGrad = Utilz.Matmul(Utilz.TransposeArray(grad), x);
            // If using bias: get bias grad
            var usebias = _bias != null;
            List<float>[] biasGradls = null;
            if (usebias)
                biasGradls = Utilz.TransposeInnerList(grad);
            // If input gradient is required
            float[][] ingrad = null;
            if (requiresInGrad)
                ingrad = Utilz.Matmul(gd2, _wt);
            // Mutithreaded
            _wtGradls.Add(wtGradls);
            if (usebias)
                _biasGradls.Add(biasGradls);
            return ingrad;
        }
        /// <summary>
        /// Inside Backwards Function 3d
        /// </summary>
        /// <param name="grad">Output Gradient</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <param name="requiresInGrad">If Input gradient is needed</param>
        /// <returns>Input Gradient</returns>
        public float[][][] Backward(float[][][] grad_, object ctx, bool requiresInGrad = true)
        {
            // Flatten first 2 dims
            var grad = grad_.SelectMany(g => g).ToArray();
            // Run Backwards 2d
            var ingrad = Backward(grad, ctx, requiresInGrad);
            if (!requiresInGrad)
                return null;
            // Reshape back from flatten
            return Utilz.Reshape(ingrad, grad_.Length, grad_[0].Length, _inFeatures);
        }

        /// <summary>
        /// Reduce Gradient & Optimize
        /// </summary>
        /// <param name="len">Length of smooth</param>
        /// <returns>Max magnitude of gradient</returns>
        public override float SmoothMax(int len)
        {
            // Validation
            if (_wtGradls.Count == 0)
                throw new Exception("There are no gradients. You need to call Backward() first.");
            // Reduce
            var wgl = _wtGradls.ToList();
            _wtGradls = new ConcurrentBag<List<float>[][]>();
            var ls = wgl[0];
            if (wgl.Count > 1)
            {
                var cnt = Enumerable.Range(1, wgl.Count - 1).ToArray();
                for (int i = 0; i < ls.Length; i++)
                {
                    var wg = ls[i];
                    for (int j = 0; j < wg.Length; j++)
                        wg[j].AddRange(cnt.SelectMany(k => wgl[k][i][j]));
                }
            }
            // Gradient Weight
            _wtGrad = ls.Select(g => g.Select(l => BestDirection(EMA.RunSeries(l, len))).ToArray()).ToArray();
            // Optimizer
            _wtGrad = _optim[0].Update(_wt, _wtGrad);
            // Max Magnitude
            var max = _wtGrad.SelectMany(g => g).Select(Math.Abs).Max();
            // Bias
            if (_bias != null)
            {
                // Reduce
                var bg = _biasGradls.ToList();
                _biasGradls = new ConcurrentBag<List<float>[]>();
                var bls = bg[0];
                if (bg.Count > 1)
                {
                    var cnt = Enumerable.Range(1, bg.Count - 1).ToArray();
                    for (int i = 0; i < bls.Length; i++)
                        bls[i].AddRange(cnt.SelectMany(j => bg[j][i]));
                }
                // Gradient Weight
                _biasGrad = bls.Select(g => BestDirection(EMA.RunSeries(g, len))).ToArray();
                // Optimizer
                _biasGrad = _optim[1].Update(_bias, _biasGrad);
                // Max Magnitude
                max = Math.Max(max, _biasGrad.Select(Math.Abs).Max());
            }
            return max;
        }

        /// <summary>
        /// Step parameters from gradient
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="div">Max divisor (Grad Norm)</param>
        public override void Step(float lr, float div = 1)
        {
            div = 1 / div;
            if (_wtGrad == null)    // calculate
                SmoothMax(1);
            var wtGrad = _wtGrad;
            _wtGrad = null;
            // Grad Norm
            if (div != 1)
                wtGrad = wtGrad.Select(l => l.Select(v => v * div).ToArray()).ToArray();
            // Update Weights
            for (int i = 0; i < _wt.Length; i++)
            {
                var wti = _wt[i];
                var wgi = wtGrad[i];
                for (int j = 0; j < wti.Length; j++)
                    wti[j] -= wgi[j] * lr;
            }
            // If bias is used
            if (_bias != null)
            {
                var biasGrad = _biasGrad;
                _biasGrad = null;
                // Grad Norm
                if (div != 1)
                    biasGrad = biasGrad.Select(g => g * div).ToArray();
                // Update Bias
                _bias = _bias.Select((b, i) => b - biasGrad[i] * lr).ToArray();
            }
        }

        /// <summary>
        /// The Same as Matmul() except values are not summed
        /// </summary>
        /// <param name="a">Array a</param>
        /// <param name="b">Array b</param>
        /// <returns>List of values without the sum</returns>
        public static List<float>[][] Mul(float[][] a, float[][] b)
        {
            var m = a.Length;
            var n = a[0].Length;
            var p = b[0].Length;
            // Validate
            if (n != b.Length) throw new Exception("a[0].Length != b.Length");
            var c = new int[m].Select(no => new int[p].Select(n2 => new List<float>(n)).ToArray()).ToArray();
            // Loop a.Length
            for (int i = 0; i < m; i++)
            {
                var ai = a[i];
                var ci = c[i];
                // Loop a[0].Length or b.Length
                for (int k = 0; k < n; k++)
                {
                    var aik = ai[k];
                    var bk = b[k];
                    // Loop b[0].Length
                    for (int j = 0; j < p; j++)
                        // Do not sum
                        ci[j].Add(aik * bk[j]);
                }
            }
            return c;
        }

        /// <summary>
        /// Number of Parameters
        /// </summary>
        /// <returns>Number of Parameters</returns>
        public override long NumParams()
        {
            var nbias = (_bias != null) ? _bias.Length : 0;
            return (long)_wt.Length * _nInput + nbias;
        }

        /// <summary>
        /// Deep Copy of Parameters
        /// </summary>
        /// <returns>Deep Copy of Parameters</returns>
        public override object Parameters()
        {
            var ls = new List<object> { _wt.Select(w => w.ToArray()).ToArray() };   // Deep Copy   
            if (_bias != null)
                ls.Add(_bias.ToArray());                                            // Deep Copy
            return ls;
        }

        /// <summary>
        /// Load State Deep Copy
        /// </summary>
        /// <param name="items">object state</param>
        public override void LoadState(object items)
        {
            var ls = (List<object>)items;
            // Validation
            if (_bias != null && ls.Count != 2) throw new Exception($"There should be 2 set of weights here! got {ls.Count}.");
            else if (_bias == null && ls.Count != 1) throw new Exception($"The should only be 1 set of weights! got {ls.Count}.");
            // Check Weights shape
            var w = (float[][])ls[0];
            if (w.Length != _wt.Length || w[0].Length != _wt[0].Length)
                throw new Exception($"Loaded weight shape is [{w.Length}][{w[0].Length }] should be [{_wt.Length}][{_wt[0].Length }].");
            // Deep copy
            _wt = w.Select(s => s.ToArray()).ToArray();
            // If using bias
            if (_bias != null)
            {
                // Check bias shape
                var b = (float[])ls[1];
                if (b.Length != _bias.Length)
                    throw new Exception($"Loaded bias shape is [{b.Length}] should be [{_bias.Length}].");
                // Deep copy
                _bias = b.ToArray();
            }
        }

        /// <summary>
        /// Load State JSON
        /// </summary>
        /// <param name="items">JSON object</param>
        public override void LoadStateJSON(object items)
        {
            var ls = JsonConvert.DeserializeObject<List<object>>(items.ToString(), _settings);
            ls[0] = JsonConvert.DeserializeObject<float[][]>(ls[0].ToString(), _settings);
            if (ls.Count > 1)
                ls[1] = JsonConvert.DeserializeObject<float[]>(ls[1].ToString(), _settings);
            LoadState(ls);
        }

        /// <summary>
        /// Linear or Fully Connected Deep Layer
        /// </summary>
        /// <param name="nInput">In Features</param>
        /// <param name="nOut">Out Features</param>
        /// <param name="bias">Do you want to use Biases?</param>
        /// <param name="name0">Parents names</param>
        public Linear(int nInput, int nOut, bool bias = true, string name0 = null)
            : base(name0, $"Linear(nInput:{nInput}, nOut:{nOut}, bias:{bias})")
        {
            // Validation
            if (nInput <= 0) throw new Exception("nInput <= 0");
            if (nOut <= 0) throw new Exception("nOut <= 0");
            _inFeatures = _nInput = nInput;
            _outFeatures = nOut;
            // Initialize Weights
            _wt = new int[nOut].Select(no => new float[nInput]).ToArray();
            InitWts(_wt, Math.Sqrt(_nInput));
            // Optimizer
            _optim = new List<Adam> { new Adam(_wt) };
            // If using bias
            if (bias)
            {
                // Initialize Bias
                _bias = new float[nOut];
                InitWts(_bias);
                // Optimizer
                _optim.Add(new Adam(_bias));
                _biasGradls = new ConcurrentBag<List<float>[]>();
            }
            _wtGradls = new ConcurrentBag<List<float>[][]>();
        }
    }
}
