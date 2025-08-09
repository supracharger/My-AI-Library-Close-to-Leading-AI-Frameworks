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
    /// My version of convolution 1D.
    /// Convolution is usually used for images, but I'm using it in place of a fully connected layer.
    /// </summary>
    public class Conv1d : Module
    {
        // vars
        readonly int _len, _stride, _size, _channels, _pad;
        public readonly int _inFeatures, _outFeatures;
        // Weight & Gradients
        float[][][] _wt, _wtGrad;
        ConcurrentBag<List<float>[][][]> _wtGradls;
        // Optimizer
        readonly Adam _optim;

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="x">Input node</param>
        /// <returns>Shape of Tensor(float[batch][_channels][_outFeatures])</returns>
        public Tensor<float[][][]> Forward(Tensor<float[][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }

        /// <summary>
        /// Internal Forward pass.
        /// </summary>
        /// <param name="x">Input node</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <returns>Shape of float[batch][_channels][_outFeatures]</returns>
        public float[][][] Forward(float[][] x, out object ctx)
        {
            // Validate Shape
            if (x[0].Length != _inFeatures)
                throw new Exception($"Invalid shape of inputs. Should be [batch][{_inFeatures}] got [{x.Length}][{x[0].Length}]");
            // If uses padding, add it
            if (_pad > 0)
                x = x.Select(s => s.Concat(new float[_pad]).ToArray()).ToArray();
            ctx = x;
            var outs = new float[x.Length][][];
            // My Average Activation Function
            var div = 1 / (float)Math.Sqrt(_len);
            // Loop batch
            for (int i = 0; i < x.Length; i++)
            {
                var s = x[i];
                var ooo = outs[i] = new float[_channels][];
                // Loop Channels
                for (int c = 0; c < _channels; c++)
                {
                    var www = _wt[c];
                    var oc = ooo[c] = new float[_size];
                    var jjj = 0;
                    // Loop Out Features
                    for (int l = 0; l < _size; l++)
                    {
                        var w = www[l];
                        var j = jjj;
                        var ensum = 0f;
                        for (int k = 0; k < w.Length; k++)
                            ensum += w[k] * s[j++];
                        // Output
                        oc[l] = ensum * div;
                        // Shift by stride
                        jjj += _stride;
                    }
                }
            }
            return outs;
        }

        /// <summary>
        /// Internal Backwards pass
        /// </summary>
        /// <param name="y">Forward Output node.</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }
        /// <summary>
        /// Internal Backwards pass
        /// </summary>
        /// <param name="grad">Output Gradient</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <param name="requiresInGrad">If in grad is needed</param>
        /// <returns>Shape[batch][_inFeatures]</returns>
        public float[][] Backward(float[][][] grad, object ctx, bool requiresInGrad = true)
        {
            // grad[batch][channels][size]
            // Validate Shape
            var x = (float[][])ctx;
            if (grad.Length != x.Length || grad[0].Length != _channels || grad[0][0].Length != _size)
                throw new Exception($"Invalid shape. Should be [{x.Length}][{_channels}][{_size}] got [{grad.Length}][{grad[0].Length}][{grad[0][0].Length}]");
            // My Average Activation Function
            var div = 1 / (float)Math.Sqrt(_len);
            MultiplyInPlace(grad, div);
            //grad = grad.Select(g => g.Select(s => s.Select(v => v * div).ToArray()).ToArray()).ToArray();
            var ln2 = x[0].Length;
            // Weight Grad
            var wg = new int[_channels].Select(n3 => new int[_size].Select(no =>
                    new int[_len].Select(n2 => new List<float>(grad.Length)).ToArray()).ToArray()).ToArray();
            // Needs In Grad .......................................
            float[][] og = null;
            if (requiresInGrad)
            {
                og = new int[x.Length].Select(no => new float[ln2]).ToArray();
                // Loop batch
                for (int b = 0; b < grad.Length; b++)
                {
                    var ggg = grad[b];
                    var xxx = x[b];
                    var oogg = og[b];
                    // Loop channels
                    for (int c = 0; c < _channels; c++)
                    {
                        var ggcc = ggg[c];
                        var www = _wt[c];
                        var wggg = wg[c];
                        var jjj = 0;
                        // Loop out features
                        for (int si = 0; si < _size; si++)
                        {
                            var g = ggcc[si];
                            var w = www[si];
                            var wtGrad = wggg[si];
                            var j = jjj;
                            // Loop len
                            for (int k = 0; k < _len; k++)
                            {
                                wtGrad[k].Add(xxx[j] * g);  // gradient
                                oogg[j] += w[k] * g;        // Input grad
                                j++;
                            }
                            // Shift by stride
                            jjj += _stride;
                        }
                    }
                }
            }
            // Does Not need In Grad .............................
            else
            {
                // Loop batch
                for (int b = 0; b < grad.Length; b++)
                {
                    var ggg = grad[b];
                    var xxx = x[b];
                    // Loop channels
                    for (int c = 0; c < _channels; c++)
                    {
                        var ggcc = ggg[c];
                        var www = _wt[c];
                        var wggg = wg[c];
                        var jjj = 0;
                        // Loop out features
                        for (int si = 0; si < _size; si++)
                        {
                            var g = ggcc[si];
                            var w = www[si];
                            var wtGrad = wggg[si];
                            var j = jjj;
                            // Loop len
                            for (int k = 0; k < _len; k++)
                            {
                                wtGrad[k].Add(xxx[j] * g);      // Gradient
                                j++;
                            }
                            // Shift by stride
                            jjj += _stride;
                        }
                    }
                }
            }
            // Multithreading
            _wtGradls.Add(wg);
            // If requires in grad & uses padding
            if (_pad > 0 && requiresInGrad)
            {
                var take = ln2 - _pad;
                og = og.Select(l => l.Take(take).ToArray()).ToArray();
            }
            return og;
        }

        /// <summary>
        /// Smooth, Get BestDirection for grad & call optimizer
        /// </summary>
        /// <param name="len">Smooth Length</param>
        /// <returns>The largest Magnitude</returns>
        public override float SmoothMax(int len)
        {
            // Validation
            if (_wtGradls.Count == 0)
                throw new Exception("There are no gradients. You need to call Backward() first.");
            // Reduce
            var wgl = _wtGradls.ToList();
            _wtGradls = new ConcurrentBag<List<float>[][][]>();
            var wtGradls = wgl[0];
            if (wgl.Count > 1)
            {
                var cnt = Enumerable.Range(1, wgl.Count-1).ToArray();
                for (int i = 0; i < wtGradls.Length; i++)
                {
                    var wgi = wtGradls[i];
                    for (int j = 0; j < wgi.Length; j++)
                    {
                        var wg = wgi[j];
                        for (int k = 0; k < wg.Length; k++)
                            wg[k].AddRange(cnt.SelectMany(l => wgl[l][i][j][k]));
                    }
                }
            }
            // Weight Gradient
            _wtGrad = wtGradls.Select(c => c.Select(s => s.Select(l => BestDirection(EMA.RunSeries(l, len))).ToArray()).ToArray()).ToArray();
            // Optimizer
            _wtGrad = _optim.Update(_wt, _wtGrad);
            // Get Max Gradient Magnitude
            return _wtGrad.SelectMany(g => g.SelectMany(s => s)).Select(Math.Abs).Max();
        }

        /// <summary>
        /// Step parameters using gradient.
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="div">Max Divisor (Grad Norm)</param>
        public override void Step(float lr, float div = 1)
        {
            div = 1 / div;
            if (_wtGrad == null)    // Need to calculate
                SmoothMax(1);
            var wtGrad = _wtGrad;
            _wtGrad = null;
            // Grad Norm
            if (div != 1)
                wtGrad = wtGrad.Select(g => g.Select(s => s.Select(v => v * div).ToArray()).ToArray()).ToArray();
            // Step .........
            for (int i = 0; i < _wt.Length; i++)
            {
                var w = _wt[i];
                var g = wtGrad[i];
                for (int j = 0; j < w.Length; j++)
                {
                    var ww = w[j];
                    var gg = g[j];
                    // Step
                    for (int k = 0; k < ww.Length; k++)
                        ww[k] -= gg[k] * lr;
                }
            }
        }

        /// <summary>
        /// Number of Parameters
        /// </summary>
        /// <returns>Number of Parameters</returns>
        public override long NumParams()
        => (long)_channels * _size * _len;

        /// <summary>
        /// Deep copy of Parameters
        /// </summary>
        /// <returns>Deep copy of Parameters</returns>
        public override object Parameters()
            => _wt.Select(w => w.Select(ww => ww.ToArray()).ToArray()).ToArray();    // Deep Copy

        /// <summary>
        /// Load State Parameters
        /// </summary>
        /// <param name="items">object with parameters</param>
        public override void LoadState(object items)
        {
            var w = (float[][][])items;
            // Validate Shape
            if (w.Length != _wt.Length || w[0].Length != _wt[0].Length || w[0][0].Length != _wt[0][0].Length)
                throw new Exception($"Loaded weight shape is [{w.Length}][{w[0].Length }][{w[0][0].Length}] should be [{_wt.Length}][{_wt[0].Length }][{_wt[0][0].Length}].");
            // Deep copy Parameters
            _wt = w.Select(s => s.Select(ss => ss.ToArray()).ToArray()).ToArray();
        }

        /// <summary>
        /// Load State JSON
        /// </summary>
        /// <param name="items">JSON object</param>
        public override void LoadStateJSON(object items)
        {
            var w = JsonConvert.DeserializeObject<float[][][]>(items.ToString(), _settings);
            LoadState(w);
        }

        /// <summary>
        /// My version of convolution 1D.
        /// Convolution is usually used for images, but I'm using it in place of a fully connected layer.
        /// </summary>
        /// <param name="n">Number of in Features</param>
        /// <param name="channels">Number of Channels</param>
        /// <param name="len">Length of inputs in each node</param>
        /// <param name="stride">The Stride</param>
        /// <param name="padding">The Padding</param>
        /// <param name="name0">Parent(s) names</param>
        public Conv1d(int n, int channels, int len, int stride, int padding = 0, string name0 = null)
            : base(name0, $"Conv1d(n:{n}, channels:{channels}, len:{len}, stride:{stride}, padding:{padding})")
        {
            if (n <= 0) throw new Exception("n <= 0");
            if (channels <= 0) throw new Exception("channels <= 0");
            if (len <= 0) throw new Exception("len <= 0");
            if (stride <= 0) throw new Exception("stride <= 0");
            if (padding < 0) throw new Exception("padding < 0");
            _inFeatures = n;
            _channels = channels;
            _len = len;
            _stride = stride;
            _pad = padding;
            var s0 = ((double)(n - len + padding) / stride) + 1;
            var size = (int)s0;
            if (s0 != size)
                throw new Exception($"Incompatable dimensions using n={n} len={len} stride={stride} padding={padding}.");
            //var size = 1;
            //var cnt = 0;
            //n += padding;
            //while (cnt + len < n)
            //{
            //    cnt += stride;
            //    size++;
            //}
            //if (cnt + len > n) throw new Exception("cnt + len > n + padding");
            // Init Weights
            _wt = new int[channels].Select(n2 => new int[size].Select(no => new float[len]).ToArray()).ToArray();
            InitWts(_wt, Math.Sqrt(_len));
            _outFeatures = _size = size;
            // Optimizer
            _optim = new Adam(_wt);
            _wtGradls = new ConcurrentBag<List<float>[][][]>();
        }
    }
}
