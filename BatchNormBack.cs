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
    /// Batch Norm on the last dimention.
    /// Input can be float[2d or 3d]
    /// </summary>
    public class BatchNormBack : Module
    {
        // Vars
        public readonly int _dim, _nSeq, _dim0;
        // Parameters
        float[] _avg, _std;
        // Gradient
        ConcurrentBag<float[][]> _backVals;

        /// <summary>
        /// Back Norm Forwards pass
        /// </summary>
        /// <param name="x">Input Tensor</param>
        /// <returns>Same shape as Tensor 'x'</returns>
        public Tensor<float[][]> Forward(Tensor<float[][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward2d, true);
        }
        /// <summary>
        /// Back Norm Forwards pass
        /// </summary>
        /// <param name="x">Input Tensor</param>
        /// <returns>Same shape as Tensor 'x'</returns>
        public Tensor<float[][][]> Forward(Tensor<float[][][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward3d, true);
        }
        /// <summary>
        /// Internal Forward pass
        /// </summary>
        /// <param name="x">input</param>
        /// <param name="ctx">Context for backwards pass.</param>
        /// <returns>Same shape as Tensor 'x'</returns>
        public float[][] Forward(float[][] x, out object ctx)
        {
            // Validate Shape
            if (x[0].Length != _dim)
                throw new Exception($"Invalid shape of inputs: got (batch, {x[0].Length}) should be (batch, {_dim}).");
            // Output Clamped
            var outsclamp = new float[x.Length][];
            // Output before Clamp
            var y = new float[x.Length][];
            // Loop batch
            for (int i = 0; i < x.Length; i++)
            {
                var xi = x[i];
                var oc = outsclamp[i] = new float[_dim];
                var yi = y[i] = new float[_dim];
                // Loop dim
                for (int j = 0; j < _dim; j++)
                {
                    // Norm
                    var oj = yi[j] = (xi[j] - _avg[j]) / _std[j];
                    // Apply clamp if needed
                    if (Math.Abs(oj) > 3)
                        oj = Math.Sign(oj) * 3;
                    oc[j] = oj;
                }
            }
            ctx = new[] { x, y };
            return outsclamp;
        }
        /// <summary>
        /// Internal Forward pass
        /// </summary>
        /// <param name="x">input</param>
        /// <param name="ctx">Context for backwards pass.</param>
        /// <returns>Same shape as Tensor 'x'</returns>
        public float[][][] Forward(float[][][] x_, out object ctx_)
        {
            // Validate Shape
            if (x_[0].Length != _nSeq || x_[0][0].Length != _dim0)
                throw new Exception($"Invalid input shape: got (batch, {x_[0].Length}, {x_[0][0].Length}) " +
                            $"it should be (batch, {_nSeq}, {_dim0}).");
            // Flatten last 2 dimentions
            var x = x_.Select(s => s.SelectMany(l => l).ToArray()).ToArray();
            // Forward pass 2d
            var y = Forward(x, out ctx_);
            // Reshape to 3d
            return Utilz.Reshape(y, y.Length, _nSeq, _dim0);
        }

        /// <summary>
        /// Backwards Pass 2d
        /// </summary>
        /// <param name="y">Output Node of Forwards pass.</param>
        void Backward2d(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }
        /// <summary>
        /// Backwards Pass 3d
        /// </summary>
        /// <param name="y">Output Node of Forwards pass.</param>
        void Backward3d(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }

        /// <summary>
        /// Internal Backwards pass.
        /// </summary>
        /// <param name="grad">Output gradient</param>
        /// <param name="ctx">Context for backwards pass.</param>
        /// <param name="requiresInGrad">If input gradient is required</param>
        /// <returns>Input Gradient if requiresInGrad is True</returns>
        public float[][] Backward(float[][] grad, object ctx, bool requiresInGrad = true)
        {
            var s = (float[][][])ctx;
            var x = s[0];       // Input in forwards pass
            var y = s[1];       // output in forwards pass before clamp
            // Validate shape
            if (grad.Length != x.Length || grad[0].Length != _dim)
                throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}) should be ({x.Length}, {_dim}).");
            // Input gradient
            float[][] inpgrad = null;
            if (requiresInGrad)
            {
                inpgrad = new float[grad.Length][];
                // Loop batch
                for (int i = 0; i < grad.Length; i++)
                {
                    var yi = y[i];
                    var pi = inpgrad[i] = new float[_dim];
                    var gi = grad[i];
                    // Loop dim
                    for (int j = 0; j < _dim; j++)
                    {
                        var yy = yi[j];
                        var g = gi[j];
                        if (yy < -3)
                            g = -Math.Abs(g);
                        else if (yy > 3)
                            g = Math.Abs(g);
                        pi[j] = g / _std[j];
                    }
                }
            }
            // Multithreaded
            _backVals.Add(x);
            return inpgrad;
        }
        /// <summary>
        /// Internal Backwards pass.
        /// </summary>
        /// <param name="grad">Output gradient</param>
        /// <param name="ctx">Context for backwards pass.</param>
        /// <param name="requiresInGrad">If input gradient is required</param>
        /// <returns>Input Gradient if requiresInGrad is True</returns>
        public float[][][] Backward(float[][][] grad_, object ctx_, bool requiresInGrad = true)
        {
            // Validate Shape
            if (grad_[0].Length != _nSeq || grad_[0][0].Length != _dim0)
                throw new Exception($"Invalid gradient shape: got (batch, {grad_[0].Length}, {grad_[0][0].Length}) " +
                            $"it should be (batch, {_nSeq}, {_dim0}).");
            // Flatten last 2 dims
            var grad = grad_.Select(g => g.SelectMany(l => l).ToArray()).ToArray();
            var ingrad = Backward(grad, ctx_, requiresInGrad);
            if (!requiresInGrad) return null;
            return Utilz.Reshape(ingrad, ingrad.Length, _nSeq, _dim0);
        }

        /// <summary>
        /// Parameters
        /// </summary>
        /// <returns>Parameters</returns>
        public override object Parameters()
        => new[] { _avg.ToArray(), _std.ToArray() };

        /// <summary>
        /// Returns Zero.
        /// I don't could batch norm vars in the number of parameters
        /// </summary>
        /// <returns></returns>
        public override long NumParams()
        //=> _avg.Length + _std.Length;
            => 0;

        /// <summary>
        /// Load State
        /// </summary>
        /// <param name="items">The state</param>
        public override void LoadState(object items)
        {
            var w = (float[][])items;
            if (w.Length != 2 || w[0].Length != _dim || w[1].Length != _dim)
                throw new Exception($"Invalid shape of weights: got ({w.Length}, {w[0].Length}-{w[1].Length}) " +
                    $"should be (2, {_dim}-{_dim}).");
            _avg = w[0].ToArray();
            _std = w[1].ToArray();
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
        /// Not needed.
        /// </summary>
        /// <param name="len"></param>
        /// <returns></returns>
        public override float SmoothMax(int len)
        => float.NegativeInfinity;

        /// <summary>
        /// Step parameters using gradient
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="div">Max divisor (Grad Norm)</param>
        public override void Step(float lr, float div = 1)
        {
            // Validation
            if (_backVals.Count == 0)
                throw new Exception("There are no gradients. You need to call Backward() first.");
            // Reduce
            var vals = _backVals.SelectMany(s => s).ToArray();
            vals = Utilz.TransposeArray(vals);
            _backVals = new ConcurrentBag<float[][]>();
            // Calc Target Average
            var avg = vals.Select(s => s.Average()).ToArray();
            // Calc Target Standard Deviation
            var std = vals.Select((s, i) =>
            {
                var a = avg[i];
                var varr = s.Select(v => (float)Math.Pow(v - a, 2)).Average();
                return (float)Math.Sqrt(varr + 1e-12);
            }).ToArray();
            // Calc Gradient
            var avggrad = new float[_dim];
            var stdgrad = new float[_dim];
            for (int i = 0; i < _dim; i++)
            {
                avggrad[i] = _avg[i] - avg[i];
                stdgrad[i] = _std[i] - std[i];
            }
            // Grad Norm
            var max = Math.Max(avggrad.Select(Math.Abs).Max(), stdgrad.Select(Math.Abs).Max());
            if (max > 1)
            {
                avggrad = avggrad.Select(v => v / max).ToArray();
                stdgrad = stdgrad.Select(v => v / max).ToArray();
            }
            // Step
            for (int i = 0; i < _dim; i++)
            {
                _avg[i] -= avggrad[i] * lr;
                _std[i] -= stdgrad[i] * lr;
                // Remove undef
                if (_std[i] == 0)
                    _std[i] = 1e-9f;
            }
        }

        /// <summary>
        /// Batch Norm on the last dimention.
        /// Input can be float[2d or 3d]
        /// </summary>
        /// <param name="nInput">In features size</param>
        /// <param name="nSeq">In Sequence if defined input is float[3d] else float[2d]</param>
        /// <param name="name0">Parents Names</param>
        public BatchNormBack(int nInput, int nSeq = -1, string name0 = null)
            : base(name0, $"BatchNormBack(nInput: {nInput}, nSeq:{nSeq})")
        {
            _dim = nInput;
            // If defined input is float[3d] else float[2d]
            // If defined, last 2 dims of float[3d] are flattened then unflattend.
            if (nSeq > 0)
            {
                _dim *= nSeq;
                _nSeq = nSeq;
                _dim0 = nInput;
            }
            _avg = new float[_dim];
            _std = new bool[_dim].Select(no => 1f).ToArray();
            _backVals = new ConcurrentBag<float[][]>();
        }
    }
}
