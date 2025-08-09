using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Multiple Conv1d's are stacked ontop of each other over a sequence.
    /// </summary>
    public class Conv1dSparse : Module
    {
        // vars
        public readonly int _inFeatures, _outFeatures, _nSeq, _channels, _dim1;
        // stack
        readonly List<Conv1d> _cs;

        /// <summary>
        /// Forwards Pass
        /// </summary>
        /// <param name="x">Shape Tensor(float[batch][nSeq][_inFeatures])</param>
        /// <returns>Shape Tensor(float[batch][nSeq][_outFeatures])</returns>
        public Tensor<float[][][]> Forward(Tensor<float[][][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }

        /// <summary>
        /// Internal Forwards Pass
        /// </summary>
        /// <param name="x">Shape float[batch][nSeq][_inFeatures]</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <param name="transposed">If first 2 dims are transposed</param>
        /// <returns>Shape float[batch][nSeq][_outFeatures]</returns>
        public float[][][] Forward(float[][][] x, out object ctx_, bool transposed = false)
        {
            // If not transposed: Validate Shape
            if (!transposed && (x[0].Length != _nSeq || x[0][0].Length != _inFeatures))
                throw new Exception($"Invalid input shape: got ({x.Length}, {x[0].Length}, {x[0][0].Length}) " +
                        $"it should be (batch, {_nSeq}, {_inFeatures}).");
            // If transposed: Validate Shape
            if (transposed && (x.Length != _nSeq || x[0][0].Length != _inFeatures))
                throw new Exception($"Invalid input shape: got ({x.Length}, {x[0].Length}, {x[0][0].Length}) " +
                        $"it should be ({_nSeq}, batch, {_inFeatures}).");
            var ctx = new List<object>(_nSeq + 1);
            ctx_ = ctx;
            if (!transposed)                    // Transpose
                x = Utilz.TransposeArray(x);
            ctx.Add(x[0].Length);              // batch size
            // Conv1d for each Sequence
            var y0 = new float[_nSeq][][][];
            for (int i = 0; i < _nSeq; i++)
            {
                // Conv1d Forwards Pass
                y0[i] = _cs[i].Forward(x[i], out object ct);
                ctx.Add(ct);
            }
            // Flatten last dimentions
            var y = y0.Select(yi => yi.Select(yj => yj.SelectMany(yk => yk).ToArray()).ToArray()).ToArray();
            return Utilz.TransposeArray(y);     // Transpose
        }

        /// <summary>
        /// Internal Backwards Pass
        /// </summary>
        /// <param name="y">Output node of forwards pass.</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }

        /// <summary>
        /// Internal Backwards Pass 
        /// </summary>
        /// <param name="grad">Output gradient shape float[batch][nSeq][_outFeatures]</param>
        /// <param name="ctx_">Context for Backwards pass</param>
        /// <param name="requiresInGrad">If in grad is required</param>
        /// <returns>In grad shape float[batch][nSeq][_inFeatures]</returns>
        public float[][][] Backward(float[][][] grad, object ctx_, bool requiresInGrad = true)
        {
            var ctx = (List<object>)ctx_;
            // Validate Context Count
            if (ctx.Count != _nSeq + 1)
                throw new Exception($"ctx.Count({ctx.Count}) != {_nSeq + 1}");
            // Validate Shape
            var batch = (int)ctx[0];
            if (grad.Length != batch || grad[0].Length != _nSeq || grad[0][0].Length != _outFeatures)
                throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                        $"it should be ({batch}, {_nSeq}, {_outFeatures}).");
            var vix = 1;
            grad = Utilz.TransposeArray(grad);                              // Transpose
            var g0 = Utilz.Reshape(grad, _nSeq, batch, _channels, _dim1);   // Reshape for outgrad of each conv1d
            // Loop each Conv1d in Sequence
            var ingrad = new float[_nSeq][][];
            for (int i = 0; i < _nSeq; i++)
                ingrad[i] = _cs[i].Backward(g0[i], ctx[vix++], requiresInGrad);
            // Validate
            if (vix != ctx.Count) throw new Exception($"vix({vix}) != ctx.Count({ctx.Count})");
            // Return In Grad
            if (requiresInGrad)
                return Utilz.TransposeArray(ingrad);
            // Do not Return In Grad
            else
                return null;
        }

        /// <summary>
        /// Multiple Conv1dSparse are stacked ontop of each other.
        /// </summary>
        /// <param name="nInput">In Features size</param>
        /// <param name="channels">Number of channels</param>
        /// <param name="len">Len of inputs for each node in Conv1d</param>
        /// <param name="stride">The stride</param>
        /// <param name="nSequence">Sequence length. (The number of Conv1d's to stack)</param>
        /// <param name="padding">The padding</param>
        /// <param name="name0">Parents names</param>
        public Conv1dSparse(int nInput, int channels, int len, int stride, int nSequence, int padding = 0, string name0 = null)
            : base(name0, $"Conv1dSparse(nInput:{nInput}, channels:{channels}, len:{len}, stride:{stride}, " +
                            $"nSequence:{nSequence}, padding:{padding})")
        {
            _inFeatures = nInput;
            _nSeq = nSequence;
            // Conv1d's stacked by sequence length
            _cs = new bool[nSequence].Select(no =>
                    new Conv1d(nInput, channels, len, stride, padding, name0: ToString())).ToList();
            _outFeatures = channels * _cs[0]._outFeatures;
            _channels = channels;
            _dim1 = _cs[0]._outFeatures;
            _modules.AddRange(_cs);
            GetAllModules(_modules);
        }

        // These functions would never be called if used correctly. Be sure to use GetAllModules(_modules).
        // Since this is a module of modules these functions should not be called.
        public override object Parameters()
        { throw new Exception("Did you forget to call GetAllModules(_modules) at the end of the constructor?"); }
        public override void LoadState(object items)
        { throw new Exception("Did you forget to call GetAllModules(_modules) at the end of the constructor?"); }
        public override void LoadStateJSON(object items)
        { throw new Exception("Did you forget to call GetAllModules(_modules) at the end of the constructor?"); }
        public override long NumParams()
        { throw new Exception("Did you forget to call GetAllModules(_modules) at the end of the constructor?"); }
        public override float SmoothMax(int len)
        { throw new Exception("Did you forget to call GetAllModules(_modules) at the end of the constructor?"); }
        public override void Step(float lr, float div = 1)
        { throw new Exception("Did you forget to call GetAllModules(_modules) at the end of the constructor?"); }
    }
}
