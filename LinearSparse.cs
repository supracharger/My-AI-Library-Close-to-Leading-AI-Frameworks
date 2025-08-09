using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{   
    /// <summary>
    /// Stacked Linear Layers by size 'dim'
    /// </summary>
    public class LinearSparse : Module
    {
        // vars
        public readonly int _inFeatures, _outFeatures, _dim;
        // Linear Layers
        readonly List<Linear> _fcs;

        /// <summary>
        /// Forwards Pass for Stacked Linear Layers by size '_dim'
        /// </summary>
        /// <param name="x">Shape[_dim][batch][_inFeatures]</param>
        /// <returns>Shape[_dim][batch][_outFeatures]</returns>
        public Tensor<float[][][]> Forward(Tensor<float[][][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }
        /// <summary>
        /// Forwards Pass for Stacked Linear Layers by size '_dim'
        /// </summary>
        /// <param name="x">Shape[_dim][batch][_inFeatures]</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <returns>Shape[_dim][batch][_outFeatures]</returns>
        public float[][][] Forward(float[][][] x, out object ctx_)
        {
            // Validate Shape
            if (x.Length != _dim || x[0][0].Length != _inFeatures)
                throw new Exception($"Invalid Shape of Inputs: got ({x.Length}, batch, {x[0][0].Length}) " +
                                    $"should be ({_dim}, batch, {_inFeatures})");
            var ctx = new List<object>(_fcs.Count);
            ctx_ = ctx;
            // Loop each Sequence by size dim
            return x.Select((s, i) =>
            {
                var o = _fcs[i].Forward(s, out object obj);
                ctx.Add(obj);
                return o;
            }).ToArray();
        }

        /// <summary>
        /// Inner Backwards Function
        /// </summary>
        /// <param name="y">Output node from forwards pass</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }
        /// <summary>
        /// Inner Backwards Function
        /// </summary>
        /// <param name="outgrad">Output Gradient</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <param name="requiresInGrad">If requires input gradient</param>
        /// <returns>Input Gradient</returns>
        public float[][][] Backward(float[][][] outgrad, object ctx_, bool requiresInGrad = true)
        {
            // Validate Shape
            if (outgrad.Length != _dim || outgrad[0][0].Length != _outFeatures)
                throw new Exception($"Invalid gradient shape: got ({outgrad.Length}, batch, {outgrad[0][0].Length}) " +
                            $"should be ({_dim}, batch, {_outFeatures}).");
            // Validate context size
            var ctx = (List<object>)ctx_;
            if (ctx.Count != _dim)
                throw new Exception($"ctx.Count({ctx.Count}) != _dim({_dim})");
            var vix = 0;
            // Backwards pass
            var ingrad = outgrad.Select((g, i) => _fcs[i].Backward(g, ctx[vix++], requiresInGrad)).ToArray();
            // validation
            if (vix != ctx.Count) throw new Exception($"vix({vix}) != ctx.Count({ctx.Count})");
            return ingrad;
        }

        /// <summary>
        /// Stacked Linear Layers by size '_dim'
        /// </summary>
        /// <param name="nInput">In Features</param>
        /// <param name="dim">Linear layers to stack. Usually sequence length.</param>
        /// <param name="nOut">Out Features</param>
        /// <param name="bias">If using bias</param>
        /// <param name="name0">Parents Names</param>
        public LinearSparse(int nInput, int dim, int nOut, bool bias = true, string name0 = null)
            :base(name0, $"LinearSparse(nInput:{nInput}, dim:{dim}, nOut:{nOut}, bias:{bias})")
        {
            _inFeatures = nInput;
            _dim = dim;
            _outFeatures = nOut;
            // Stacked Linear layers by dim
            _fcs = new bool[dim].Select(no => new Linear(nInput, nOut, bias, name0: ToString())).ToList();
            _modules.AddRange(_fcs);
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
