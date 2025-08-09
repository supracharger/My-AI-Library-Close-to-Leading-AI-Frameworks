using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace MiniNet
{
    /// <summary>
    /// My Interesting interpretation of combining a Transformer & Convolutional 1d layer.
    /// </summary>
    public class TransConv1d : Module
    {
        // vars
        public readonly int _nInput, _inFeatures, _outFeatures, _nChannelHeads;
        // Layers
        readonly Conv1d _qc, _vc, _kc;
        readonly BatchNormBack _norm;

        /// <summary>
        /// Forwards Pass
        /// </summary>
        /// <param name="x">Input Tensor to Layer</param>
        /// <returns>Output Tensor of Layer</returns>
        public Tensor<float[][][]> Forward(Tensor<float[][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }

        /// <summary>
        /// Inside Forwards pass
        /// </summary>
        /// <param name="x">Input into Layer</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <returns>Output of Layer</returns>
        public float[][][] Forward(float[][] x, out object ctx_)
        {
            var ctx = new List<object>();
            ctx_ = ctx;
            object obj;
            // If using batch norm
            if (_norm != null)
            {
                x = _norm.Forward(x, out obj);
                ctx.Add(obj);
            }
            // Tranformer head 1
            var h1 = _qc.Forward(x, out obj);                               // (batch, hc, i)
            ctx.Add(obj);
            // Tranformer head 2
            var h2 = Utilz.Transpose(_kc.Forward(x, out obj), 1, 2);        // (batch, i, hc)
            ctx.Add(obj);
            // Tranformer Scores: head 1 & 2
            var scores = Function.Matmul.Fwd(h1, h2, out obj);
            ctx.Add(obj);
            // Activation Function
            scores = Function.Sqrt.Fwd(scores, out obj);                    // (batch, hc, hc)
            ctx.Add(obj);
            // Tranformer value head
            var hv = _vc.Forward(x, out obj);                               // (batch, hc, i)
            ctx.Add(obj);
            // Scores X value head
            var y = Function.Matmul.Fwd(scores, hv, out obj);               // (batch, hc, i)
            ctx.Add(obj);
            return y;
        }

        /// <summary>
        /// Inside Backwards Function
        /// </summary>
        /// <param name="y">Output node of forward pass</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }

        /// <summary>
        /// Inside Backward pass
        /// </summary>
        /// <param name="grad">Output gradient</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <param name="requiresInGrad">If requires input gradient</param>
        /// <returns>Input Gradient</returns>
        public float[][] Backward(float[][][] grad, object ctx_, bool requiresInGrad = true)
        {
            var ctx = (List<object>)ctx_;
            var vix = ctx.Count - 1;
            var useNorm = _norm != null;
            float[][][] gradScores, gradV,
                        gradQ, gradK;
            // Validate Context
            if (ctx.Count - (useNorm ? 1 : 0) != 6)
                throw new Exception($"Incorrect length of backwars 'ctx': got ({ctx.Count}) should be ({6 + (useNorm ? 1 : 0)}).");
            // Validate Shape
            if (grad[0].Length != _nChannelHeads || grad[0][0].Length != _outFeatures)
                throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                                    $"should be (batch, {_nChannelHeads}, {_outFeatures}).");
            var requiresInGradNorm = useNorm || requiresInGrad;
            // Gradient for Scores & value head
            Function.Matmul.Bck(grad, out gradScores, out gradV, ctx[vix--]);
            var gradV2 = _vc.Backward(gradV, ctx[vix--], requiresInGradNorm);
            // Activation Function
            gradScores = Function.Sqrt.Bck(gradScores, ctx[vix--]);
            // Gradient for head Q & K
            Function.Matmul.Bck(gradScores, out gradQ, out gradK, ctx[vix--]);
            // Gradient for head K
            var gradK2 = _kc.Backward(Utilz.Transpose(gradK, 1, 2), ctx[vix--], requiresInGradNorm);
            // Gradient for head Q
            var gradQ2 = _qc.Backward(gradQ, ctx[vix--], requiresInGradNorm);
            // If requires in grad or using BatchNorm
            var ingrad = gradV2;
            if (requiresInGradNorm)
            {
                // Gradint sum Tranformer Heads
                for (int i = 0; i < ingrad.Length; i++)
                {
                    var k = gradK2[i];
                    var q = gradQ2[i];
                    var g = ingrad[i];
                    // Gradient Sum
                    for (int j = 0; j < g.Length; j++)
                        g[j] += k[j] + q[j];
                }
                // If using batch Norm
                if (useNorm)
                    ingrad = _norm.Backward(ingrad, ctx[vix--], requiresInGrad);
            }
            // Validate
            if (vix != -1) throw new Exception("vix != -1");
            return ingrad;
        }

        /// <summary>
        /// My Interesting interpretation of combining a Transformer & Convolutional 1d layer
        /// </summary>
        /// <param name="nInput">In Features</param>
        /// <param name="nChannelHeads">Convolutional Channels & Tranformer heads Combined into one</param>
        /// <param name="len">Length of Convolutional Span</param>
        /// <param name="stride">Stride for Convolution</param>
        /// <param name="padding">Padding for Convolution</param>
        /// <param name="batchNorm">If using BatchNorm</param>
        /// <param name="name0">Names of Parents</param>
        public TransConv1d(int nInput, int nChannelHeads, int len, int stride, int padding = 0, bool batchNorm = true,
            string name0 = null)
            :base(name0, $"TransConv1d(nInput:{nInput}, nChannelHeads:{nChannelHeads}, len:{len}, stride:{stride}, " +
                 $"padding:{padding}, batchNorm:{batchNorm})")
        {
            _inFeatures = _nInput = nInput;
            _nChannelHeads = nChannelHeads;
            // Transformer heads
            _qc = new Conv1d(nInput, nChannelHeads, len, stride, padding, name0:ToString());
            _vc = new Conv1d(nInput, nChannelHeads, len, stride, padding, name0:ToString());
            _kc = new Conv1d(nInput, nChannelHeads, len, stride, padding, name0:ToString());
            _modules.AddRange(new[] { _qc, _vc, _kc });
            _outFeatures = _qc._outFeatures;
            // If using batchnorm
            if (batchNorm)
            {
                _norm = new BatchNormBack(nInput, name0:ToString());
                _modules.Add(_norm);
            }
            // Make sure this is always called at the end of the constructor!
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
