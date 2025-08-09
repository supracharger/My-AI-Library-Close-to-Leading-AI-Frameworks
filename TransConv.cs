using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// My Transformer Convolutional 1d Layer
    /// Very close to a Transformer layer, but using Conv1d layers instead of fully connected layers
    /// </summary>
    public class TransConv : Module
    {
        // Vars
        public readonly int _inFeatures, _nSeq, _heads, _outFeatures, _dk;
        // Layers
        readonly Conv1dSparse _qc, _vc, _kc;
        readonly BatchNormBack _norm;
        readonly Linear _fc;

        /// <summary>
        /// Forwards Pass
        /// </summary>
        /// <param name="x">Tensor input to layer</param>
        /// <returns>Tensor Network output</returns>
        public Tensor<float[][][]> Forward(Tensor<float[][][]> x)
        {
            var y = Forward(x._data, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }

        /// <summary>
        /// Inside Forwards pass
        /// </summary>
        /// <param name="x">Input to layer</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <returns>Output of layer</returns>
        public float[][][] Forward(float[][][] x, out object ctx_)
        {
            var ctx = new List<object>();
            ctx_ = ctx;
            object obj;
            // validation
            if (x[0].Length != _nSeq || x[0][0].Length != _inFeatures)
                throw new Exception($"Invalid input shape: got ({x.Length}, {x[0].Length}, {x[0][0].Length}) " +
                        $"it should be (batch, {_nSeq}, {_inFeatures}).");
            // Add batch size
            ctx.Add(x.Length);
            // If using BatchNorm
            if (_norm != null)
            {
                x = _norm.Forward(x, out obj);
                ctx.Add(obj);
            }
            // Transformer head 1
            x = Utilz.TransposeArray(x);
            var h10 = _qc.Forward(x, out obj, true);                        // (batch, sq, i)
            var h1 = Utilz.Reshape(h10, h10.Length, -1, _heads, _dk);       // (batch, -1, h, dk)
            h1 = Utilz.Transpose(h1, 1, 2);                                 // (batch, h, -1, dk)
            ctx.Add(obj);
            // Transformer head 2
            var h20 = _kc.Forward(x, out obj, true);                        // (batch, sq, i)
            var h2 = Utilz.Reshape(h20, h20.Length, -1, _heads, _dk);       // (batch, -1, h, dk)
            h2 = Utilz.ReOrder(h2, 0, 2, 3, 1);                             // (batch, h, dk, -1)
            ctx.Add(obj);
            // Tranformer Scores
            var scores = Function.Matmul.Fwd(h1, h2, out obj);
            ctx.Add(obj);
            // Activation Function
            scores = Function.Sqrt.Fwd(scores, out obj);                    // (batch, h, -1, -1)
            ctx.Add(obj);
            // Tranformer value head
            var hv0 = _vc.Forward(x, out obj, true);                        // (batch, sq, i)
            var hv = Utilz.Reshape(hv0, hv0.Length, -1, _heads, _dk);       // (batch, -1, h, dk)
            hv = Utilz.Transpose(hv, 1, 2);                                 // (batch, h, -1, dk)
            ctx.Add(obj);
            // Matmul scores with value head
            var y = Function.Matmul.Fwd(scores, hv, out obj);               // (batch, h, -1, dk)
            ctx.Add(obj);
            y = Utilz.Transpose(y, 1, 2);                                   // (batch, -1, h, dk)
            var y2 =  Utilz.Reshape(y, y.Length, _nSeq, _qc._outFeatures);  // (batch, sq, i)
            // Linear Layer
            y2 = _fc.Forward(y2, out obj);                                  // (batch, sq, _outFeatures)
            ctx.Add(obj);
            return y2;
        }

        /// <summary>
        /// Inside Backwards pass
        /// </summary>
        /// <param name="grad">Output Gradient</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <param name="requiresInGrad">If requires Input Gradient</param>
        /// <returns>Input Gradient</returns>
        public float[][][] Backward(float[][][] grad, object ctx_, bool requiresInGrad = true)
        {
            var ctx = (List<object>)ctx_;
            // If using BatchNorm
            var useNorm = _norm != null;
            // Validation
            var ctxlen = 8 + (useNorm ? 1 : 0);
            if (ctx.Count != ctxlen)
                throw new Exception($"ctx.Count({ctx.Count}) != {ctxlen}");
            // Validate Shape
            var batch = (int)ctx[0];
            if (grad.Length != batch)
                throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                        $"it should be ({batch}, ...).");
            var requiresInGradNorm = useNorm || requiresInGrad;
            var vix = ctx.Count - 1;
            // Linear Layer
            var g22 = _fc.Backward(grad, ctx[vix--]);
            var g = Utilz.Reshape(g22, batch, -1, _heads, _dk);
            g = Utilz.Transpose(g, 1, 2);
            // Gradient for scores & value head
            float[][][][] scores, gv;
            Function.Matmul.Bck(g, out scores, out gv, ctx[vix--]);
            // Gradient for value head
            gv = Utilz.Transpose(gv, 1, 2);
            var gv0 = Utilz.Reshape(gv, batch, _nSeq, _qc._outFeatures);
            gv0 = _vc.Backward(gv0, ctx[vix--], requiresInGradNorm);
            // Activation Function
            scores = Function.Sqrt.Bck(scores, ctx[vix--]);
            // Gradient for head 1 & 2
            float[][][][] g1, g2;
            Function.Matmul.Bck(scores, out g1, out g2, ctx[vix--]);
            // Gradient for head 2
            g2 = Utilz.ReOrder(g2, 0, 3, 1, 2);
            var g20 = Utilz.Reshape(g2, batch, _nSeq, _qc._outFeatures);
            g20 = _kc.Backward(g20, ctx[vix--], requiresInGradNorm);
            // Gradient for head 1
            g1 = Utilz.Transpose(g1, 1, 2);
            var g10 = Utilz.Reshape(g1, batch, _nSeq, _qc._outFeatures);
            g10 = _qc.Backward(g10, ctx[vix--], requiresInGradNorm);
            // If requires in grad or using BatchNorm
            var ingrad = gv0;
            if (requiresInGradNorm)
            {
                // Loop batch
                for (int i = 0; i < batch; i++)
                {
                    var ingi = ingrad[i];
                    var g2i = g20[i];
                    var g1i = g10[i];
                    // Loop Sequence
                    for (int j = 0; j < _nSeq; j++)
                    {
                        var ingj = ingi[j];
                        var g2j = g2i[j];
                        var g1j = g1i[j];
                        // Grad Sum in Features
                        for (int k = 0; k < _inFeatures; k++)
                            ingj[k] += g2j[k] + g1j[k];
                    }
                }
                // If using Batch Norm
                if (useNorm)
                    ingrad = _norm.Backward(ingrad, ctx[vix--], requiresInGrad);
            }
            // Validation
            if (vix != 0) throw new Exception("vix != 0");
            return ingrad;
        }

        /// <summary>
        /// Inside backwards pass
        /// </summary>
        /// <param name="y">Output node from forward pass</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = Backward((float[][][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }

        /// <summary>
        /// My Transformer Convolutional 1d Layer
        /// Very close to a Transformer layer, but using Conv1d layers instead of fully connected layers
        /// </summary>
        /// <param name="nInput">In Features</param>
        /// <param name="nSequence">Sequence size</param>
        /// <param name="heads">Number of Transformer heads</param>
        /// <param name="channels">Number of Convolutional Channels</param>
        /// <param name="len">Length of span for Convolution</param>
        /// <param name="stride">Stride of Convolution</param>
        /// <param name="nOut">Out features: default is the size of in features (nInput)</param>
        /// <param name="padding">Padding for Convolution</param>
        /// <param name="batchNorm">If using batch norm</param>
        /// <param name="fcBias">If using bias for Linear Layer</param>
        /// <param name="name0">Parents names</param>
        public TransConv(int nInput, int nSequence, int heads, int channels, int len, int stride, 
                int nOut = -1, int padding = 0, bool batchNorm = true, bool fcBias = true, string name0= null)
            :base(name0, $"TransConv(nInput:{nInput}, nSequence:{nSequence}, heads:{heads}, channels:{channels}, len:{len}, " +
                            $"stride:{stride}, nOut:{nOut}, padding:{padding}, batchNorm:{batchNorm}, fcBias:{fcBias})")
        {
            _inFeatures = nInput;
            _nSeq = nSequence;
            _heads = heads;
            // Transformer Heads
            _qc = new Conv1dSparse(nInput, channels, len, stride, nSequence, padding, name0: ToString());
            _vc = new Conv1dSparse(nInput, channels, len, stride, nSequence, padding, name0: ToString());
            _kc = new Conv1dSparse(nInput, channels, len, stride, nSequence, padding, name0: ToString());
            _dk = _qc._outFeatures / heads;
            // Validation
            var dim = _dk * heads;
            if (_qc._outFeatures != dim || heads < 0)
            {
                var tryit = Enumerable.Range(2, _qc._outFeatures - 2).Where(n => _qc._outFeatures % n == 0);
                throw new Exception($"Attension outputs must be modified {_qc._outFeatures} => {dim} try: ({string.Join(", ", tryit)}).");
            }
            _outFeatures = (nOut <= 0) ? nInput : nOut;
            // Linear Layer
            _fc = new Linear(_qc._outFeatures, _outFeatures, fcBias, name0: ToString());
            _modules.AddRange(new[] { _qc, _vc, _kc });
            _modules.Add(_fc);
            // If using batch norm
            if (batchNorm)
            {
                _norm = new BatchNormBack(nInput, nSequence, name0: ToString());
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
