using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Example Net using TransConv1d & LinearSparse
    /// </summary>
    public class Net : NetModule<float[][]>
    {
        // Norms
        public readonly NormInf _normx, _normy;
        // Layers
        readonly TransConv1d _trans1, _trans2, _ixtrans;
        readonly LinearSparse _fcs;
        readonly BatchNormBack _normfc;

        /// <summary>
        /// Evaluate Forwards pass
        /// </summary>
        /// <param name="x">Input into network</param>
        /// <returns>Output from network</returns>
        public float[][] ForwardEval(float[][] x)
        {
            Eval();
            x = _normx.Normalize(x);
            var y = (float[][])Forward(x);
            return _normy.UnNorm(y);
        }

        /// <summary>
        /// Must override _Fwd() Function for Forward pass
        /// </summary>
        /// <param name="x">Input tensor into network</param>
        /// <returns>Output tensor from network</returns>
        public override Tensor<float[][]> _Fwd(Tensor<object> x)
        {
            return Foward(Tensor<object>.CastFloat2d(x));
        }

        // All you have to do is this (Define the Forward pass) !! I use it!
        // You don't have to define the backwards pass.
        // Again, You do not need to define the forward pass & backwards pass explicitly
        public Tensor<float[][]> Foward(Tensor<float[][]> x)
        {
            // Split into 2 one dim1
            var split = Function.Split.Fwd(x, 1, 12, 12);
            x = split[0];
            var xix = split[1];
            // Transformer 1
            var y = _trans1.Forward(x);
            // Take 4 on dim1
            y = Function.Take.Fwd(y, _fcs._dim, 1);
            // Flatten dim1 & dim2
            var y2 = Function.Flatten.Fwd(y, 1);
            // Transformer 2
            y = _trans2.Forward(y2);
            // Take 4 on dim1
            y = Function.Take.Fwd(y, _fcs._dim, 1);
            // Transformer 3
            var yix = _ixtrans.Forward(xix);
            // Take 4 on dim1
            yix = Function.Take.Fwd(yix, _fcs._dim, 1);
            // Concatenate y & yix on dim2
            y = Function.Cat.Fwd(2, y, yix);
            // BatchNorm on dim -1
            y = _normfc.Forward(y);
            // Reshape
            var x3 = Function.T.Fwd(Function.Reshape.Fwd(x, -1, 4, 3));     // Transpose dim1 & dim0
            y = Function.T.Fwd(y);                                          // Transpose dim1 & dim0
            // Concatenate x3 & y on dim2
            y = Function.Cat.Fwd(2, x3, y);
            // Linear Sparse
            y = Function.T.Fwd(_fcs.Forward(y));                            // Transpose dim1 & dim0
            // Flatten dim1 & dim2
            return Function.Flatten.Fwd(y, 1);
        }

        /******
        /// <summary>
        /// Net Forward pass
        /// </summary>
        /// <param name="x">Input into network</param>
        /// <returns>Output from network</returns>
        public override Tensor<float[][]> _Fwd(Tensor<object> x)
        {
            var y = Forward((float[][])x, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }

        /// <summary>
        /// One does not need to do this! Just define the forwards pass above!
        /// Inside Forward Function
        /// </summary>
        /// <param name="x">Input into network</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <returns>Output from network</returns>
        public float[][] Forward(float[][] x, out object ctx_)
        {
            var ctx = new List<object>();
            ctx_ = ctx;
            object obj;
            // Split x on dim1
            var xxx = Function.Split.Fwd_dim1(x, out obj, 12, 12);
            ctx.Add(obj);
            x = xxx[0];
            var xix = xxx[1];
            // TransConv1d Layer
            var y = _trans1.Forward(x, out obj);
            ctx.Add(obj);
            // Take first values on dim 1
            y = Function.Take.Fwd(y, _fcs._dim, 1, out obj);
            ctx.Add(obj);
            // Flatten dims 1 & 2
            var x0 = Function.Flatten.Fwd_d1(y, out obj);
            ctx.Add(obj);
            // TransConv1d Layer
            y = _trans2.Forward(x0, out obj);
            ctx.Add(obj);
            // Take first values on dim 1
            y = Function.Take.Fwd(y, _fcs._dim, 1, out obj);
            ctx.Add(obj);
            // TransConv1d Layer
            var yiy = _ixtrans.Forward(xix, out obj);
            ctx.Add(obj);
            // Take first values on dim 1
            yiy = Function.Take.Fwd(yiy, _fcs._dim, 1, out obj);
            ctx.Add(obj);
            // Concatenate on dim 2
            y = Function.Cat.Fwd_d2(new[] { y, yiy }, out obj);
            ctx.Add(obj);
            // BatchNorm dim -1
            y = _normfc.Forward(y, out obj);
            ctx.Add(obj);
            // Transpose & Reshape
            var xs = Utilz.TransposeArray(Function.Reshape.Fwd(x, x.Length, 4, 3, out obj));
            ctx.Add(obj);
            y = Utilz.TransposeArray(y);
            // Concatenate on dim 2
            y = Function.Cat.Fwd_d2(new[] { xs, y }, out obj);
            ctx.Add(obj);
            // Linear Sparse
            y = Utilz.TransposeArray(_fcs.Forward(y, out obj));
            ctx.Add(obj);
            // Flatten dim 1 & 2
            var ye = Function.Flatten.Fwd_d1(y, out obj);
            ctx.Add(obj);
            return ye;
        }

        /// <summary>
        /// One does not need to do this! Just define the forwards pass above!
        /// Inside Backwards Function
        /// </summary>
        /// <param name="y">Output node from forward pass</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var ingrad = Backward((float[][])y._grad, y._ctx, y._parent._requiresGrad);
            if (ingrad != null)
                y.GradSum(ingrad);
        }
        /// <summary>
        /// One does not need to do this! Just define the forwards pass above!
        /// Inside Backwards Function
        /// </summary>
        /// <param name="grad">Output Gradient</param>
        /// <param name="ctx_">Context for backwards pass</param>
        /// <param name="xRequiresGrad">If 'x' needs a gradient</param>
        /// <returns>Input Gradient</returns>
        public float[][] Backward(float[][] grad, object ctx_, bool xRequiresGrad)
        {
            var ctx = (List<object>)ctx_;
            var vix = ctx.Count - 1;
            // UnFlatten dims 1 & 2
            var g3 = Utilz.TransposeArray(Function.Flatten.Bck_d1(grad, ctx[vix--]));
            // Linear Sparse
            g3 = _fcs.Backward(g3, ctx[vix--]);
            // UnConcat on dim2
            var c0 = Function.Cat.Bck_d2(g3, ctx[vix--]);
            var xsg = c0[0];
            g3 = c0[1];
            g3 = Utilz.TransposeArray(g3);
            float[][] xgrad = null;
            // If x requires grad: compute
            if (xRequiresGrad)
            {
                xsg = Utilz.TransposeArray(xsg);
                xgrad = Function.Reshape.Bck2d_3d(xsg, ctx[vix--]);
            }
            else vix--;
            // BatchNorm
            g3 = _normfc.Backward(g3, ctx[vix--]);
            // UnConcat on dim2
            var cat = Function.Cat.Bck_d2(g3, ctx[vix--]);
            g3 = cat[0];
            var gix = cat[1];
            // UnTake
            gix = Function.Take.Bck(gix, ctx[vix--]);
            // TransConv1d
            var xixgrad = _ixtrans.Backward(gix, ctx[vix--], xRequiresGrad);

            // UnTake
            g3 = Function.Take.Bck(g3, ctx[vix--]);
            // TransConv1d
            var xgc0 = _trans2.Backward(g3, ctx[vix--]);
            // UnFlatten dims 1 & 2
            g3 = Function.Flatten.Bck_d1(xgc0, ctx[vix--]);
            // UnTake
            g3 = Function.Take.Bck(g3, ctx[vix--]);
            // TransConv1d
            var xgrad2 = _trans1.Backward(g3, ctx[vix--], xRequiresGrad);

            // If x needs grad calulate
            if (xRequiresGrad)
            {
                // xgrad += xgrad2
                for (int i = 0; i < xgrad.Length; i++)
                {
                    var g = xgrad[i];
                    var g2 = xgrad2[i];
                    for (int j = 0; j < g.Length; j++)
                        g[j] += g2[j];
                }
                // Reverse split: Combine
                xgrad = Function.Split.Bck_dim1(new[] { xgrad, xixgrad }, ctx[vix--]);
            }
            else vix--;
            // Validate
            if (vix != -1) throw new Exception("vix != -1");
            return xgrad;
        }
        ***/

        /// <summary>
        /// Test Network Multithreaded
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="minEpoch">Minimum Epochs to reach</param>
        /// <param name="over">Early stopping size</param>
        /// <param name="iter">Number of times to decay the learning rate</param>
        public void Test(float lr = 0.1f, int minEpoch = 300, int over = 30, int iter = 2)
        {
            // Random Inputs & Random Targets
            var x = Rand2.Randn(100, 24);
            var labels = Rand2.Randn(100, 8);
            // Train it in Parallel: MultiThreaded
            TrainIt(lr, minEpoch, over, iter, x, labels, out int epochs, out double bestError,
                    normx: _normx, normy:_normy);
        }

        /// <summary>
        /// Example Net using TransConv1d & LinearSparse
        /// </summary>
        /// <param name="nInput">24 In Features</param>
        /// <param name="nOut">8 out Features</param>
        /// <param name="smoothBatchLen">smooth of 30</param>
        /// <param name="name0">Parents Names</param>
        public Net(int nInput = 24, int nOut = 8, int smoothBatchLen = 30, string name0 = null)
            : base(smoothBatchLen, name0, "Net")
        {
            _normx = new NormInf(nInput);
            _normy = new NormInf(nOut);
            // TransConv1d
            _trans1 = new TransConv1d(nInput / 2, 7, 6, 3, batchNorm: false, name0:ToString());
            _trans2 = new TransConv1d(nInput / 2, 7, 6, 3, name0: ToString());
            _ixtrans = new TransConv1d(nInput / 2, 7, 6, 3, name0: ToString());
            // Linear Sparse
            _fcs = new LinearSparse(9, 4, 2, name0: ToString());
            // Batch Norm
            _normfc = new BatchNormBack(6, 4, name0: ToString());
            _modules.AddRange(new[] { _normx, _normy });
            _modules.Add(_trans1);
            _modules.Add(_trans2);
            _modules.Add(_ixtrans);
            _modules.Add(_fcs);
            _modules.Add(_normfc);
            // Be sure to call this at the end of the constuctor
            GetAllModules(_modules);
        }
    }
}
