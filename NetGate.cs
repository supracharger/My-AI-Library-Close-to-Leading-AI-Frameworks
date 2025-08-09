using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Gate Module as full network.
    /// </summary>
    public class NetGate : NetModule<float[][]>
    {
        // Norms
        public readonly NormInf _normx, _normy;
        // Gate
        readonly Gate _gate;
        // Vars
        readonly int _nInput;

        /// <summary>
        /// Inference Forwards pass
        /// </summary>
        /// <param name="x">Inputs into network</param>
        /// <returns>Outputs of network</returns>
        public float[][] ForwardEval(float[][] x)
        {
            // Norm X
            x = _normx.Normalize(x);
            // Forwards Pass
            var y = (float[][])Forward(x);
            // Undo Norm Y
            return _normy.UnNorm(y);
        }

        /// <summary>
        /// You don't need this function!
        ///     Just use the commented function below!
        /// Explicitly defined forwards pass
        /// </summary>
        /// <param name="x">Tensor to be inputted into network</param>
        /// <returns>Output Tensor of network</returns>
        public override Tensor<float[][]> _Fwd(Tensor<object> x)
        {
            var y = _gate.Forward((float[][])x, out object ctx);
            return x.AddNode(y, ctx, Backward, true);
        }

        /*** All you have to do is do this! 
         * You don't have to explicitly define the Forwards pass & Backwards pass!
        public override Tensor<float[][]> _Fwd(Tensor<object> x_)
        {
            var x = Tensor<object>.CastFloat2d(x_);
            return _gate.Forward(x);
        }
        ***/

        /// <summary>
        /// You don't need this function!
        ///     Just use the commented function above!
        /// Explicitly defined backwards pass
        /// </summary>
        /// <param name="y">Output node of forwards pass</param>
        void Backward(AutoGrad.BackwardNode y)
        {
            var requiresInGrad = y._parent == null || y._parent._requiresGrad;
            var ingrad = _gate.Backward((float[][])y._grad, y._ctx, requiresInGrad);
            if (requiresInGrad)
                y.GradSum(ingrad);
        }

        /// <summary>
        /// Test Network on Random values.
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="minEpoch">Minimum epoch to reach</param>
        /// <param name="over">Early stopping, give up if it didn't find best network in 'over' epochs.</param>
        /// <param name="iter">Number of times to decay the learning rate</param>
        public void Test(float lr = 0.1f, int minEpoch = 300, int over = 30, int iter = 2)
        {
            var rand = new Random();
            // Random X & Label values from -1 => 1
            Func<int, float> BiRand = no => (float)rand.NextDouble() * 2 - 1;
            var x = new int[100].Select(no => new int[16].Select(BiRand).ToArray()).ToArray();
            var labels = new int[100].Select(no => new int[8].Select(BiRand).ToArray()).ToArray();
            // Errors for each epoch
            var errors = new List<float>(minEpoch * iter);
            // best error
            var best = float.PositiveInfinity;
            // Best network parameters
            object bestParams = null;
            // Epoch counter
            var ctr = 0;
            // Do this because below it will be /=10
            lr *= 10;
            // Loop 'iter' times
            for (int i = 0; i < iter; i++)
            {
                var c = 0;
                lr /= 10;       // Decay learning rate
                // Early stopping and has reached minimum epochs
                while (c < over || ctr < minEpoch)
                {
                    ctr++;
                    // Forwards pass
                    var y = Forward(x);
                    // Loss
                    var loss = Criterion.MSE(y, labels);
                    // Backwards Pass
                    loss.Backward();
                    // Error
                    var er = loss.Item();
                    errors.Add(er);
                    // If better error: Save best parameters
                    if (er < best)
                    {
                        best = er;
                        bestParams = Parameters();
                        c = 0;
                    }
                    else
                        c++;
                    // Update parameters of network
                    Step(lr);
                }
                // Load best network
                LoadState(bestParams);
            }
        }

        /// <summary>
        /// Gate Module as full network.
        /// </summary>
        /// <param name="nInput">Network in features</param>
        /// <param name="minWt">Minimum weight size for Gate module</param>
        /// <param name="maxWt">Maximum weight size for Gate module</param>
        /// <param name="smoothBatchLen">Smooth gradients length</param>
        /// <param name="name0">Name of Parents</param>
        public NetGate(int nInput = 16, double minWt = 0.01, double maxWt = 0.99, int smoothBatchLen = 30, string name0 = null) 
            : base(smoothBatchLen, name0, "NetGate")
        {
            _nInput = nInput;
            _normx = new NormInf(nInput);
            _normy = new NormInf(nInput / 2);
            // Gate Module
            _gate = new Gate(nInput, minWt, maxWt, name0:ToString());
            _modules.AddRange(new[] { _normx, _normy });
            _modules.Add(_gate);
            // Make sure this is always called at the end of the constructor!
            GetAllModules(_modules);
        }
    }
}
