using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Test My 1d Convolutional network
    /// </summary>
    public class TestConv : NetModule<float[][][]>
    {
        // layers
        readonly Conv1d _conv;

        /// <summary>
        /// Forwards Pass
        /// </summary>
        /// <param name="x_">Tensor object Network input</param>
        /// <returns>Tensor object Netwok output</returns>
        public override Tensor<float[][][]> _Fwd(Tensor<object> x_)
        {
            var x = Tensor<object>.CastFloat2d(x_);
            return _conv.Forward(x);
        }

        /// <summary>
        /// Test the network with random values
        /// </summary>
        /// <param name="lr">learning rate</param>
        public void Test(float lr = 0.1f)
        {
            // Random X & Y
            var x = Rand2.Randn(100, 12);
            var labels = Rand2.Randn(100, 3, 3);
            //var y = Forward(x);
            // Error for each epoch
            var errors = new List<float>(100);
            // Best Error
            var best = float.PositiveInfinity;
            // Best Network
            object bestParams = null;
            // Loop n iterations
            for (int i = 0; i < 100; i++)
            {
                // Forward pass
                var y = Forward(x);
                // Backwards pass
                var loss = Criterion.Regular(y, labels);
                loss.Backward();
                // Error
                var er = loss.Item();
                errors.Add(er);
                // If better error: save network in memory
                if (er < best)
                {
                    best = er;
                    bestParams = Parameters();
                }
                // Update network parameters
                Step(lr);
            }
            // Load best network
            LoadState(bestParams);
        }

        /// <summary>
        /// Test My 1d Convolutional network
        /// </summary>
        /// <param name="name0">Name of Parents</param>
        public TestConv(string name0 = null)
            : base(30, name0, "TestConv")
        {
            _conv = new Conv1d(12, 3, 6, 3, name0:ToString());
            _modules.Add(_conv);
            // Make sure this is always called at the end of the constructor!
            GetAllModules(_modules);
        }
    }
}
