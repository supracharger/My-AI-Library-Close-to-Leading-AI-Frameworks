using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace MiniNet
{
    /// <summary>
    /// Test Linear Layers
    /// </summary>
    public class TestLinear : NetModule<float[][]>
    {
        // Layers
        readonly List<Linear> _fc;

        /// <summary>
        /// Forwards Pass
        /// </summary>
        /// <param name="x">Tensor object to be inputted into network</param>
        /// <returns>Tensor to be outputted from network</returns>
        public override Tensor<float[][]> _Fwd(Tensor<object> x)
        {
            var y = Tensor<object>.CastFloat2d(x);
            // Go through Linear Layers
            foreach (var fc in _fc)
                y = fc.Forward(y);
            // Bi-Sqrt Activation Function
            return Function.Sqrt.Fwd(y);
        }

        /// <summary>
        /// Test the network with random values.
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        public void Test(float lr = 0.1f)
        {
            // Random X & Labels
            var x = Rand2.Randn(100, 3);
            var labels = Rand2.Randn(100, 2);
            //var yy = Forward(x);
            // Errors for each epoch
            var errors = new List<float>(100);
            // Best Error
            var best = float.PositiveInfinity;
            // Best Network
            object bestParams = null;
            // Loop through n iterations
            for (int i = 0; i < 100; i++)
            {
                // Forwards pass
                var y = Forward(x);
                // Backwards pass
                var loss = (y - labels).Abs().Mean();
                loss.Backward();
                // Errors
                var er = loss.Item();
                errors.Add(er);
                // If better error: save network to memory
                if (er < best)
                {
                    best = er;
                    bestParams = Parameters();
                }
                // Update parameters of network
                Step(lr);
            }
            // Load best network
            LoadState(bestParams);
        }

        /// <summary>
        /// Test Linear Layers
        /// </summary>
        /// <param name="name0">Parents Names</param>
        public TestLinear(int smoothBatchLen = 30, string name0 = null)
            : base(smoothBatchLen, name0, "TestLinear")
        {
            // Linear Layers
            _fc = new List<Linear>
            {
                new Linear(3, 7, name0:ToString()),
                new Linear(7, 2, name0:ToString())
            };
            _modules.AddRange(_fc);
            // Make sure this is always called at the end of the constructor!
            GetAllModules(_modules);
        }
    }
}
