using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Test Embedding Network
    /// </summary>
    public class TestEmbedding : NetModule<float[][]>
    {
        // Layers
        readonly Embedding _embed;
        readonly List<Linear> _fcs;

        /// <summary>
        /// Forward Pass
        /// </summary>
        /// <param name="items_">Tensor object of network input</param>
        /// <returns>Tensor of output of network</returns>
        public override Tensor<float[][]> _Fwd(Tensor<object> items_)
        {
            // Get X & Embeddings
            var items = (NodeArray<object, object>)items_;
            var x = Tensor<object>.CastFloat2d(items[0]);
            var ii = Tensor<object>.CastInt1d(items[1]);
            // Run Embedding
            var ix = _embed.Forward(ii);
            // Concatenate them on dim 1
            x = Function.Cat.Fwd(1, x, ix);
            // Go through Linear Layers
            var y = x;
            foreach (var fc in _fcs)
                y = fc.Forward(y);
            return y;
        }

        /// <summary>
        /// Test the network with random values.
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        public void Test(float lr = 0.1f)
        {
            // Random X & Y values
            var x = Rand2.Randn(100, 4);
            var labels = Rand2.Randn(100, 2);
            // A Way to get embedding values
            var enorm = Utilz.Norm(x.Select(xi => xi.Average()).ToArray());
            // Max embedding index
            var maxval = _embed._size - 1;
            // Convert float values to integers
            var div = 1f / (_embed._size - 1);
            var embed = Utilz.Modulo(enorm, div, maxval);
            // Errors for each epoch
            var errors = new List<float>(100);
            // Best Error
            var best = float.PositiveInfinity;
            // Best Network
            object bestParams = null;
            // Iterate through n iterations
            for (int i = 0; i < 100; i++)
            {
                // Set to unnoised embedding values
                var em2 = embed;
                // Add noise
                var addnoise = Rand2.Uniform() < 0.3;
                if (addnoise)
                    em2 = Rand2.NoiseIt(em2, _embed._size, 0.1);
                // Forwards pass
                var y = Forward(true, x, em2);
                // Backwards pass
                var loss = Criterion.Regular(y, labels);
                loss.Backward();
                // If noise was added: rerun the network to get correct error
                if (addnoise)
                    y = Forward(true, x, embed);
                // Errors
                var errorbatch = Criterion.GetError(Criterion.Regular((float[][])y, labels));
                var er = errorbatch.Average();
                errors.Add(er);
                // If error is better: Save network to memory
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
        /// Test Embedding Network
        /// </summary>
        /// <param name="nInput">Network input size of 4</param>
        /// <param name="nOut">Network output size of 2</param>
        /// <param name="name0">Parents Names</param>
        public TestEmbedding(int nInput = 4, int nOut = 2, string name0 = null)
            :base(30, name0, "TestEmbedding")
        {
            // Embedding Layer
            _embed = new Embedding(10, 4, name0: ToString());
            // Linear Layers
            _fcs = new List<Linear>
            {
                new Linear(nInput + _embed._dim, 4, name0:ToString()),
                new Linear(4, nOut, name0:ToString())
            };
            _modules.Add(_embed);
            _modules.AddRange(_fcs);
            // Make sure this is always called at the end of the constructor!
            GetAllModules(_modules);
        }
    }
}
