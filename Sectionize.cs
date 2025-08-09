using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Scoring Function for Error values.
    /// Usually used in time series
    /// </summary>
    public class Sectionize
    {
        // vars
        public readonly int _size, _nSplit;
        // Sections
        readonly List<int> _lens;

        /// <summary>
        /// Run Scoring Function
        /// </summary>
        /// <param name="values">Errors to score</param>
        /// <returns>Error Score</returns>
        public double Run(List<float> values)
        {
            // Validation
            if (values.Count != _size) throw new Exception($"values.Count({values.Count}) != _size({_size})");
            var ls = new List<double>(_nSplit);
            var start = 0;
            // Loop by sections
            for (int s = 0; s < _nSplit; s++)
            {
                // Length of section
                var len = _lens[s];
                var end = start + len;
                // Average errors in section
                var ensum = 0d;
                for (int i = start; i < end; i++)
                    ensum += values[i];
                ls.Add(ensum / len);
                // Move to next section
                start += len;
            }
            // Validation
            if (start != _size) throw new Exception();
            // Score
            var prod = 1d;
            foreach (var v in ls)
                prod *= 1 + v;
            return prod;
        }

        /// <summary>
        /// Chunk into Sections
        /// </summary>
        /// <param name="size">The size of the errors</param>
        /// <param name="split">Number of Sections</param>
        /// <returns>int array of the length of each section</returns>
        static List<int> GetLens(int size, int split)
        {
            // Validation
            if (size <= 0) throw new Exception("size <= 0");
            if (split <= 0) throw new Exception("split <= 0");
            if (size < split) throw new Exception("size < split");
            // Set to minimun size
            var len = size / split;
            var ls = new bool[split].Select(no => len).ToList();
            // Add remainder
            foreach (var r in Enumerable.Range(0, size % split))
                ls[r]++;
            // Validation
            if (ls.Sum() != size) throw new Exception();
            return ls;
        }

        /// <summary>
        /// Scoring Function for Error values.
        /// Usually used in time series.
        /// </summary>
        /// <param name="size">Length of the errors</param>
        /// <param name="nSplit">Number of Sections</param>
        public Sectionize(int size, int nSplit)
        {
            _size = size;
            _nSplit = nSplit;
            // Get length of each section
            _lens = GetLens(size, nSplit);
        }
    }
}
