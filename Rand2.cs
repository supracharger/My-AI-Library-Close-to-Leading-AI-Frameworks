using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Random Values.
    /// Used for Random Uniform & Adding Noise
    /// </summary>
    public static class Rand2
    {
        static Random _rand = new Random();

        /// <summary>
        /// Uniform distribution of random values from 0->1
        /// </summary>
        /// <param name="size">Array size</param>
        /// <returns>Uniform distribution of random values from 0->1</returns>
        public static float[] Uniform(int size)
        {
            // Get twice the values then crop
            var values = new bool[size * 2].Select(no => (float)_rand.NextDouble()).ToArray();
            // Uniform Normalization
            var min = values.Min();
            var div = (values.Max() - min) + 1e-12f;
            // Normalize & Crop
            return values.Take(size).Select(v => (v - min) / div).ToArray();
        }
        /// <summary>
        /// Uniform distribution of random values from 0->1
        /// </summary>
        /// <param name="dim0">Array dim 0</param>
        /// <param name="dim1">Array dim 1</param>
        /// <returns>Uniform distribution of random values from 0->1</returns>
        public static float[][] Uniform(int dim0, int dim1)
        {
            // Flatten dims
            var values = Uniform(dim0 * dim1);
            // Reshape to desired shape
            return Utilz.Reshape(values, dim0, dim1);
        }
        /// <summary>
        /// Uniform distribution of random values from 0->1
        /// </summary>
        /// <param name="dim0">Array dim 0</param>
        /// <param name="dim1">Array dim 1</param>
        /// <param name="dim2">Array dim 2</param>
        /// <returns>Uniform distribution of random values from 0->1</returns>
        public static float[][][] Uniform(int dim0, int dim1, int dim2)
        {
            // Flatten dims
            var values = Uniform(dim0 * dim1 * dim2);
            // Reshape to desired shape
            return Utilz.Reshape(values, dim0, dim1, dim2);
        }
        /// <summary>
        /// Single values of Uniform distribution of random values from 0->1
        /// </summary>
        /// <returns>Single values of Uniform distribution of random values from 0->1</returns>
        public static float Uniform() => Uniform(5)[0];   // 5*2 = 10 

        /// <summary>
        /// Random Uniform Ranging from -1 => 1
        /// </summary>
        /// <param name="dim">dimension of Array</param>
        /// <returns>Array of Random Uniform Ranging from -1 => 1</returns>
        public static float[] Randn(int dim)
            => Uniform(dim).Select(v => v * 2 - 1).ToArray();
        /// <summary>
        /// Random Uniform Ranging from -1 => 1
        /// </summary>
        /// <param name="dim0">dimension (0) of Output Array</param>
        /// <param name="dim1">dimension (1) of Output Array</param>
        /// <returns>Array of Random Uniform Ranging from -1 => 1</returns>
        public static float[][] Randn(int dim0, int dim1)
        {
            var flat = Uniform(dim0 * dim1).Select(v => v * 2 - 1).ToArray();
            return Utilz.Reshape(flat, dim0, dim1);
        }
        /// <summary>
        /// Random Uniform Ranging from -1 => 1
        /// </summary>
        /// <param name="dim0">dimension (0) of Output Array</param>
        /// <param name="dim1">dimension (1) of Output Array</param>
        /// <param name="dim2">dimension (2) of Output Array</param>
        /// <returns>Array of Random Uniform Ranging from -1 => 1</returns>
        public static float[][][] Randn(int dim0, int dim1, int dim2)
        {
            var flat = Uniform(dim0 * dim1 * dim2).Select(v => v * 2 - 1).ToArray();
            return Utilz.Reshape(flat, dim0, dim1, dim2);
        }

        /// <summary>
        /// Adds noise to values.
        /// </summary>
        /// <param name="values">Values to add noise</param>
        /// <param name="selectProb">Probability from 0->1 that the element is selected to add noise</param>
        /// <param name="amountProb">Max amount of noise from 0->1. amountProb=0.2 means +/-0.2</param>
        /// <returns>New Array of noised values</returns>
        public static float[] AddNoise(float[] values, float selectProb, float amountProb)
        => AddNoise(values,
                    Uniform(values.Length).Select(v => v <= selectProb).ToArray(),
                    Uniform(values.Length).Select(v => amountProb * (v * 2 - 1) + 1).ToArray());
        /// <summary>
        /// Adds noise to values.
        /// </summary>
        /// <param name="values">Values to add noise</param>
        /// <param name="selectProb">Probability from 0->1 that the element is selected to add noise</param>
        /// <param name="amountProb">Max amount of noise from 0->1. amountProb=0.2 means +/-0.2</param>
        /// <returns>New Array of noised values</returns>
        public static float[][] AddNoise(float[][] values, float selectProb, float amountProb)
        => AddNoise(values,
                    Uniform(values.Length).Select(v => v <= selectProb).ToArray(),
                    Uniform(values.Length, values[0].Length)
                        .Select(s => s.Select(v => v <= selectProb).ToArray()).ToArray(),
                    Uniform(values.Length, values[0].Length)
                        .Select(s => s.Select(v => amountProb * (v * 2 - 1) + 1).ToArray()).ToArray());
        /// <summary>
        /// Adds noise to values.
        /// </summary>
        /// <param name="values">Values to add noise</param>
        /// <param name="selectProb">Probability from 0->1 that the element is selected to add noise</param>
        /// <param name="amountProb">Max amount of noise from 0->1. amountProb=0.2 means +/-0.2</param>
        /// <returns>New Array of noised values</returns>
        public static float[][][] AddNoise(float[][][] values, float selectProb, float amountProb)
        => AddNoise(values,
                    Uniform(values.Length).Select(v => v <= selectProb).ToArray(),
                    Uniform(values.Length, values[0].Length)
                        .Select(s => s.Select(v => v <= selectProb).ToArray()).ToArray(),
                    Uniform(values.Length, values[0].Length, values[0][0].Length)
                        .Select(si => si.Select(s => s.Select(v => v <= selectProb).ToArray()).ToArray()).ToArray(),
                    Uniform(values.Length, values[0].Length, values[0][0].Length)
                        .Select(si => si.Select(s => s.Select(v => amountProb * (v * 2 - 1) + 1).ToArray()).ToArray()).ToArray()
            );
        /// <summary>
        /// Adds noise to values.
        /// </summary>
        /// <param name="values">Values to add noise</param>
        /// <param name="select2">Probability on first dim: from 0->1 that the element is selected to add noise</param>
        /// <param name="select">Probability on last dim: from 0->1 that the element is selected to add noise</param>
        /// <param name="amount">Max amount of noise from 0->1. amount=0.2 means +/-0.2</param>
        /// <returns>New Array of noised values</returns>
        static float[][] AddNoise(float[][] values, bool[] select2, bool[][] select, float[][] amount)
        => values.Select((v, i) =>
        {
            if (select2[i])
                return AddNoise(v, select[i], amount[i]);
            return v.ToArray();
        }).ToArray();
        /// <summary>
        /// Adds noise to values.
        /// </summary>
        /// <param name="values">Values to add noise</param>
        /// <param name="select">Probability from 0->1 that the element is selected to add noise</param>
        /// <param name="amount">Max amount of noise from 0->1. amount=0.2 means +/-0.2</param>
        /// <returns>New Array of noised values</returns>
        static float[] AddNoise(float[] values, bool[] select, float[] amount)
        => values.Select((v, i) => select[i] ? (v * amount[i]) : v).ToArray();
        /// <summary>
        /// Adds noise to values.
        /// </summary>
        /// <param name="values">Values to add noise</param>
        /// <param name="select3">Probability on first dim: from 0->1 that the element is selected to add noise</param>
        /// <param name="select2">Probability on dim (1): from 0->1 that the element is selected to add noise</param>
        /// <param name="select">Probability on last dim: from 0->1 that the element is selected to add noise</param>
        /// <param name="amount">Max amount of noise from 0->1. amount=0.2 means +/-0.2</param>
        /// <returns>New Array of noised values</returns>
        static float[][][] AddNoise(float[][][] values, bool[] select3, bool[][] select2, bool[][][] select, float[][][] amount)
        => values.Select((v, i) =>
        {
            if (select3[i])
                return AddNoise(v, select2[i], select[i], amount[i]);
            return v.ToArray();
        }).ToArray();

        /// <summary>
        /// Embedding Noise
        /// </summary>
        /// <param name="values">int values to noise</param>
        /// <param name="embedsize">The length of the embedding size</param>
        /// <param name="noise">Max amount of noise from 0->1. noise=0.2 means +/-0.2</param>
        /// <param name="selectProb">Probability from 0->1 that the element is selected to add noise</param>
        /// <returns>New Array of noised values</returns>
        public static int[] NoiseIt(int[] values, int embedsize, double noise, double selectProb = 0.3)
        {
            // Validation
            if (noise < 0 || noise >= 1) throw new Exception("noise < 0 || noise >= 1");
            if (selectProb <= 0 || selectProb > 1) throw new Exception("selectProb <= 0 || selectProb > 1");
            // Exit: No noise to add
            if (noise == 0) return values;
            // Maxiumun Index
            var maxval = embedsize - 1;
            // Max noise int size
            var ns = Math.Min(Math.Max((int)Math.Round(maxval * noise), 1), embedsize - 2);
            return values = values.Select(v =>
            {
                // Return regual value: if not selected
                if (Uniform() > selectProb)
                    return v;
                // Add Noise
                v = v + (_rand.Next(ns * 2 + 1) - ns);
                // Bound
                return Math.Min(Math.Max(v, 0), maxval);
            }).ToArray();
        }

        /// <summary>
        /// Add Noise uniformly.
        /// First get random noise, then on second round do the opposite noise
        /// </summary>
        public class Noise
        {
            // If element is selected
            bool[][][] _select;
            bool[][] _sel2;
            bool[] _sel3;
            // Noise Amount
            float[][][] _amount;
            // Select & Noise amount probability
            readonly float _vselect, _vamount;
            
            /// <summary>
            /// Add Noise
            /// </summary>
            /// <param name="values">values to add Noise to</param>
            /// <returns>New Array of noised values</returns>
            public float[][] Add(float[][] values)
            {
                // Get new noise
                if (_select == null)
                {
                    // Selection Probability
                    _sel2 = new[] { Uniform(values.Length).Select(v => v <= _vselect).ToArray() };
                    _select = new[] {Uniform(values.Length, values[0].Length)
                        .Select(s => s.Select(v => v <= _vselect).ToArray()).ToArray() };
                    // Noise Amount
                    _amount = new[] { Uniform(values.Length, values[0].Length)
                        .Select(s => s.Select(v => _vamount * (v * 2 - 1)).ToArray()).ToArray() };
                    // +/- Noise
                    var amount = _amount[0].Select(s => s.Select(v => v + 1).ToArray()).ToArray();
                    // Add noise
                    return AddNoise(values, _sel2[0], _select[0], amount);
                }
                // Do the inverse of noise on the same values
                {
                    // Validation
                    if (_amount.Length != 1 || values.Length != _amount[0].Length || values[0].Length != _amount[0][0].Length)
                        throw new Exception($"values.Length(1, {values.Length}, {values[0].Length}) != _amount.Length({_amount.Length}, {_amount[0].Length}, {_amount[0][0].Length}).");
                    // Load Selection Probability
                    var sel2 = _sel2[0];
                    var select = _select[0];
                    // Add Inverse noise: +/- Noise
                    var amount = _amount[0].Select(s => s.Select(v => -v + 1).ToArray()).ToArray();
                    // Unset to get new noise next round
                    _sel2 = null;
                    _select = null;
                    _amount = null;
                    // Add noise
                    return AddNoise(values, sel2, select, amount);
                }
            }
            /// <summary>
            /// Add Noise
            /// </summary>
            /// <param name="values">values to add Noise to</param>
            /// <returns>New Array of noised values</returns>
            public float[][][] Add(float[][][] values)
            {
                // Get new noise
                if (_select == null)
                {
                    // Selection Probability
                    _sel3 = Uniform(values.Length).Select(v => v <= _vselect).ToArray();
                    _sel2 = Uniform(values.Length, values[0].Length).Select(s => 
                            s.Select(v => v <= _vselect).ToArray()).ToArray();
                    _select = Uniform(values.Length, values[0].Length, values[0][0].Length)
                        .Select(si => si.Select(s => s.Select(v => v <= _vselect).ToArray()).ToArray()).ToArray();
                    // Noise Amount
                    _amount = Uniform(values.Length, values[0].Length, values[0][0].Length)
                        .Select(si => si.Select(s => s.Select(v => _vamount * (v * 2 - 1)).ToArray()).ToArray()).ToArray();
                    // +/- Noise
                    var amount = _amount.Select(si => si.Select(s => s.Select(v => v + 1).ToArray()).ToArray()).ToArray();
                    // Add noise
                    return AddNoise(values, _sel3, _sel2, _select, amount);
                }
                // Do the inverse of noise on the same values
                {
                    // Validation
                    if (values.Length != _amount.Length || values[0].Length != _amount[0].Length || values[0][0].Length != _amount[0][0].Length)
                        throw new Exception($"values.Length({values.Length}, {values[0].Length}, {values[0][0].Length}) != " +
                                $"_amount.Length({_amount.Length}, {_amount[0].Length}, {_amount[0][0].Length}).");
                    // Load Selection Probability
                    var select = _select;
                    // Add Inverse noise: +/- Noise
                    var amount = _amount.Select(ai => ai.Select(s => s.Select(v => -v + 1).ToArray()).ToArray()).ToArray();
                    // Unset to get new noise next round
                    _select = null;
                    _amount = null;
                    // Add noise
                    return AddNoise(values, _sel3, _sel2, select, amount);
                }
            }
            /// <summary>
            /// Add Noise
            /// </summary>
            /// <param name="values">values to add Noise to</param>
            /// <returns>New Array of noised values</returns>
            public float[] Add(float[] values)
                => Add(new[] { values })[0];

            /// <summary>
            /// Add Noise uniformly.
            /// First get random noise, then on second round do the opposite noise
            /// </summary>
            /// <param name="select">Probability from 0->1 that the element is selected to add noise</param>
            /// <param name="amount">Max amount of noise from 0->1. amount=0.2 means +/-0.2 noise</param>
            public Noise(double select, double amount)
            {
                if (select <= 0 || select > 1) throw new Exception("select <= 0 || select > 1");
                if (amount <= 0 || amount > 1) throw new Exception("amount <= 0 || amount > 1");
                _vselect = (float)select;
                _vamount = (float)amount;
            }
        }

        /// <summary>
        /// Add Noise uniformly for integer values.
        /// First get random noise, then on second round do the opposite noise
        /// </summary>
        public class NoiseInt
        {
            // vars
            readonly int _embedsize, _noise, _minix;
            readonly float _selectProb;
            // Previous noise
            int[] _prev;

            /// <summary>
            /// Add Noise
            /// </summary>
            /// <param name="values">values to add noise to</param>
            /// <returns>New Array with added noise</returns>
            public int[] AddNoise(int[] values)
            {
                int[] noise;
                // Get new noise values
                if (_prev == null)
                    noise = _prev = GetNoise(values.Length);
                // Use previous & invert them
                else
                {
                    // Validate
                    if (values.Length != _prev.Length)
                        throw new Exception($"values.Length({values.Length}) != _prev.Length({_prev.Length})");
                    // Invert Noise
                    noise = _prev.Select(v => -v).ToArray();
                    // Unset to get new noise next round
                    _prev = null;
                }
                // Maximum Index
                var maxval = _embedsize - 1;
                // Add Noise & Bound to correct values
                return values.Zip(noise, (v, n) => Math.Min(Math.Max(v + n, _minix), maxval)).ToArray();
            }

            /// <summary>
            /// Get the noise
            /// </summary>
            /// <param name="size">The size of the values</param>
            /// <returns>Noise for the values</returns>
            int[] GetNoise(int size)
            {
                // Select Probability
                var select = Uniform(size);
                // +/- Noise
                var top = _noise * 2 + 1;
                // Get Noise
                return Enumerable.Range(0, size).Select(i =>
                {
                    // If not selected: return no noise
                    if (select[i] > _selectProb)
                        return 0;
                    return _rand.Next(top) - _noise;
                }).ToArray();
            }

            /// <summary>
            /// Add Noise uniformly for integer values.
            /// First get random noise, then on second round do the opposite noise
            /// </summary>
            /// <param name="embedsize">The length of the embedding</param>
            /// <param name="noise">Max amount of noise from 0->1. noise=0.2 means +/-0.2 noise</param>
            /// <param name="selectProb">Probability from 0->1 that the element is selected to add noise</param>
            public NoiseInt(int embedsize, double noise, double selectProb = 0.3, int minIndex = 0)
            {
                if (noise <= 0 || noise >= 1) throw new Exception("noise <= 0 || noise >= 1");
                if (selectProb <= 0 || selectProb > 1) throw new Exception("selectProb <= 0 || selectProb > 1");
                _embedsize = embedsize;
                _selectProb = (float)selectProb;
                var maxval = embedsize - 1;
                _minix = minIndex;
                _noise = Math.Min(Math.Max((int)Math.Round(maxval * noise), 1), embedsize - 2);
            }
        }
    }
}
