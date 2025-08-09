using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace MiniNet
{
    /// <summary>
    /// Normalization. Usually for inputs & output of the network
    /// </summary>
    public class NormInf : Module
    {
        // vars
        public readonly int _nSeq;
        bool _hasBeenFit;
        // Norms foreach dim
        readonly List<Norm> _norms;

        /// <summary>
        /// Normalize series
        /// </summary>
        /// <param name="series">Series to normalize</param>
        /// <param name="fitAgainOk">Override prompt to fit again.</param>
        /// <returns>Normalized series of the same shape as 'series'</returns>
        public float[][] Fit(float[][] series, bool fitAgainOk = false)
        {
            // Transpose to series[dim][batch]
            series = Utilz.TransposeArray(series);
            // Validation
            if (series.Length != _norms.Count)
                throw new Exception($"Invalid shape of series: got [{series.Length}][{series[0].Length}] should be [{_norms.Count}][batch].");
            // If already Fit and fitAgainOk=false: prompt user
            if (_hasBeenFit && !fitAgainOk)
            {
                // Infinate Loop
                while (true)
                {
                    Console.WriteLine("The Norms have already been fitted on previous values." +
                        " Do you want to Recreate the norms? The values will be overwritten. (Y / N): ");
                    var inp = Console.ReadLine().Trim();
                    if (inp == "Y")         // Fit Again
                        break;
                    else if (inp == "N")    // Else use previous normalization
                    {
                        Console.WriteLine("Ok, using previous norm values.");
                        return Utilz.TransposeArray(NormSub(series));
                    }
                    else
                        Console.WriteLine("Invalid Input.");
                }
            }
            _hasBeenFit = true;     // Flag to know it has been fit
            // Fit Series
            FitSub(series);
            // Normalize Series
            // Transpose back to origonal shape[batch][dim]
            return Utilz.TransposeArray(NormSub(series));
        }
        /// <summary>
        /// Normalize series
        /// </summary>
        /// <param name="series">Series to normalize</param>
        /// <param name="fitAgainOk">Override prompt to fit again.</param>
        /// <returns>Normalized series of the same shape as 'series'</returns>
        public float[][][] Fit(float[][][] series, bool fitAgainOk = false)
        {
            // Norm inputs for each sequence are the same
            int dim1, dim2;
            float[][] sr2;
            // If Normalization does not need to be done on sequence
            if (_nSeq == -1)
            {
                dim1 = series.Length;
                dim2 = series[0].Length;
                // Flatten last 2 dims
                sr2 = series.SelectMany(s => s).ToArray();
            }
            // Norm inputs for each sequence are different
            else
            {
                dim1 = series[0].Length;
                dim2 = series[0][0].Length;
                // Flatten first 2 dims
                sr2 = series.Select(s => s.SelectMany(si => si).ToArray()).ToArray();
            }
            // Transpose to shape[dim][batch]
            sr2 = Utilz.TransposeArray(sr2);
            // Validation
            if (sr2.Length != _norms.Count)
                throw new Exception($"Invalid shape of series: got [{sr2.Length}][{sr2[0].Length}] should be [{_norms.Count}][batch].");
            // If already Fit and fitAgainOk=false: prompt user
            if (_hasBeenFit && !fitAgainOk)
            {
                // Infinate Loop
                while (true)
                {
                    Console.WriteLine("The Norms have already been fitted on previous values." +
                        " Do you want to Recreate the norms? The values will be overwritten. (Y / N): ");
                    var inp = Console.ReadLine().Trim();
                    if (inp == "Y")         // Fit Again
                        break;
                    else if (inp == "N")    // Else use previous normalization
                    {
                        Console.WriteLine("Ok, using previous norm values.");
                        sr2 = Utilz.TransposeArray(NormSub(sr2));
                        if (_nSeq == -1)
                            return Utilz.Reshape(sr2, dim1, dim2);
                        // Norm inputs for each sequence are different
                        return Utilz.Reshape(sr2, sr2.Length, dim1, dim2);
                    }
                    else
                        Console.WriteLine("Invalid Input.");
                }
            }
            // Flag to know it has been fit
            _hasBeenFit = true;

            // Fit Series
            FitSub(sr2);
            // Normalize Series
            sr2 = NormSub(sr2);
            sr2 = Utilz.TransposeArray(sr2);
            // Norm inputs for each sequence are the same
            if (_nSeq == -1)
                return Utilz.Reshape(sr2, dim1, dim2);
            // Norm inputs for each sequence are different
            return Utilz.Reshape(sr2, sr2.Length, dim1, dim2);
        }

        /// <summary>
        /// Normalize series
        /// </summary>
        /// <param name="series">Series to normalize</param>
        /// <returns>Normalized series</returns>
        public float[][] Normalize(float[][] series)
            => Utilz.TransposeArray(NormSub(Utilz.TransposeArray(series)));
        /// <summary>
        /// Normalize series
        /// </summary>
        /// <param name="series">Series to normalize</param>
        /// <returns>Normalized series</returns>
        public float[][][] Normalize(float[][][] series)
        {
            // Norm inputs for each sequence are the same
            if (_nSeq == -1)
            {
                var dim0 = series.Length;
                var dim1 = series[0].Length;
                // Flatten first 2 dims
                var sr2 = series.SelectMany(s => s).ToArray();
                // Normalize
                sr2 = NormSub(Utilz.TransposeArray(sr2));
                // Reshape to origonal dims
                return Utilz.Reshape(Utilz.TransposeArray(sr2), dim0, dim1);
            }
            // Norm inputs for each sequence are different
            {
                var dim1 = series[0].Length;
                var dim2 = series[0][0].Length;
                // Flatten last 2 dims
                var sr2 = series.Select(s => s.SelectMany(si => si).ToArray()).ToArray();
                // Normalize
                sr2 = NormSub(Utilz.TransposeArray(sr2));
                // Reshape to origonal dims
                sr2 = Utilz.TransposeArray(sr2);
                return Utilz.Reshape(sr2, sr2.Length, dim1, dim2);
            }
        }

        /// <summary>
        /// Undo Normalization
        /// </summary>
        /// <param name="series">Series to UnNormalize</param>
        /// <returns>UnNormalized series</returns>
        public float[][] UnNorm(float[][] series)
            => Utilz.TransposeArray(UnNormSub(Utilz.TransposeArray(series)));
        /// <summary>
        /// Undo Normalization
        /// </summary>
        /// <param name="series">Series to UnNormalize</param>
        /// <returns>UnNormalized series</returns>
        public float[][][] UnNorm(float[][][] series)
        {
            // UnNorm inputs for each sequence are the same
            if (_nSeq == -1)
            {
                var dim0 = series.Length;
                var dim1 = series[0].Length;
                // Flatten first 2 dims
                var sr2 = series.SelectMany(s => s).ToArray();
                // UnNormalize
                sr2 = UnNormSub(Utilz.TransposeArray(sr2));
                // Reshape to origonal dims
                return Utilz.Reshape(Utilz.TransposeArray(sr2), dim0, dim1);
            }
            // UnNorm inputs for each sequence are different
            {
                var dim1 = series[0].Length;
                var dim2 = series[0][0].Length;
                // Flatten last 2 dims
                var sr2 = series.Select(s => s.SelectMany(si => si).ToArray()).ToArray();
                // UnNormalize
                sr2 = UnNormSub(Utilz.TransposeArray(sr2));
                // Reshape to origonal dims
                sr2 = Utilz.TransposeArray(sr2);
                return Utilz.Reshape(sr2, sr2.Length, dim1, dim2);
            }
        }

        /// <summary>
        /// Fit Series
        /// </summary>
        /// <param name="series">Series to fit</param>
        void FitSub(float[][] series)
        {
            // Loop each dim
            for (int i = 0; i < _norms.Count; i++)
                _norms[i].Calc(series[i]);
        }

        /// <summary>
        /// Normalize
        /// </summary>
        /// <param name="series">Series to Normalize</param>
        /// <returns>Normalized series</returns>
        float[][] NormSub(float[][] series)
        {
            // Validation
            if (series.Length != _norms.Count)
                throw new Exception($"Invalid shape of series: got [{series.Length}][{series[0].Length}] should be [{_norms.Count}][batch].");
            // Normalize
            return series.Select((s, i) => _norms[i].Normalize(s)).ToArray();
        }

        /// <summary>
        /// Undo Normalization
        /// </summary>
        /// <param name="series">Series to UnNormalize</param>
        /// <returns>UnNormalized series</returns>
        float[][] UnNormSub(float[][] series)
        {
            // Validation
            if (series.Length != _norms.Count)
                throw new Exception($"Invalid shape of series: got [{series.Length}][{series[0].Length}] should be [{_norms.Count}][batch].");
            // Undo Normalization
            return series.Select((s, i) => _norms[i].UnNorm(s)).ToArray();
        }

        /// <summary>
        /// Parameter for Normalization
        /// </summary>
        /// <returns>Object of Parameter for Normalization</returns>
        public override object Parameters()
        => _norms.Select(n => n.Parameters()).ToList();

        /// <summary>
        /// Number of Parameters.
        /// I don't count Normalization as a parameter
        /// </summary>
        /// <returns>Number of Parameters</returns>
        public override long NumParams() => 0;

        /// <summary>
        /// Load State Deep Copy
        /// </summary>
        /// <param name="items">Parameters object</param>
        public override void LoadState(object items)
        {
            // Flag has been fit
            _hasBeenFit = true;
            // Validation
            var ls = (List<object>)items;
            if (ls.Count != _norms.Count)
                throw new Exception($"There should be ({_norms.Count}), but got ({ls.Count}) items.");
            // Load Deep Copy of Parameters
            for (int i = 0; i < _norms.Count; i++)
                _norms[i].LoadState(ls[i]);
        }

        /// <summary>
        /// Load State JSON
        /// </summary>
        /// <param name="items">JSON object</param>
        public override void LoadStateJSON(object items)
        {
            // Flag has been fit
            _hasBeenFit = true;
            var ls = JsonConvert.DeserializeObject<List<object>>(items.ToString(), _settings);
            // Validation
            if (ls.Count != _norms.Count)
                throw new Exception($"There should be ({_norms.Count}), but got ({ls.Count}) items.");
            // Load Deep Copy of Parameters
            for (int i = 0; i < _norms.Count; i++)
                _norms[i].LoadStateJSON(ls[i]);
        }

        /// <summary>
        /// No Gradients to Calculate
        /// </summary>
        /// <param name="len">Not used</param>
        /// <returns>Negative Infinity</returns>
        public override float SmoothMax(int len) 
            => float.NegativeInfinity;
        /// <summary>
        /// Nothing to Step
        /// </summary>
        /// <param name="lr">Not used</param>
        /// <param name="div">Not used</param>
        public override void Step(float lr, float div = 1) { }

        /// <summary>
        /// Normalization. Usually for inputs & output of the network
        /// </summary>
        /// <param name="dim">In features</param>
        /// <param name="nSequence">For shape[3d]. Default value=-1: Do not normalize on Sequence also. Else: Normalize on sequence</param>
        /// <param name="pct">Percent 0->1 to Normalize on</param>
        /// <param name="name0">Parents names</param>
        public NormInf(int dim, int nSequence = -1, double pct = 0.7, string name0 = null)
            : base(name0, $"NormInf(dim:{dim}, nSequence:{nSequence}, pct:{pct})")
        {
            _nSeq = nSequence;
            // Normalize on Sequence also
            if (nSequence > 0)
                dim *= nSequence;
            // Create for each dim
            _norms = new bool[dim].Select(no => new Norm(pct)).ToList();
        }

        /// <summary>
        /// Normalization for each dim in the series
        /// </summary>
        class Norm
        {
            // vars
            readonly float _crop;
            // Parameters
            float _min, _div;

            /// <summary>
            /// Fit on dim
            /// </summary>
            /// <param name="values">Batches of dims to fit on</param>
            public void Calc(float[] values)
            {
                // Size to crop out of normalization
                var crop = (int)Math.Round(_crop * values.Length);
                // Ignore NaN values: Good for labels if you don't have the target
                values = values.Where(v => !float.IsNaN(v)).OrderBy(v => v).ToArray();
                // Min Parameter
                _min = values[Math.Min(Math.Max(crop - 1, 0), values.Length - 1)];
                // Max value
                var max = values[values.Length - Math.Min(Math.Max(crop, 1), values.Length)];
                // Validation
                if (max == _min) throw new Exception("max == _min: All the values are the same");
                // Div Parameter
                _div = (max - _min) + 1e-12f;
            }

            /// <summary>
            /// Normalize values
            /// </summary>
            /// <param name="values">values to Normalize</param>
            /// <returns>Normalized values</returns>
            public float[] Normalize(float[] values)
            => values.Select(v =>
            {
                v = Math.Min(Math.Max((v - _min) / _div, 0), 1);
                // Range -1 => 1
                return v * 2 - 1;
            }).ToArray();

            /// <summary>
            /// Undo Normalization
            /// </summary>
            /// <param name="values">values to UnNormalize</param>
            /// <returns>UnNormalized values</returns>
            public float[] UnNorm(float[] values)
                => values.Select(v =>
                {
                    // Bound
                    v = Math.Min(Math.Max(v, -1), 1);
                    // Undo Normalization
                    v = (v + 1) / 2;
                    return v * _div + _min;
                }).ToArray();

            /// <summary>
            /// Deep copy of Parameters
            /// </summary>
            /// <returns>Object of Deep copy of Parameters</returns>
            public object Parameters()
            => new[] { _min, _div };

            /// <summary>
            /// Parameters to Load
            /// </summary>
            /// <param name="items">Object of parameters to load</param>
            public void LoadState(object items)
            {
                // Validation
                var w = (float[])items;
                if (w.Length != 2)
                    throw new Exception($"There should be 2 items: got ({w.Length}) for {{min, div}}.");
                // Load State
                _min = w[0];
                _div = w[1];
            }

            /// <summary>
            /// Load State JSON
            /// </summary>
            /// <param name="items">JSON object</param>
            public void LoadStateJSON(object items)
            {
                var w = JsonConvert.DeserializeObject<float[]>(items.ToString(), _settings);
                LoadState(w);
            }

            /// <summary>
            /// Normalization for each dim in the series
            /// </summary>
            /// <param name="pct">Percent from 0->1 to Normalize on</param>
            public Norm(double pct = 0.7)
            {
                // Validation
                if (pct <= 0 || pct > 1) throw new Exception("pct <= 0 || pct > 1");
                // Get Crop
                _crop = (float)(1 - pct) / 2;
            }
        }
    }
}
