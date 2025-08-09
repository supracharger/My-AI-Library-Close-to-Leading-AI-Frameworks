using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

    /// <summary>
    /// Exponential Moving Average
    /// </summary>
    class EMA
    {
        // vars
        public readonly float _beta, _betaInv;
        public readonly int _len;
        int _sz;
        // Avg
        public float _ema { get; private set; }
        
        /// <summary>
        /// The whole series is Averaged
        /// </summary>
        /// <param name="price">Series to Average</param>
        /// <param name="len">Length of the Average</param>
        /// <returns>Average Series of same length</returns>
        public static List<float> RunSeries(List<float> price, int len)
        {
            // Exit
            if (len == 1) return price.ToList();
            // Validation
            if (price.Count < (len + 1) * 2) throw new Exception("price.Length < (len + 1) * 2");
            // EMA
            var ema = new EMA(len);
            // Output
            var ls = new List<float>(price.Count);
            // Warm-Up
            for (int i = 0; i <= len; i++)
                ema.Run(price[i]);
            // Get Values
            for (int i = len + 1; i < price.Count; i++)
                ls.Add(ema.Run(price[i]));
            // Go Backwards to get the first few values
            ema = new EMA(len);
            // Output of first few values
            var lbefore = new List<float>(len + 1);
            // Reverse to get values
            var pr = price.Take((len + 1) * 2).Reverse().ToArray();
            // Warm-Up
            for (int i = 0; i <= len; i++)
                ema.Run(pr[i]);
            // Get values
            for (int i = len + 1; i < pr.Length; i++)
                lbefore.Add(ema.Run(pr[i]));
            // Concatenate
            return lbefore.AsEnumerable().Reverse().Concat(ls).ToList();
        }

    /// <summary>
    /// Run EMA through one iteration
    /// </summary>
    /// <param name="price">Value to average</param>
    /// <returns>Output of EMA</returns>
    public float Run(float price)
        {
            // Warm-Up
            var warmup = _sz < _len;
            if (_len <= 1)
                return price;
            if (warmup)
            {
                _sz++;
                if (_sz == 1)
                {
                    _ema = price;
                    return float.NaN;
                }
            }
            // Average
            _ema = price * _beta + _ema * _betaInv;
            // If on warmup: return NaN
            if (warmup) return float.NaN;
            // Return Average
            return _ema;
        }

        /// <summary>
        /// Run EMA through one iteration without Warm-Up
        /// </summary>
        /// <param name="value">New value to average</param>
        /// <returns>EMA Average value</returns>
        public float RunRaw(float value)
            => _ema = value * _beta + _ema * _betaInv;

    /// <summary>
    /// Exponential Moving Average. If using average Length
    /// </summary>
    /// <param name="length">Length of Average</param>
    public EMA(int length)
        {
            _len = Math.Max(length, 1);
            _beta = 2f / (length + 1);
            _betaInv = 1 - _beta;
        }

    /// <summary>
    /// Exponential Moving Average. If using the percent of the value in the Calculation.
    /// </summary>
    /// <param name="beta">Decimal from 0 => 1 of the weight of the value</param>
    /// <param name="startAtOne">Initialize Average to 1 instead of zero</param>
    public EMA(double beta, bool startAtOne = false)
        {
            // Validation
            if (beta <= 0 || beta >= 1)
                throw new Exception("beta <= 0 || beta >= 1");
            _beta = (float)beta;
            _betaInv = 1 - _beta;
            // If initialize to one
            if (startAtOne)
                _ema = 1;
        }

    /// <summary>
    /// Exponential Moving Average. For Specific 'beta' & 'length'
    /// </summary>
    /// <param name="beta">Decimal from 0 => 1 of the weight of the value</param>
    /// <param name="length">Length of Average</param>
    public EMA(float beta, int length)
        {
            _len = Math.Max(length, 1);
            _beta = Math.Min(beta, 1);
            _betaInv = 1 - _beta;
        }
    }
