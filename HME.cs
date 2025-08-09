using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
    
    /// <summary>
    /// Hull Moving Average.
    /// However, using EMA averages instead of Weighted averages for faster calculation.
    /// </summary>
    class HME
    {
        // Averages
        readonly EMA _ema1, _ema2, _diff;

    /// <summary>
    /// Hull Moving Average.
    /// However, using EMA averages instead of Weighted averages for faster calculation.
    /// </summary>
    /// <param name="omega">Decimal weight from 0 => 1 of the weight of the new value</param>
    /// <param name="startAtOne">If you want to initialize the averages with a value of 1 else 0.</param>
    public HME(double omega, bool startAtOne = false)
        {
            // Validation
            if (omega <= 0 || omega >= 1)
                throw new Exception("omega <= 0 || omega >= 1");
            // Components
            omega = Math.Min(omega, 1);
            _ema1 = new EMA(omega * 2, startAtOne);
            _ema2 = new EMA(omega, startAtOne);
            _diff = new EMA(Math.Sqrt(omega), startAtOne);
        }

        /// <summary>
        /// Run HME One Iteration
        /// </summary>
        /// <param name="value">New value to average</param>
        /// <returns>Updated HME Average</returns>
        public float Run(float value)
        {
            var v1 = _ema1.RunRaw(value);
            var v2 = _ema2.RunRaw(value);
            return _diff.RunRaw(2 * v1 - v2);
        }
    }
