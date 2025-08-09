using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/****
 * Modified Adam gradient optimizer.
 * It can you float[1d - 3d] gradient values.
 ****/

namespace MiniNet
{
    public class Adam
    {
        // Moments
        HME[][][] _m, _v, _vf;
        // Hyper params
        float _beta1, _beta2, _beta3, _limit;

        /// <summary>
        /// Update the 3d gradient using moded Adam Optimizer
        /// </summary>
        /// <param name="wt">The weight of the gradient</param>
        /// <param name="grad">The gradient of the weight</param>
        /// <returns>Updated Gradient</returns>
        public float[][][] Update(float[][][] wt, float[][][] grad)
        {
            var outgrad = new float[wt.Length][][];
            // Loop through dim0
            for (int i = 0; i < grad.Length; i++)
            {
                var wi = wt[i];
                var gi = grad[i];
                var mi = _m[i];
                var vi = _v[i];
                var fi = _vf[i];
                var oi = outgrad[i] = new float[wi.Length][];
                // Loop through dim1
                for (int j = 0; j < gi.Length; j++)
                {
                    var wj = wi[j];
                    var gj = gi[j];
                    var mj = mi[j];
                    var vj = vi[j];
                    var fj = fi[j];
                    var o = oi[j] = new float[wj.Length];
                    // Loop through dim2
                    for (int k = 0; k < gj.Length; k++)
                    {
                        // Modified m value
                        var target = wj[k] - 1.7f * gj[k];
                        // v value
                        var v = (float)Math.Pow(gj[k], 2);
                        var m = mj[k].Run(target);              // Smooth m with HME
                        //var m = mij[k] = _beta1 * mij[k] + (1 - _beta1) * target;
                        // var .....................
                        var f = fj[k].Run(v);                   // Smooth f with HME
                        //var f = fij[k] = _beta3 * fij[k] + (1 - _beta3) * v;
                        v = vj[k].Run(v);                       // Smooth v with HME
                        //v = vij[k] = _beta2 * vij[k] + (1 - _beta2) * v;
                        v = (float)Math.Sqrt(Math.Max(v, 0));
                        f = (float)Math.Sqrt(Math.Max(f, 0));
                        // When slow avg is less than fast avg: step slow & vise versa.
                        if (v != 0 && f != 0)
                            v *= Math.Min(Math.Max(1 + (f - v) / (v + 1e-12f), 0.7f), 3);
                        // Calc
                        // Convert back to gradient
                        m = wj[k] - m;
                        // Adam moments calc
                        var ot = o[k] = m / (v + 0.3f);
                        // Undef values
                        if (float.IsNaN(ot) || float.IsInfinity(ot))
                            throw new Exception("float.IsNaN(ot) || float.IsInfinity(ot)");
                    }
                }
            }
            // Set small magnitudes to zero that are less than pct
            if (_limit != 0)
            {
                // Condition based on magnitude
                var abs = outgrad.Select(gi => gi.Select(g => g.Select(Math.Abs).ToArray()).ToArray()).ToArray();
                // Sort to find top pct
                var sorted = abs.SelectMany(gi => gi.SelectMany(g => g)).OrderByDescending(a => a).ToList();
                // Find magnitude to crop by
                var crop = sorted[(int)Math.Min(Math.Max(Math.Round(sorted.Count * _limit), 0), sorted.Count - 1)];
                // if magnitude is less than crop: set gradient to zero.
                outgrad = outgrad.Zip(abs, (gi, ai) => gi.Zip(ai, (gj, aj) => gj.Zip(aj, (g, a) =>
                {
                    if (a < crop) return 0;
                    return g;
                }).ToArray()).ToArray()).ToArray();
            }
            return outgrad;
        }
        /// <summary>
        /// Update the 2d gradient using moded Adam Optimizer
        /// </summary>
        /// <param name="wt">The weight of the gradient</param>
        /// <param name="grad">The gradient of the weight</param>
        /// <returns>Updated Gradient</returns>
        public float[][] Update(float[][] wt, float[][] grad)
            => Update(new[] { wt }, new[] { grad })[0];
        /// <summary>
        /// Update the 1d gradient using moded Adam Optimizer
        /// </summary>
        /// <param name="wt">The weight of the gradient</param>
        /// <param name="grad">The gradient of the weight</param>
        /// <returns>Updated Gradient</returns>
        public float[] Update(float[] wt, float[] grad)
            => Update(new[] { new[] { wt } }, new[] { new[] { grad } })[0][0];

        /// <summary>
        /// Adam Optimizer 3d
        /// </summary>
        /// <param name="wt">The weight to be optimized</param>
        /// <param name="beta1">origonal value 0.9</param>
        /// <param name="beta2">origonal value 0.999</param>
        /// <param name="beta3">Modified Adam. Must be smaller than beta2</param>
        /// <param name="limitpct">Hyper parameter of: Set small magnitudes to zero that are less than pct.</param>
        public Adam(float[][][] wt, double beta1= 0.95, double beta2= 0.99, double beta3 = 0.97, double limitpct = 0)
        => Init(wt, beta1, beta2, beta3, limitpct);
        /// <summary>
        /// Adam Optimizer 2d
        /// </summary>
        /// <param name="wt">The weight to be optimized</param>
        /// <param name="beta1">origonal value 0.9</param>
        /// <param name="beta2">origonal value 0.999</param>
        /// <param name="beta3">Modified Adam. Must be smaller than beta2</param>
        /// <param name="limitpct">Hyper parameter of: Set small magnitudes to zero that are less than pct.</param>
        public Adam(float[][] wt, double beta1 = 0.95, double beta2 = 0.99, double beta3 = 0.97, double limitpct = 0)
        => Init(new[] { wt }, beta1, beta2, beta3, limitpct);
        /// <summary>
        /// Adam Optimizer 1d
        /// </summary>
        /// <param name="wt">The weight to be optimized</param>
        /// <param name="beta1">origonal value 0.9</param>
        /// <param name="beta2">origonal value 0.999</param>
        /// <param name="beta3">Modified Adam. Must be smaller than beta2</param>
        /// <param name="limitpct">Hyper parameter of: Set small magnitudes to zero that are less than pct.</param>
        public Adam(float[] wt, double beta1 = 0.95, double beta2 = 0.99, double beta3 = 0.97, double limitpct = 0)
       => Init(new[] { new[] { wt } }, beta1, beta2, beta3, limitpct);

        void Init(float[][][] wt, double beta1 = 0.95, double beta2 = 0.99, double beta3 = 0.97, double limitpct = 0)
        {
            // Conditionals
            if (beta3 >= beta2) throw new Exception("beta3 >= beta2");
            if (new[] { beta1, beta2, beta3 }.Any(b => b <= 0 || b >= 1)) 
                throw new Exception("new[] {beta1, beta2, beta3}.Any(b => b <= 0 || b >= 1)");
            if (limitpct < 0 || limitpct > 1)
                throw new Exception("limitpct < 0 || limitpct > 1");
            // Set Hyper params.
            _beta1 = (float)beta1;
            _beta2 = (float)beta2;
            _beta3 = (float)beta3;
            _limit = (limitpct < 1) ? (float)limitpct : 0;
            // Moments using HME()
            _m = wt.Select(w => w.Select(ww => ww.Select(no => new HME(1 - beta1)).ToArray()).ToArray()).ToArray();
            _v = wt.Select(w => w.Select(ww => ww.Select(no => new HME(1 - beta2, true)).ToArray()).ToArray()).ToArray();
            _vf = wt.Select(w => w.Select(ww => ww.Select(no => new HME(1 - beta3, true)).ToArray()).ToArray()).ToArray();
            //_m = wt.Select(w => w.Select(ww => new float[ww.Length]).ToArray()).ToArray();
            //_v = wt.Select(w => w.Select(ww => ww.Select(no => 1f).ToArray()).ToArray()).ToArray();
            //_vf = wt.Select(w => w.Select(ww => ww.Select(no => 1f).ToArray()).ToArray()).ToArray();
        }
    }
}
