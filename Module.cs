using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// This should be the inherited parent for all modules except for the outer most module you
    /// want to train, then use NetModule. This is for Module nesting and new created modules.
    /// </summary>
    public abstract class Module
    {
        // Identifiers
        public readonly string _name0, _name;
        // Modules
        protected readonly List<Module> _modules;
        // Json settings
        public static readonly JsonSerializerSettings _settings = JsonSettings();

        /// <summary>
        /// Step Parameters using Weight Grad
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="div">Max divisor (Grad Norm)</param>
        public abstract void Step(float lr, float div = 1);

        /// <summary>
        /// Reduces the gradient & Uses Optimizer
        /// </summary>
        /// <param name="len">Length to Smooth</param>
        /// <returns>The value of the max magnitude gradient value</returns>
        public virtual float SmoothMax(int len)
        { throw new NotImplementedException(); }

        /// <summary>
        /// Deep copy of Parameters
        /// </summary>
        /// <returns>Deep copy of Parameters</returns>
        public abstract object Parameters();
        /// <summary>
        /// Load in Weights Deep copy
        /// </summary>
        /// <param name="items">object state</param>
        public abstract void LoadState(object items);
        /// <summary>
        /// Load State JSON
        /// </summary>
        /// <param name="items">object JSON</param>
        public abstract void LoadStateJSON(object items);
        /// <summary>
        /// Number of Parameters
        /// </summary>
        /// <returns>Number of Parameters</returns>
        public abstract long NumParams();

        /// <summary>
        /// Gets Modules
        /// </summary>
        /// <returns>Modules</returns>
        List<Module> GetModules()
        {
            if (_modules.Count == 0)
                return new List<Module> { this };
            return _modules.ToList();
        }

        /// <summary>
        /// MUST be called at the end of the contructor if nesting any modules or
        /// inheritin from NetModule
        /// </summary>
        /// <param name="modules">Modules to find nested modules</param>
        protected static void GetAllModules(List<Module> modules)
        { 
            // Get All Modules
            var mod2 = modules.SelectMany(m => m.GetModules()).ToList();
            // Mutate List
            modules.Clear();
            modules.AddRange(mod2);
        }

        /// <summary>
        /// JSON Settings to try to make the state as most the same as possible
        /// </summary>
        public static JsonSerializerSettings JsonSettings()
        => new JsonSerializerSettings
            {
                Culture = CultureInfo.InvariantCulture,
                FloatFormatHandling = FloatFormatHandling.String,    // preserve text
                FloatParseHandling = FloatParseHandling.Decimal     // parse back as decimal
            };

        /// <summary>
        /// Initialize Weights 2d
        /// </summary>
        /// <param name="w">Weights to initialize</param>
        /// <param name="modify">value to modify fanin size: fanin /= modify</param>
        public static void InitWts(float[][] w, double? modify = null)
        {
            float fanin = w[0].Length;
            if (modify != null)
                fanin /= (float)modify.Value;
            fanin = 1 / Math.Max(fanin, 1);
            var ln2 = w[0].Length;
            // Uniformly distributed random values
            var rand = new bool[w.Length].Select(no => Rand2.Uniform(ln2)).ToArray();
            // Initialize weight
            for (int i = 0; i < w.Length; i++)
            {
                var wi = w[i];
                var r = rand[i];
                for (int j = 0; j < wi.Length; j++)
                    wi[j] = fanin * r[j];
            }
        }
        /// <summary>
        /// Initialize Weights 3d
        /// </summary>
        /// <param name="w">Weights to initialize</param>
        /// <param name="modify">value to modify fanin size: fanin /= modify</param>
        public static void InitWts(float[][][] w, double? modify = null)
        {
            for (int i = 0; i < w.Length; i++)
                InitWts(w[i], modify);
        }
        /// <summary>
        /// Initialize Weights 1d
        /// </summary>
        /// <param name="w">Weights to initialize</param>
        /// <param name="amount">Maximum random value</param>
        public static void InitWts(float[] w, float amount = 0.1f)
        {
            var rand = Rand2.Uniform(w.Length);
            for (int i = 0; i < w.Length; i++)
                w[i] = amount * rand[i];
        }

        /// <summary>
        /// Tries to find the best direction for the gradient.
        /// Gradients are Summed, ok if using Grad Norm
        /// </summary>
        /// <param name="values">Gradients to find best direction for</param>
        /// <returns>Best direction for gradients</returns>
        public static float BestDirection(List<float> values)
        {
            float plus = 0, minus = 0;
            int pc = 0, mc = 0;
            // Loop each gradient value
            foreach(var val in values)
            {
                // Ensum & Count Positive Grad
                if (val > 0)
                {
                    plus += val;
                    pc++;
                }
                // Ensum & Count Negative Grad
                else if (val < 0)
                {
                    minus += val;
                    mc++;
                }
            }
            // Find Percent Positve & Negative
            var ppct = (float)pc / values.Count;
            var mpct = (float)mc / values.Count;
            // If less than 70% of one direction of the other: Just return the sum
            if (Math.Max(ppct, mpct) < 0.7)
                return plus + minus;
            // If it favors positive gradient: decay the negative gradient
            else if (ppct > mpct)
                return plus + minus / 2;
            // If it favors negative gradient: decay the positive gradient
            return plus / 2 + minus;
        }
        
        /// <summary>
        /// Multiplication on 'values' & 'mult'
        /// </summary>
        /// <param name="values">multiplication Values</param>
        /// <param name="mult">Value to multiply by</param>
        /// <returns>New array of values multiplied</returns>
        public static float[] Multiply(float[] values, float mult)
        {
            var outs = new float[values.Length];
            for (int i = 0; i < values.Length; i++)
                outs[i] = values[i] * mult;
            return outs;
        }
        /// <summary>
        /// Multiplication on 'values' & 'mult'
        /// </summary>
        /// <param name="values">multiplication Values</param>
        /// <param name="mult">Value to multiply by</param>
        /// <returns>New array of values multiplied</returns>
        public static float[][] Multiply(float[][] values, float mult)
        {
            var outs = new float[values.Length][];
            for (int i = 0; i < values.Length; i++)
                outs[i] = Multiply(values[i], mult);
            return outs;
        }
        /// <summary>
        /// Multiplication on 'values' & 'mult'
        /// </summary>
        /// <param name="values">multiplication Values</param>
        /// <param name="mult">Value to multiply by</param>
        /// <returns>New array of values multiplied</returns>
        public static float[][][] Multiply(float[][][] values, float mult)
        {
            var outs = new float[values.Length][][];
            for (int i = 0; i < values.Length; i++)
                outs[i] = Multiply(values[i], mult);
            return outs;
        }
        /// <summary>
        /// Multiplication on 'values' & 'mult'
        /// </summary>
        /// <param name="values">multiplication Values</param>
        /// <param name="mult">Value to multiply by</param>
        /// <returns>New array of values multiplied</returns>
        public static float[][][][] Multiply(float[][][][] values, float mult)
        {
            var outs = new float[values.Length][][][];
            for (int i = 0; i < values.Length; i++)
                outs[i] = Multiply(values[i], mult);
            return outs;
        }

        /// <summary>
        /// In place Multiplication on 'values'
        /// </summary>
        /// <param name="values">Inplace multiplication</param>
        /// <param name="mult">Value to multiply by</param>
        public static void MultiplyInPlace(float[] values, float mult)
        {
            for (int i = 0; i < values.Length; i++)
                values[i] *= mult;
        }
        /// <summary>
        /// In place Multiplication on 'values'
        /// </summary>
        /// <param name="values">Inplace multiplication</param>
        /// <param name="mult">Value to multiply by</param>
        public static void MultiplyInPlace(float[][] values, float mult)
        {
            for (int i = 0; i < values.Length; i++)
                MultiplyInPlace(values[i], mult);
        }
        /// <summary>
        /// In place Multiplication on 'values'
        /// </summary>
        /// <param name="values">Inplace multiplication</param>
        /// <param name="mult">Value to multiply by</param>
        public static void MultiplyInPlace(float[][][] values, float mult)
        {
            for (int i = 0; i < values.Length; i++)
                MultiplyInPlace(values[i], mult);
        }
        /// <summary>
        /// In place Multiplication on 'values'
        /// </summary>
        /// <param name="values">Inplace multiplication</param>
        /// <param name="mult">Value to multiply by</param>
        public static void MultiplyInPlace(float[][][][] values, float mult)
        {
            for (int i = 0; i < values.Length; i++)
                MultiplyInPlace(values[i], mult);
        }

        // Module identifier
        public override string ToString()
        => _name0 + _name;

        /// <summary>
        /// This should be the inherited parent for all modules except for the outer most module you
        /// want to train, then use NetModule. This is for Module nesting and new created modules.
        /// </summary>
        /// <param name="name0">Nested Modules Parents Names</param>
        /// <param name="name">Name of Module</param>
        public Module(string name0, string name)
        {
            _name0 = "";
            if (name0 != null)
            {
                // Remove Parents Parenthesis
                var pos = name0.IndexOf("(");
                if (pos >= 0)
                    name0 = name0.Substring(0, pos);
                else if (name0.IndexOf(")") >= 0)
                    throw new Exception("Unbalance parenthisis in 'name0'.");
                // Delimiter
                _name0 = name0 + ".";
            }
            _name = name;
            _modules = new List<Module>();
        }
    }
}
