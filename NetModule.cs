using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using MessagePack;
using MessagePack.Resolvers;

namespace MiniNet
{
    /// <summary>
    /// When you want to inference or train your network: inherit this class.
    /// It has specific functions over Module for inference & training.
    /// </summary>
    /// <typeparam name="T">Type of the output from the forwards pass: public abstract Tensor(T) _Fwd(Tensor(object) x) </typeparam>
    public abstract class NetModule<T> : Module
    {
        // Smooth gradients length
        public readonly int _smoothBatch;
        // Parallize workers to the number of cores on CPU
        protected readonly ParallelOptions _options = new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount
        };

        /// <summary>
        /// Each NetModule must override this function.
        /// Trying to make it easier so you don't have to call the AutoGrad() constructor
        /// </summary>
        /// <param name="x">Input tensor into network</param>
        /// <returns>Output tensor of network</returns>
        public abstract Tensor<T> _Fwd(Tensor<object> x);

        // Trying to make it easier so you don't have to call the AutoGrad() constructor 
        public Tensor<T> Forward(AutoGrad.BackwardNode x, bool requiresGrad)
            => _Fwd(new AutoGrad().AddRoot(x, requiresGrad));
        public Tensor<T> Forward(object x)
            => _Fwd(new AutoGrad().AddRoot(x));
        public Tensor<T> Forward(bool notused, params object[] inputs)
            => _Fwd(new AutoGrad().AddRootCollection(inputs));
        public Tensor<T> Forward(params AutoGrad.BackwardNode[] inputs)
            => _Fwd(new AutoGrad().AddRootCollection(inputs));

        /// <summary>
        /// Multithreaded Network train function
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="minEpoch">Minimum Number of epochs</param>
        /// <param name="over">Early Stopping: Reload best network after 'over' times epochs</param>
        /// <param name="iter">The number of times the learning rate is increased</param>
        /// <param name="x">Inputs into network</param>
        /// <param name="labels">Targets to reach</param>
        /// <param name="ctr">Output of the number of epochs reached</param>
        /// <param name="best">Output best score</param>
        /// <param name="noiseProb">Probability from 0->1 that noise is inputted into network</param>
        /// <param name="noiseSelect">Probability from 0->1 that the item is selected to add noise</param>
        /// <param name="noiseAmount">Max amount of noise from 0->1. noiseAmount=0.2 means +/-0.2</param>
        /// <param name="randAlwaysInputs">If using noise always noise the inputs</param>
        /// <param name="fitAgainOk">If loading in Network, it can redo the X & Y Normalization</param>
        /// <param name="cropiter">Early stopping in 'lrSchedule' list if not better net is found</param>
        /// <param name="savePath">Path to save best network</param>
        public void TrainIt(float lr, int minEpoch, int over, int iter, float[][] x, float[][] labels,
                            out int ctr, out double best, NormInf normx = null, NormInf normy = null, 
                            double noiseProb = 0.3, double noiseSelect = 0.3,
                            double noiseAmount = 0.15, bool randAlwaysInputs = false, bool fitAgainOk = false,
                            bool cropiter = false, string savePath = null)
        {
            List<float[][]> yls = null;
            float[][] x2 = null, labels2 = null;
            List<List<int>> ils = null;
            List<int[]> labelsMask = null;
            // True: run backwards pass, False: only run forwards pass
            var backward = true;
            // Parallize workers to the number of cores on CPU
            var workers = _options.MaxDegreeOfParallelism;
            // Get more batches for each thread
            Func<List<int>, List<List<int>>, List<List<int>>> ReBatchIx = (ix, iils) =>
            {
                var ls = new List<List<int>>(iils.Count);
                var start = 0;
                foreach (var ii in iils)
                {
                    var len = start + ii.Count;
                    var l = new List<int>(len - start);
                    ls.Add(l);
                    for (int i = start; i < len; i++)
                        l.Add(ix[i]);
                    start += ii.Count;
                }
                if (start != ix.Count) throw new Exception("start != ix.Count");
                return ls;
            };
            // Create groups for each thread that sum up to the data set size
            Func<int, List<int>, List<List<int>>> InitBatches = (n, ix) =>
            {
                var min = ix.Count / n;
                var lens = new bool[n].Select(no => min).ToList();
                foreach (var r in Enumerable.Range(0, ix.Count % n))
                    lens[r]++;
                return lens.Select(l => new int[l].ToList()).ToList();
            };
            // Run Network
            Action<int> EachBatch = batchid =>
            {
                var iils = ils[batchid];                        // Indexs for this batch
                var xx = iils.Select(i => x2[i]).ToArray();     // x batch
                // Forward Pass
                var y = Forward(xx);
                // Get Outputs of Network
                var yval = (float[][])y;
                // Backward Pass
                if (backward)
                {
                    // Deep copy labels
                    var lab = iils.Select(i => labels2[i].ToArray()).ToArray();
                    var mask = iils.Select(i => labelsMask[i]).ToArray();
                    // There may not be targets for some values: make sure they have a gradient of zero
                    for (int i = 0; i < iils.Count; i++)
                    {
                        var m = mask[i];
                        if (m == null) continue;
                        var l = lab[i];
                        var yi = yval[i];
                        foreach (var j in m)
                            l[j] = yi[j];
                    }
                    // Backwards pass
                    var loss = Criterion.Conservative((Tensor<float[][]>)(object)y, lab);
                    loss.Backward();
                }
                // Save network output to memory
                yls[batchid] = yval;
            };
            // Get Error
            Func<List<float[][]>, List<List<int>>, float[][], List<float>> GetEr = (y_, ii_, lab) =>
            {
                // flatten from batches
                var ii = ii_.SelectMany(i => i).ToList();
                var y0 = y_.SelectMany(yi => yi).ToList();
                // Unshuffle y to make sure it is in the correct order
                var yy = new float[y0.Count][];
                for (int i = 0; i < ii.Count; i++)
                    yy[ii[i]] = y0[i];
                // Deep copy labels
                lab = lab.Select(s => s.ToArray()).ToArray();
                // There may not be targets for some values: make sure they have a gradient of zero
                for (int i = 0; i < lab.Length; i++)
                {
                    var l = lab[i];
                    var yi = yy[i];
                    for (int j = 0; j < l.Length; j++)
                        // if no target for value: make sure it has a gradient of zero
                        if (float.IsNaN(l[j]))
                            l[j] = yi[j];
                }
                // Get Error
                return Criterion.GetError(Criterion.Conservative(yy, lab));
            };
            // Mask of where there is no label for target
            // Some targets don't have labels (defined as Nan): make sure it has a gradient of zero
            Func<float[][], List<int[]>> GetMask = l =>
            {
                var d = new List<int[]>(l.Length);
                for (int i = 0; i < l.Length; i++)
                {
                    var li = l[i];
                    var lss = new List<int>();
                    for (int j = 0; j < li.Length; j++)
                        if (float.IsNaN(li[j]))
                            lss.Add(j);

                    if (lss.Count == 0)
                        d.Add(null);
                    else
                        d.Add(lss.ToArray());
                }
                return d;
            };
            // Make sure there are enouph values
            if (x.Length < workers) throw new Exception("The size of 'workers' is too big becuase of the number of samples.");
            // Fit the Norms ..........
            if ((normx != null) != (normy != null))
            {
                if (normx == null)
                    throw new Exception("There is normalization for y, but not for x.");
                else
                    throw new Exception("There is normalization for x, but not for y.");
            }
            if (normx != null)
                x = normx.Fit(x, fitAgainOk);
            if (normy != null)
                labels = normy.Fit(labels, fitAgainOk);
            // Indexs
            var idx = Enumerable.Range(0, x.Length).ToList();
            // batch of indexs: Just have the correct array sizes only
            ils = InitBatches(workers, idx);
            // Mask of where there is no label for target
            labelsMask = GetMask(labels);
            // Inputs noise
            var select2d = (float)Math.Sqrt(noiseSelect);
            var noiseX = new Rand2.Noise(select2d, noiseAmount);
            // Gradients noise
            var noiseGrad = new Rand2.Noise(select2d, noiseAmount);
            // Error Scoreing function
            var section = new Sectionize(x.Length, 20);
            // All errors
            var errors = new List<double>(iter * minEpoch);
            best = double.PositiveInfinity;     // best score
            object bestParams = null;           // best network
            ctr = 0;                            // epoch counter
            var itv = 0;                        // used to exit early
            var better = true;                  // used to exit early
            Train();                            // Empty Function
            //Backward(x, labels, 0);
            // Iterations to run
            for (int i = 0; i < iter; i++)
            {
                var c = 0;
                var cnt = 0;
                // Learning Rate Warm-Up: Increase Learning Rage
                if (i > 0)
                    lr = Math.Min(lr * (float)Math.Sqrt(10), 1.3f);
                // exit early
                itv++;
                if (better) itv = 0;
                if (itv >= 5 && cropiter) break;
                better = false;
                // c<over: give up if no better network
                // ctr<minEpoch: make sure minEpochs have ran though
                while (c < over || ctr < minEpoch)
                {
                    ctr++;
                    cnt++;
                    // Get new shuffled batch
                    Utilz.Shuffle(idx);
                    ils = ReBatchIx(idx, ils);
                    // Set to unnoised values
                    x2 = x;
                    labels2 = labels;
                    // Noise
                    var randinputs = false;
                    // Probability for noise
                    if (Rand2.Uniform() <= noiseProb)      // Use Noise
                    {
                        // Noise inputs probabiltiy
                        randinputs = Rand2.Uniform() <= 0.5;
                        // Add Noise to inputs
                        if (randinputs || randAlwaysInputs)
                            x2 = noiseX.Add(x2);
                        // Add Noise in gradients
                        if (!randinputs)
                            labels2 = noiseGrad.Add(labels2);
                        randinputs = randinputs || randAlwaysInputs;
                    }
                    // MultiThread
                    yls = new float[workers][][].ToList();
                    // Run in Parallel
                    Parallel.For(0, workers, _options, EachBatch);
                    // If random inputs were used: Set input values to unnoised value to get the error score
                    if (randinputs)
                    {
                        backward = false;           // Do not run backwards pass
                        // Set to unnoised values
                        x2 = x;
                        yls = new float[workers][][].ToList();
                        // Run in Parallel
                        Parallel.For(0, workers, _options, EachBatch);
                        backward = true;            // Now run backwards pass
                    }
                    // Get the Error
                    var ers = GetEr(yls, ils, labels);
                    // Get the Error Score
                    var erScore = section.Run(ers);
                    var er = erScore;
                    errors.Add(er);
                    // If better error score: save the parameters
                    if (er < best)
                    {
                        best = er;
                        bestParams = Parameters();
                        c = 0;
                        better = true;
                    }
                    // else not better: inc of c<over
                    else
                        c++;
                    // Step the network parameters
                    Step(lr);
                    // Display to console
                    if (ctr % 5 == 0)
                        Console.Write($"\r   Epochs:{ctr} ErScore:{best:E3}...     ");
                }
                // Load best network
                LoadState(bestParams);
                // Save best network to disk
                if (savePath != null)
                    Save(savePath);
            }
            // Load best network
            LoadState(bestParams);
            // Display to console
            Console.WriteLine($"\rFinished Epochs: {ctr} ErScore:{best:E3}.        ");
        }
        /// <summary>
        /// Multithreaded Network train function
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="minEpoch">Minimum Number of epochs</param>
        /// <param name="over">Early Stopping: Reload best network after 'over' times epochs</param>
        /// <param name="iter">The number of times the learning rate is increased</param>
        /// <param name="x">Inputs into network</param>
        /// <param name="labels">Targets to reach</param>
        /// <param name="ctr">Output of the number of epochs reached</param>
        /// <param name="best">Output best score</param>
        /// <param name="noiseProb">Probability from 0->1 that noise is inputted into network</param>
        /// <param name="noiseSelect">Probability from 0->1 that the item is selected to add noise</param>
        /// <param name="noiseAmount">Max amount of noise from 0->1. noiseAmount=0.2 means +/-0.2</param>
        /// <param name="randAlwaysInputs">If using noise always noise the inputs</param>
        /// <param name="fitAgainOk">If loading in Network, it can redo the X & Y Normalization</param>
        /// <param name="cropiter">Early stopping in 'lrSchedule' list if not better net is found</param>
        /// <param name="savePath">Path to save best network</param>
        public void TrainIt(float lr, int minEpoch, int over, int iter, float[][][] x, float[][] labels,
                            out int ctr, out double best, NormInf normx = null, NormInf normy = null,
                            double noiseProb = 0.3, double noiseSelect = 0.3,
                            double noiseAmount = 0.15, bool randAlwaysInputs = false, bool fitAgainOk = false,
                            bool cropiter = false, string savePath = null)
        {
            List<float[][]> yls = null;
            float[][][] x2 = null;
            float[][] labels2 = null;
            List<List<int>> ils = null;
            List<int[]> labelsMask = null;
            // True: run backwards pass, False: only run forwards pass
            var backward = true;
            // Parallize workers to the number of cores on CPU
            var workers = _options.MaxDegreeOfParallelism;
            // Get more batches for each thread
            Func<List<int>, List<List<int>>, List<List<int>>> ReBatchIx = (ix, iils) =>
            {
                var ls = new List<List<int>>(iils.Count);
                var start = 0;
                foreach (var ii in iils)
                {
                    var len = start + ii.Count;
                    var l = new List<int>(len - start);
                    ls.Add(l);
                    for (int i = start; i < len; i++)
                        l.Add(ix[i]);
                    start += ii.Count;
                }
                if (start != ix.Count) throw new Exception("start != ix.Count");
                return ls;
            };
            // Create groups for each thread that sum up to the data set size
            Func<int, List<int>, List<List<int>>> InitBatches = (n, ix) =>
            {
                var min = ix.Count / n;
                var lens = new bool[n].Select(no => min).ToList();
                foreach (var r in Enumerable.Range(0, ix.Count % n))
                    lens[r]++;
                return lens.Select(l => new int[l].ToList()).ToList();
            };
            // Run Network
            Action<int> EachBatch = batchid =>
            {
                var iils = ils[batchid];                        // Indexs for this batch
                var xx = iils.Select(i => x2[i]).ToArray();     // x batch
                // Forward Pass
                var y = Forward(xx);
                // Get Outputs of Network
                var yval = (float[][])y;
                // Backward Pass
                if (backward)
                {
                    // Deep copy labels
                    var lab = iils.Select(i => labels2[i].ToArray()).ToArray();
                    var mask = iils.Select(i => labelsMask[i]).ToArray();
                    // There may not be targets for some values: make sure they have a gradient of zero
                    for (int i = 0; i < iils.Count; i++)
                    {
                        var m = mask[i];
                        if (m == null) continue;
                        var l = lab[i];
                        var yi = yval[i];
                        foreach (var j in m)
                            l[j] = yi[j];
                    }
                    // Backwards pass
                    var loss = Criterion.Conservative((Tensor<float[][]>)(object)y, lab);
                    loss.Backward();
                }
                // Save network output to memory
                yls[batchid] = yval;
            };
            // Get Error
            Func<List<float[][]>, List<List<int>>, float[][], List<float>> GetEr = (y_, ii_, lab) =>
            {
                // flatten from batches
                var ii = ii_.SelectMany(i => i).ToList();
                var y0 = y_.SelectMany(yi => yi).ToList();
                // Unshuffle y to make sure it is in the correct order
                var yy = new float[y0.Count][];
                for (int i = 0; i < ii.Count; i++)
                    yy[ii[i]] = y0[i];
                // Deep copy labels
                lab = lab.Select(s => s.ToArray()).ToArray();
                // There may not be targets for some values: make sure they have a gradient of zero
                for (int i = 0; i < lab.Length; i++)
                {
                    var l = lab[i];
                    var yi = yy[i];
                    for (int j = 0; j < l.Length; j++)
                        // if no target for value: make sure it has a gradient of zero
                        if (float.IsNaN(l[j]))
                            l[j] = yi[j];
                }
                // Get Error
                return Criterion.GetError(Criterion.Conservative(yy, lab));
            };
            // Mask of where there is no label for target
            // Some targets don't have labels (defined as Nan): make sure it has a gradient of zero
            Func<float[][], List<int[]>> GetMask = l =>
            {
                var d = new List<int[]>(l.Length);
                for (int i = 0; i < l.Length; i++)
                {
                    var li = l[i];
                    var lss = new List<int>();
                    for (int j = 0; j < li.Length; j++)
                        if (float.IsNaN(li[j]))
                            lss.Add(j);

                    if (lss.Count == 0)
                        d.Add(null);
                    else
                        d.Add(lss.ToArray());
                }
                return d;
            };
            // Make sure there are enouph values
            if (x.Length < workers) throw new Exception("The size of 'workers' is too big becuase of the number of samples.");
            // Fit the Norms ..........
            if ((normx != null) != (normy != null))
            {
                if (normx == null)
                    throw new Exception("There is normalization for y, but not for x.");
                else
                    throw new Exception("There is normalization for x, but not for y.");
            }
            if (normx != null)
                x = normx.Fit(x, fitAgainOk);
            if (normy != null)
                labels = normy.Fit(labels, fitAgainOk);
            // Indexs
            var idx = Enumerable.Range(0, x.Length).ToList();
            // batch of indexs: Just have the correct array sizes only
            ils = InitBatches(workers, idx);
            // Mask of where there is no label for target
            labelsMask = GetMask(labels);
            // Inputs noise
            var select2d = (float)Math.Sqrt(noiseSelect);
            var select3d = (float)Math.Pow(noiseSelect, 1 / 3.0);
            var noiseX = new Rand2.Noise(select3d, noiseAmount);
            // Gradients noise
            var noiseGrad = new Rand2.Noise(select2d, noiseAmount);
            // Error Scoreing function
            var section = new Sectionize(x.Length, 20);
            // All errors
            var errors = new List<double>(iter * minEpoch);
            best = double.PositiveInfinity;     // best score
            object bestParams = null;           // best network
            ctr = 0;                            // epoch counter
            var itv = 0;                        // used to exit early
            var better = true;                  // used to exit early
            Train();                            // Empty Function
            //Backward(x, labels, 0);
            // Iterations to run
            for (int i = 0; i < iter; i++)
            {
                var c = 0;
                var cnt = 0;
                // Learning Rate Warm-Up: Increase Learning Rage
                if (i > 0)
                    lr = Math.Min(lr * (float)Math.Sqrt(10), 1.3f);
                // exit early
                itv++;
                if (better) itv = 0;
                if (itv >= 5 && cropiter) break;
                better = false;
                // c<over: give up if no better network
                // ctr<minEpoch: make sure minEpochs have ran though
                while (c < over || ctr < minEpoch)
                {
                    ctr++;
                    cnt++;
                    // Get new shuffled batch
                    Utilz.Shuffle(idx);
                    ils = ReBatchIx(idx, ils);
                    // Set to unnoised values
                    x2 = x;
                    labels2 = labels;
                    // Noise
                    var randinputs = false;
                    // Probability for noise
                    if (Rand2.Uniform() <= noiseProb)      // Use Noise
                    {
                        // Noise inputs probabiltiy
                        randinputs = Rand2.Uniform() <= 0.5;
                        // Add Noise to inputs
                        if (randinputs || randAlwaysInputs)
                            x2 = noiseX.Add(x2);
                        // Add Noise in gradients
                        if (!randinputs)
                            labels2 = noiseGrad.Add(labels2);
                        randinputs = randinputs || randAlwaysInputs;
                    }
                    // MultiThread
                    yls = new float[workers][][].ToList();
                    // Run in Parallel
                    Parallel.For(0, workers, _options, EachBatch);
                    // If random inputs were used: Set input values to unnoised value to get the error score
                    if (randinputs)
                    {
                        backward = false;           // Do not run backwards pass
                        // Set to unnoised values
                        x2 = x;
                        yls = new float[workers][][].ToList();
                        // Run in Parallel
                        Parallel.For(0, workers, _options, EachBatch);
                        backward = true;            // Now run backwards pass
                    }
                    // Get the Error
                    var ers = GetEr(yls, ils, labels);
                    // Get the Error Score
                    var erScore = section.Run(ers);
                    var er = erScore;
                    errors.Add(er);
                    // If better error score: save the parameters
                    if (er < best)
                    {
                        best = er;
                        bestParams = Parameters();
                        c = 0;
                        better = true;
                    }
                    // else not better: inc of c<over
                    else
                        c++;
                    // Step the network parameters
                    Step(lr);
                    // Display to console
                    if (ctr % 3 == 0)
                        Console.Write($"\r   Epochs:{ctr} ErScore:{best:E3}...     ");
                }
                // Load best network
                LoadState(bestParams);
                // Save best network to disk
                if (savePath != null)
                    Save(savePath);
            }
            // Load best network
            LoadState(bestParams);
            // Display to console
            Console.WriteLine($"\rFinished Epochs: {ctr} ErScore:{best:E3}.        ");
        }

        /// <summary>
        /// Multithreaded Step function that can be call syncronosly.
        /// Updates the parameters of the network
        /// </summary>
        /// <param name="lr">Learning Rate</param>
        /// <param name="div">Not used.</param>
        public override void Step(float lr, float div = 1)
        {
            var ds = new float[_modules.Count];
            Action<int> SmoothMax = i => ds[i] = _modules[i].SmoothMax(_smoothBatch);
            Action<int> Step = i => _modules[i].Step(lr, div);
            // Run SmoothMax() for each Module
            Parallel.For(0, _modules.Count, _options, SmoothMax);
            // Grad Norm: use the max
            div = ds.Max() + 1e-12f;
            // Update parameters of the network
            Parallel.For(0, _modules.Count, _options, Step);
        }

        /// <summary>
        /// Total Number of Parameters the Network uses.
        /// </summary>
        /// <returns>Total Number of Parameters in Network</returns>
        public override long NumParams()
        => _modules.Select(m => m.NumParams()).Sum();

        /// <summary>
        /// Deep Copy of Network Parameters
        /// </summary>
        /// <returns>Deep Copy of Network Parameters as object</returns>
        public override object Parameters()
        => _modules.Select(m => m.Parameters()).ToList();

        /// <summary>
        /// Load Parameters into the network
        /// </summary>
        /// <param name="prms_">Parameter object to load</param>
        public override void LoadState(object prms_)
        {
            // Validation
            var prms = (List<object>)prms_;
            if (prms.Count != _modules.Count)
                throw new Exception($"There should be ({_modules.Count}) items in 'prms', but there is ({prms.Count}).");
            var ix = -1;
            // Load Parameters
            _modules.ForEach(m => m.LoadState(prms[++ix]));
        }

        /// <summary>
        /// Load State from file.
        /// </summary>
        /// <param name="path">Path to load the state from</param>
        /// <param name="okToFail">If set to True and fails it won't through an error.</param>
        /// <returns>If Successfull</returns>
        public bool LoadState(string path, bool okToFail = false)
        {
            // If problem: load these weights
            var wts = Parameters();
            // If file does not exist
            if (!System.IO.File.Exists(path))
            {
                Console.WriteLine($"File does Not exist: {path}");
                return false;
            }
            try
            {
                // If JSON file
                if (path.Trim().ToLower().EndsWith(".json"))
                {
                    try
                    {
                        Console.Write("\rTrying to load JSON file.");
                        // Load JSON file
                        var serialtx = System.IO.File.ReadAllText(path);
                        // Deserialize
                        var prmsjson = JsonConvert.DeserializeObject<List<object>>(serialtx, _settings);
                        // Validation
                        if (prmsjson.Count != _modules.Count)
                            throw new Exception($"There should be ({_modules.Count}) items in 'prmsjson', but there is ({prmsjson.Count}).");
                        var ix = -1;
                        // Load JSON
                        _modules.ForEach(m => m.LoadStateJSON(prmsjson[++ix]));
                        Console.Write("\r                                ");
                        Console.Write("\r");
                        // Success!
                        return true;
                    }
                    catch (Exception e)
                    {
                        // Error load in previous weights
                        LoadState(wts);
                        Console.WriteLine($"Error in loading weights json: {path}");
                        // If set to false: throw error
                        if (!okToFail) throw e;
                        return false;
                    }
                }
                // Else MessagePack file ...........
                // MessagePack options
                var opt = MessagePackSerializerOptions.Standard
                            .WithResolver(TypelessContractlessStandardResolver.Instance);  // embed CLR type names
                // Buffer in bytes
                byte[] fileBuf = System.IO.File.ReadAllBytes(path);
                // Deserialize
                var prms = MessagePackSerializer.Deserialize<List<object>>(fileBuf.AsMemory(), opt);
                // Load State
                LoadState(prms);
                // Success!
                return true;
            }
            catch (Exception e)
            {
                // Error load in previous weights
                LoadState(wts);
                Console.WriteLine($"Error in loading weights: {path}");
                // If set to false: throw error
                if (!okToFail) throw e;
            }
            return false;
        }

        /// <summary>
        /// Save Network Parameters as MessagePack File
        /// </summary>
        /// <param name="path">Path to save to</param>
        public void Save(string path)
        {
            // MessagePack options
            var opt = MessagePackSerializerOptions.Standard
                        .WithResolver(TypelessContractlessStandardResolver.Instance);  // embed CLR type names
            // Buffer in bytes
            byte[] blob = MessagePackSerializer.Serialize((List<object>)Parameters(), opt);
            // Safe Save: in case of any interruption
            var pos = path.LastIndexOf('.');
            var p = new[]
            {
                path.Substring(0, pos),
                path.Substring(pos)
            };
            var safe = $"{p[0]}-SafeNew{p[1]}";
            // Save bytes
            System.IO.File.WriteAllBytes(safe, blob);
            // Safe Save
            if (System.IO.File.Exists(path))
                System.IO.File.Delete(path);
            System.IO.File.Move(safe, path);
            //var serialtx = JsonConvert.SerializeObject(Parameters(), Formatting.Indented, Module._settings);
            //System.IO.File.WriteAllText(path, serialtx);
        }

        /// <summary>
        /// Empty Function. Does nothing yet.
        /// </summary>
        public void Eval() { }
        /// <summary>
        /// Empty Function. Does nothing yet.
        /// </summary>
        public void Train() { }

        /// <summary>
        /// Check if Array is Equal
        /// </summary>
        /// <typeparam name="S">Type of 'a' & 'b'/typeparam>
        /// <param name="a">Array 1 to check</param>
        /// <param name="b">Array 2 to check</param>
        /// <returns>True if Array are equal, false otherwise.</returns>
        public static bool ArrayEqual<S>(S[][] a, S[][] b)
        {
            // Check length
            if (a.Length != b.Length)
                return false;
            // Check values
            for (int i = 0; i < a.Length; i++)
                if (!ArrayEqual(a[i], b[i]))
                    return false;
            return true;
        }
        /// <summary>
        /// Check if Array is Equal
        /// </summary>
        /// <typeparam name="S">Type of 'a' & 'b'/typeparam>
        /// <param name="a">Array 1 to check</param>
        /// <param name="b">Array 2 to check</param>
        /// <returns>True if Array are equal, false otherwise.</returns>
        public static bool ArrayEqual<S>(S[] a, S[] b)
        {
            // Check length
            if (a.Length != b.Length) 
                return false;
            // Check values
            for (int i = 0; i < a.Length; i++)
                if (!a[i].Equals(b[i]))
                    return false;
            return true;
        }

        /// <summary>
        /// When you want to inference or train your network: inherit this class.
        /// It has specific functions over Module for inference & training.
        /// </summary>
        /// <param name="nInput">In Features size</param>
        /// <param name="nOut">Out Features size</param>
        /// <param name="smoothBatchLen">Smooth Length for gradients</param>
        /// <param name="name0">Parents Names</param>
        /// <param name="name">Name of this NetModule</param>
        public NetModule(int smoothBatchLen, string name0, string name)
            :base(name0, name)
        {
            _smoothBatch = smoothBatchLen;
        }

        // These functions would never be called if used correctly. Be sure to use GetAllModules(_modules).
        // Since this is a module of modules these functions should not be called.
        public override void LoadStateJSON(object items)
            => throw new NotSupportedException("Did you forget to call GetAllModules(_modules) at the end of the constructor?");
        public override float SmoothMax(int len)
            => throw new NotSupportedException("Did you forget to call GetAllModules(_modules) at the end of the constructor?");
    }
}
