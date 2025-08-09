using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using System.Windows.Media;

    /// <summary>
    /// Utility Functions
    /// </summary>
    static class Utilz
    {
        /// <summary>
        /// Transpose first 2 dimentions
        /// </summary>
        /// <typeparam name="T">Type of source</typeparam>
        /// <param name="source">Values to transpose</param>
        /// <returns>Values Transposed</returns>
        public static IEnumerable<List<T>> Transpose<T>(IEnumerable<IEnumerable<T>> source) =>
            source.SelectMany(inner => inner.Select((item, index) => new { item, index }))
            .GroupBy(i => i.index, i => i.item).Select(g => g.ToList());
        /// <summary>
        /// Transpose Array first 2 dimentions
        /// </summary>
        /// <typeparam name="T">Type of source</typeparam>
        /// <param name="source">Values to transpose</param>
        /// <returns>Array Transposed</returns>
        public static T[][] TransposeArray<T>(T[][] source)
        {
            var outs = new T[source[0].Length][];
            // Loop by output dim0
            for (int j = 0; j < outs.Length; j++)
            {
                var o = outs[j] = new T[source.Length];
                // Loop by output dim1
                for (int i = 0; i < o.Length; i++)
                    o[i] = source[i][j];
            }
            return outs;
        }

        /// <summary>
        /// Transpose Array first 2 dimentions
        /// </summary>
        /// <typeparam name="T">Type of source</typeparam>
        /// <param name="source">Values to transpose</param>
        /// <returns>Array Transposed with inner dim being a List</returns>
        public static List<T>[] TransposeInnerList<T>(T[][] source)
        {
            var outs = new List<T>[source[0].Length];
            // Loop by output dim0
            for (int j = 0; j < outs.Length; j++)
            {
                var o = outs[j] = new T[source.Length].ToList();
                // Loop by output dim1
                for (int i = 0; i < o.Count; i++)
                    o[i] = source[i][j];
            }
            return outs;
        }

        /// <summary>
        /// Transpose Dimentions on dim0 & dim1
        /// </summary>
        /// <typeparam name="T">Type of values</typeparam>
        /// <param name="values">Values to transpose</param>
        /// <param name="dim0">First Dimention to transpose</param>
        /// <param name="dim1">Second Dimention to transpose</param>
        /// <returns>Tranposed Array</returns>
        public static T[][][] Transpose<T>(T[][][] values, int dim0, int dim1)
        {
            // Validation
            if (new[] { dim0, dim1 }.Any(d => d < 0 || d > 2))
                throw new IndexOutOfRangeException("Dimensions are out of range.");
            if (dim0 == dim1)
                throw new Exception("Dimentions are the same.");
            // Make sure in sorted order
            if (dim0 > dim1)
            {
                var temp = dim0;
                dim0 = dim1;
                dim1 = temp;
            }
            // Should be faster. Less indexing
            if (dim0 == 0 && dim1 == 1)
                return TransposeArray(values);
            // Should be faster. Less indexing
            else if (dim0 == 1 && dim1 == 2)
                return values.Select(ar => TransposeArray(ar)).ToArray();
            // Transpose
            return TransposeSub(values, dim0, dim1);
        }
        /// <summary>
        /// Sub Function (No Validation): Transpose Dimentions on dim0 & dim1
        /// </summary>
        /// <typeparam name="T">Type of values</typeparam>
        /// <param name="values">Values to transpose</param>
        /// <param name="dim0">First Dimention to transpose</param>
        /// <param name="dim1">Second Dimention to transpose</param>
        /// <returns>Tranposed Array</returns>
        static T[][][] TransposeSub<T>(T[][][] values, int dim0, int dim1)
        {
            unsafe
            {
                // Lengths of each dim
                var i = values.Length;
                var j = values[0].Length;
                var k = values[0][0].Length;
                var dim = new int*[] { &i, &j, &k };
                // Transpose Dims by pointer
                var tmp = dim[dim0];
                dim[dim0] = dim[dim1];
                dim[dim1] = tmp;
                // Indexs for output array
                var x = dim[0];
                var y = dim[1];
                var z = dim[2];
                // Initialize output Array
                var outs = new bool[*x].Select(no => new bool[*y].Select(n => new T[*z]).ToArray()).ToArray();
                // Loop values dim0
                for (i = 0; i < values.Length; i++)
                {
                    var vi = values[i];
                    // Loop values dim1
                    for (j = 0; j < vi.Length; j++)
                    {
                        var vij = vi[j];
                        // Loop values dim2
                        for (k = 0; k < vij.Length; k++)
                            outs[*x][*y][*z] = vij[k];
                    }
                }
                return outs;
            }
        }
        /// <summary>
        /// Transpose Dimentions on dim0 & dim1
        /// </summary>
        /// <typeparam name="T">Type of values</typeparam>
        /// <param name="values">Values to transpose</param>
        /// <param name="dim0">First Dimention to transpose</param>
        /// <param name="dim1">Second Dimention to transpose</param>
        /// <returns>Tranposed Array</returns>
        public static T[][][][] Transpose<T>(T[][][][] values, int dim0, int dim1)
        {
            // Validate
            if (new[] { dim0, dim1 }.Any(d => d < 0 || d > 3))
                throw new IndexOutOfRangeException("Dimensions are out of range.");
            if (dim0 == dim1)
                throw new Exception("Dimentions are the same.");
            // Make sure in sorted order
            if (dim0 > dim1)
            {
                var temp = dim0;
                dim0 = dim1;
                dim1 = temp;
            }
            // Should be faster. Less indexing
            if (dim0 == 0 && dim1 == 1)
                return TransposeArray(values);
            // Should be faster. Less indexing
            else if (dim0 == 1 && dim1 == 2)
                return values.Select(ar => TransposeArray(ar)).ToArray();
            // Should be faster. Less indexing
            else if (dim0 == 2 && dim1 == 3)
                return values.Select(l => l.Select(ar => TransposeArray(ar)).ToArray()).ToArray();
            // Should be faster. Less indexing
            else if (dim0 != 0 && dim1 != 0)
                 return values.Select(s => TransposeSub(s, dim0 - 1, dim1 - 1)).ToArray();
            unsafe
            {
                // values dim Lengths
                var i = values.Length;
                var j = values[0].Length;
                var k = values[0][0].Length;
                var l = values[0][0].Length;
                var dim = new int*[] { &i, &j, &k, &l };
                // Transpose on dims
                var tmp = dim[dim0];
                dim[dim0] = dim[dim1];
                dim[dim1] = tmp;
                // Indexs of output
                var w = dim[0];
                var x = dim[1];
                var y = dim[2];
                var z = dim[3];
                // Initialize Array
                var outs = new bool[*w].Select(n2 => new bool[*x].Select(no => 
                            new bool[*y].Select(n => new T[*z]).ToArray()).ToArray()).ToArray();
                // Loop values dim0
                for (i = 0; i < values.Length; i++)
                {
                    var vi = values[i];
                    // Loop values dim1
                    for (j = 0; j < vi.Length; j++)
                    {
                        var vij = vi[j];
                        // Loop values dim2
                        for (k = 0; k < vij.Length; k++)
                        {
                            var v = vij[k];
                            // Loop values dim3
                            for (l = 0; l < v.Length; l++)
                                outs[*w][*x][*y][*z] = v[l];
                        }
                    }
                }
                return outs;
            }
        }

        /// <summary>
        /// ReOrder Array: Like if you transposed multiple times
        /// </summary>
        /// <typeparam name="T">Type of values</typeparam>
        /// <param name="values">Values to Transpose</param>
        /// <param name="dim0">First Dimention to transpose</param>
        /// <param name="dim1">Second Dimention to transpose</param>
        /// <param name="dim2">Third Dimention to transpose</param>
        /// <returns>Tranposed Array</returns>
        public static T[][][] ReOrder<T>(T[][][] values, int dim0, int dim1, int dim2)
        {
            var d = new[] { dim0, dim1, dim2 };
            // Validation
            if (d.Any(m => m < 0 || m > 2))
                throw new IndexOutOfRangeException("Dimensions are out of range.");
            if (dim0 == dim1 || dim0 == dim2 || dim1 == dim2)
                throw new Exception("At least 2 Dimentions are the same.");
            // Tranposed ReOrder
            return ReOrderSub(values, dim0, dim1, dim2);
        }
        /// <summary>
        /// ReOrder Array: Like if you transposed multiple times
        /// </summary>
        /// <typeparam name="T">Type of values</typeparam>
        /// <param name="values">Values to Transpose</param>
        /// <param name="dim0">First Dimention to transpose</param>
        /// <param name="dim1">Second Dimention to transpose</param>
        /// <param name="dim2">Third Dimention to transpose</param>
        /// <param name="dim3">Forth Dimention to transpose</param>
        /// <returns>Tranposed Array</returns>
        public static T[][][][] ReOrder<T>(T[][][][] values, int dim0, int dim1, int dim2, int dim3)
        {
            var d = new[] { dim0, dim1, dim2, dim3 };
            // Validation
            if (dim0 != 0)
                throw new NotSupportedException("dim0 != values.Length");
            if (d.Any(m => m < 0 || m > 3))
                throw new IndexOutOfRangeException("Dimensions are out of range.");
            if (d.ToHashSet().Count != 4)
                throw new Exception("At least 2 Dimentions are the same.");
            // ReOrder Transpose
            return values.Select(vals => ReOrderSub(vals, dim1-1, dim2-1, dim3-1)).ToArray();
        }

        /// <summary>
        /// Sub ReOrder Array: Like if you transposed multiple times
        /// </summary>
        /// <typeparam name="T">Type of values</typeparam>
        /// <param name="values">Values to Transpose</param>
        /// <param name="dim0">First Dimention to transpose</param>
        /// <param name="dim1">Second Dimention to transpose</param>
        /// <param name="dim2">Third Dimention to transpose</param>
        /// <returns>Tranposed Array</returns>
        static T[][][] ReOrderSub<T>(T[][][] values, int dim0, int dim1, int dim2)
        {
            var d = new[] { dim0, dim1, dim2 };
            unsafe
            {
                // Shape of values dims
                var i = values.Length;
                var j = values[0].Length;
                var k = values[0][0].Length;
                // ReOrder Dims
                var dm2 = new int*[] { &i, &j, &k };
                var dim = new int*[3];
                for (int ii = 0; ii < d.Length; ii++)
                    dim[ii] = dm2[d[ii]];
                // Indexs of Output Array
                var x = dim[0];
                var y = dim[1];
                var z = dim[2];
                // Initialize Array
                var outs = new bool[*x].Select(no => new bool[*y].Select(n => new T[*z]).ToArray()).ToArray();
                // Loop by values dim0
                for (i = 0; i < values.Length; i++)
                {
                    var vi = values[i];
                    // Loop by values dim1
                    for (j = 0; j < vi.Length; j++)
                    {
                        var vij = vi[j];
                        // Loop by values dim2
                        for (k = 0; k < vij.Length; k++)
                            outs[*x][*y][*z] = vij[k];
                    }
                }
                return outs;
            }
        }

        /// <summary>
        /// If 'a' & 'b' are exactly Equal
        /// </summary>
        /// <typeparam name="T">Type of 'a' & 'b'</typeparam>
        /// <param name="a">First value to check</param>
        /// <param name="b">Second value to check</param>
        /// <returns>True: if Array are exactly Equal</returns>
        public static bool IsEqual<T>(List<List<T[]>> a, List<List<T[]>> b)
        {
            // Validate
            if (a.Count != b.Count) return false;
            if (a[0].Count != b[0].Count) return false;
            if (a[0][0].Length != b[0][0].Length) return false;
            // Loop dim0
            for (int i = 0; i < a.Count; i++)
            {
                var aa = a[i];
                var bb = b[i];
                // Loop dim1
                for (int j = 0; j < aa.Count; j++)
                {
                    var aaa = aa[j];
                    var bbb = bb[j];
                    // Loop dim2
                    for (int k = 0; k < aaa.Length; k++)
                        // Make sure they are equal
                        if (! aaa[k].Equals(bbb[k]))
                            return false;
                }
            }
            return true;
        }
        /// <summary>
        /// If 'a' & 'b' are exactly Equal
        /// </summary>
        /// <typeparam name="T">Type of 'a' & 'b'</typeparam>
        /// <param name="a">First value to check</param>
        /// <param name="b">Second value to check</param>
        /// <returns>True: if Array are exactly Equal</returns>
        public static bool IsEqual<T>(T[][] a, T[][] b)
        {
            // Validate Size
            if (a.Length != b.Length) return false;
            // Loop dim0
            for (int i = 0; i < a.Length; i++)
                // Check if equal
                if (!IsEqual(a[i], b[i]))
                    return false;
            return true;
        }
        /// <summary>
        /// If 'a' & 'b' are exactly Equal
        /// </summary>
        /// <typeparam name="T">Type of 'a' & 'b'</typeparam>
        /// <param name="a">First value to check</param>
        /// <param name="b">Second value to check</param>
        /// <returns>True: if Array are exactly Equal</returns>
        public static bool IsEqual<T>(T[] a, T[] b)
        {
            // Validate Size
            if (a.Length != b.Length) return false;
            // Loop dim0
            for (int i = 0; i < a.Length; i++)
                // Check if equal
                if (!a[i].Equals(b[i]))
                    return false;
            return true;
        }
        /// <summary>
        /// If 'a' & 'b' are exactly Equal
        /// </summary>
        /// <typeparam name="T">Type of 'a' & 'b'</typeparam>
        /// <param name="a">First value to check</param>
        /// <param name="b">Second value to check</param>
        /// <returns>True: if Array are exactly Equal</returns>
        public static bool IsEqual<T>(T[][][] a, T[][][] b)
        {
            // Validate Size
            if (a.Length != b.Length) return false;
            // Loop dim0
            for (int i = 0; i < a.Length; i++)
                // Check if equal
                if (!IsEqual(a[i], b[i]))
                    return false;
            return true;
        }
        /// <summary>
        /// If 'a' & 'b' are exactly Equal
        /// </summary>
        /// <typeparam name="T">Type of 'a' & 'b'</typeparam>
        /// <param name="a">First value to check</param>
        /// <param name="b">Second value to check</param>
        /// <returns>True: if Array are exactly Equal</returns>
        public static bool IsEqual<T>(T[][][][] a, T[][][][] b)
        {
            // Validate Size
            if (a.Length != b.Length) return false;
            // Loop dim0
            for (int i = 0; i < a.Length; i++)
                // Check if equal
                if (!IsEqual(a[i], b[i]))
                    return false;
            return true;
        }

        /// <summary>
        /// Reshape List Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <param name="dim2">Third Dimention output</param>
        /// <returns>New Reshaped List according to dim values</returns>
        public static List<List<T[]>> Reshape<T>(List<T[]> series, int dim0, int dim1, int dim2)
        {
            // Total Length
            var total = series.Count * series[0].Length;
            var dim = new[] { dim0, dim1, dim2 };
            // Validation
            if (dim.Any(d => d <= 0 && d != -1)) throw new Exception("Invalid dimenstions.");
            // Product of values
            Func<int[], int, int> Product = (ar, st) =>
            {
                var prod = ar[st];
                for (int i = st + 1; i < ar.Length; i++)
                    prod *= ar[i];
                return prod;
            };
            // If it needs to find the dimention
            if (dim.Any(v => v == -1))
            {
                var total2 = total;
                int ix;
                for (ix = 0; ix < dim.Length; ix++)
                {
                    var s = dim[ix];
                    if (s == -1) break;
                    if (total2 % s != 0) throw new Exception($"dim{ix}: Not divisible");
                    total2 /= s;
                }
                if (ix < dim.Length - 1)
                {
                    var prod = Product(dim, ix + 1);
                    if (total2 % prod != 0) throw new Exception("Could Not find dim. There is No dimention for these sizes.");
                    dim[ix] = total2 / prod;
                }
                else dim[ix] = total2;
            }
            if (total != Product(dim, 0))
                throw new Exception("Cannot be reshaped with those dimentions.");
            dim0 = dim[0];
            dim1 = dim[1];
            dim2 = dim[2];
            var output = new List<T[]>[dim0].ToList();
            var flat = series.SelectMany(v => v).ToList();
            var ii = 0;
            for (int i = 0; i < dim0; i++)
            {
                var out2 = output[i] = new T[dim1][].ToList();
                for (int j = 0; j < dim1; j++)
                {
                    var o3 = out2[j] = new T[dim2];
                    for (int k = 0; k < dim2; k++)
                        o3[k] = flat[ii++];
                }
            }
            if (ii != flat.Count) throw new Exception("ii != flat.Count");
            return output;
        }
        /// <summary>
        /// Reshape List Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <returns>New List Array according to dim values</returns>
        public static List<List<T>> Reshape<T>(List<T> series, int dim0, int dim1)
        {
            // Validation
            if (series.Count % dim0 != 0) throw new Exception($"series.Count:{series.Count} % dim0:{dim0} != 0");
            if (series.Count / dim0 != dim1) throw new Exception($"series.Count:{series.Count} / dim0:{dim0} != dim1:{dim1}");
            var ix = 0;
            // Initialize List
            var ot = new int[dim0].Select(no => new T[dim1].ToList()).ToList();
            // Loop dim0
            for (int i = 0; i < dim0; i++)
            {
                var o = ot[i];
                // Loop dim1
                for (int j = 0; j < dim1; j++)
                    o[j] = series[ix++];
            }
            // Validation
            if (ix != series.Count) throw new Exception("ix != series.Count");
            return ot;
        }
        /// <summary>
        /// Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        public static T[][] Reshape<T>(T[] series, int dim0, int dim1)
        {
            // Validation
            if (dim0 == -1 && dim1 == -1) throw new Exception("Both dims cannot have a -1 value.");
            // if it needs to find dim0
            if (dim0 == -1)
            {
                // Validation
                if (series.Length % dim1 != 0)
                    throw new Exception("Cannot reshape with these dimensions");
                // Get dim size
                dim0 = series.Length / dim1;
            }
            // if it needs to find dim1
            else if (dim1 == -1)
            {
                // Validation
                if (series.Length % dim0 != 0)
                    throw new Exception("Cannot reshape with these dimensions");
                // Get dim size
                dim1 = series.Length / dim0;
            }
            // Validation
            else if (series.Length % dim0 != 0) 
                throw new Exception($"series.Count:({series.Length}) % dim0:({dim0}) != 0");
            else if (series.Length / dim0 != dim1) 
                throw new Exception($"series.Count:({series.Length}) / dim0:({dim0}) != dim1:({dim1})");
            // Reshape
            return ReshapeSub(series, dim0, dim1);   
        }
        /// <summary>
        /// Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <param name="dim2">Third Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        public static T[][][] Reshape<T>(T[] series, int dim0, int dim1, int dim2)
        {
            var d = new[] { dim0, dim1, dim2 };
            var len = series.Length;
            // Make sure a max of 1 value is negative
            var negcnt = 0;
            foreach (var v in d)
                if (v <= 0) negcnt++;
            if (negcnt > 1) throw new Exception("Only 1 dim can have a -1 value.");
            // Needs to find dimention
            if (negcnt == 1)
            {
                // if it needs to find dim0
                if (dim0 == -1)
                {
                    var p = (long)dim1 * dim2;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim0 = (int)(len / p);
                }
                // if it needs to find dim1
                else if (dim1 == -1)
                {
                    var p = (long)dim0 * dim2;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim1 = (int)(len / p);
                }
                // if it needs to find dim2
                else if (dim2 == -1)
                {
                    var p = (long)dim0 * dim1;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim2 = (int)(len / p);
                }
                else throw new Exception("The only negative value can be -1.");
            }
            // Check if dimentions are dims for valid reshape
            else
            {
                // Validation
                var p = (long)dim0 * dim1 * dim2;
                if (len != p)
                    throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
            }
            // Initialize Array
            var outs = new T[dim0][][];
            var ix = 0;
            // Loop output dim0
            for (int i = 0; i < dim0; i++)
            {
                var oi = outs[i] = new T[dim1][];
                // Loop output dim1
                for (int j = 0; j < dim1; j++)
                {
                    var o = oi[j] = new T[dim2];
                    // Loop output dim2
                    for (int k = 0; k < dim2; k++)
                        o[k] = series[ix++];
                }
            }
            // Validation
            if (ix != series.Length) throw new Exception("ix != series.Length");
            return outs;
        }
        /// <summary>
        /// Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim">Dimention of output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        public static T[] Reshape<T>(T[][] series, int dim)
        {
            // Length of all dimensions
            var len = (long)series.Length * series[0].Length;
            // Find dimention
            if (dim == -1)
                dim = (int)len;
            // Validation
            else if (dim != len)
                throw new Exception($"Cannot reshape input from ({series.Length}, {series[0].Length}) to --> ({dim}).");
            // Initialize Array
            var y = new T[dim];
            var ix = 0;
            // Series dim0
            for (int i = 0; i < series.Length; i++)
            {
                var s = series[i];
                // Series dim1
                for (int j = 0; j < s.Length; j++)
                    y[ix++] = s[j];
            }
            return y;
        }
        /// <summary>
        /// Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <param name="dim2">Third Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        public static T[][][] Reshape<T>(T[][] series, int dim0, int dim1, int dim2)
        {
            var d = new[] { dim0, dim1, dim2 };
            // Length of all dimensions
            var len = (long)series.Length * series[0].Length;
            // Make sure a max of 1 value is negative
            var negcnt = 0;
            foreach (var v in d)
                if (v <= 0) negcnt++;
            if (negcnt > 1) throw new Exception("Only 1 dim can have a -1 value.");
            // Needs to find dimention
            if (negcnt == 1)
            {
                // if it needs to find dim0
                if (dim0 == -1)
                {
                    var p = (long)dim1 * dim2;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim0 = (int)(len / p);
                }
                // if it needs to find dim1
                else if (dim1 == -1)
                {
                    var p = (long)dim0 * dim2;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim1 = (int)(len / p);
                }
                // if it needs to find dim2
                else if (dim2 == -1)
                {
                    var p = (long)dim0 * dim1;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim2 = (int)(len / p);
                }
                else throw new Exception("The only negative value can be -1.");
            }
            // Check if dimentions are dims for valid reshape
            else
            {
                var p = (long)dim0 * dim1 * dim2;
                // Validation
                if (len != p)
                    throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
            }
            // Faster: less indexing
            if (dim0 == series.Length)
                return series.Select(s => ReshapeSub(s, dim1, dim2)).ToArray();
            // Reshape
            return ReshapeSub(series, dim0, dim1, dim2);
        }
        /// <summary>
        /// Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <param name="dim2">Third Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        public static T[][][] Reshape<T>(T[][][][] series, int dim0, int dim1, int dim2)
        {
            // Find dim
            if (dim0 == -1)
                dim0 = series.Length;
            var d = new[] { dim0, dim1, dim2 };
            var ser0 = series[0];
            // Lengths of all dims together
            var len = (long)ser0.Length * ser0[0].Length * ser0[0][0].Length;
            // Validation: Make sure a max of 1 value is negative
            var negcnt = 0;
            foreach (var v in d)
                if (v <= 0) negcnt++;
            if (negcnt > 1) throw new Exception("Only 1 dim can have a -1 value.");
            if (dim0 != series.Length)
                throw new NotSupportedException("series.Length != dim0");
            // Needs to find dimention
            if (negcnt == 1)
            {
                // if it needs to find dim0
                //if (dim0 == -1)
                //{
                //    var p = (long)dim1 * dim2;
                //// Validation
                //    if (len % p != 0)
                //        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                //// Get dim size
                //    dim0 = (int)(len / p);
                //}
                // if it needs to find dim1
                if (dim1 == -1)
                {
                    var p = dim2;
                    //var p = (long)dim0 * dim2;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim1 = (int)(len / p);
                }
                // if it needs to find dim2
                else if (dim2 == -1)
                {
                    var p = dim1;
                    //var p = (long)dim0 * dim1;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim2 = (int)(len / p);
                }
                else throw new Exception("The only negative value can be -1.");
            }
            // Check if dimentions are dims for valid reshape
            else
            {
                var p = (long)dim1 * dim2;
                //var p = (long)dim0 * dim1 * dim2;
                // Validation
                if (len != p)
                    throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
            }
            // Reshape
            return series.Select(s => ReshapeSub(s, dim1, dim2)).ToArray();
        }
        /// <summary>
        /// Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        public static T[][] Reshape<T>(T[][][] series, int dim0, int dim1)
        {
            // Lengths of all dims together
            var len = (long)series.Length * series[0].Length * series[0][0].Length;
            var dims = new[] { dim0, dim1 };
            // Validation: Make sure a max of 1 value is negative -1
            var negcnt = 0;
            foreach (var v in dims)
                if (v <= 0) negcnt++;
            if (negcnt > 1)
                throw new Exception($"Only 1 dim can have a value of -1: got ({dim0}, {dim1}).");
            // Needs to find dimention
            if (negcnt == 1)
            {
                // if it needs to find dim0
                if (dim0 == -1)
                {
                    // Validation
                    if (len % dim1 != 0)
                        throw new Exception($"Cannot reshape from ({series.Length}, {series[0].Length}, {series[0][0].Length}) to --> ({dim0}, {dim1}).");
                    // Get dim size
                    dim0 = (int)(len / dim1);
                }
                // if it needs to find dim1
                else if (dim1 == -1)
                {
                    // Validation
                    if (len % dim0 != 0)
                        throw new Exception($"Cannot reshape from ({series.Length}, {series[0].Length}, {series[0][0].Length}) to --> ({dim0}, {dim1}).");
                    // Get dim size
                    dim1 = (int)(len / dim0);
                }
                else
                    throw new Exception($"Invalid value for at least one dim: got ({dim0}, {dim1}).");
            }
            // Check if dimentions are dims for valid reshape
            else if (len != (long)dim0 * dim1)
                throw new Exception($"Cannot reshape from ({series.Length}, {series[0].Length}, {series[0][0].Length}) to --> ({dim0}, {dim1}).");
            // Initialize Array
            var outs = new T[dim0][];
            var x = 0;      // outs dim0
            var y = -1;     // outs dim1
            var sub = outs[0] = new T[dim1];
            // Loop series dim0
            for (int i = 0; i < series.Length; i++)
            {
                var si = series[i];
                // Loop series dim1
                for (int j = 0; j < si.Length; j++)
                {
                    var s = si[j];
                    // Loop series dim2
                    for (int k = 0; k < s.Length; k++)
                    {
                        // If over size: move to next array
                        if (++y == sub.Length)
                        {
                            y = 0;
                            sub = outs[++x] = new T[dim1];
                        }
                        sub[y] = s[k];
                    }
                }
            }
            // Validation
            if (x != outs.Length - 1 || y != sub.Length - 1)
                throw new Exception($"x({x}) != {outs.Length - 1} || y({y}) != {sub.Length - 1}");
            return outs;
        }
        /// <summary>
        /// Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series_">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <param name="dim2">Third Dimention output</param>
        /// <param name="dim3">Forth Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        public static T[][][][] Reshape<T>(T[][][] series_, int dim0, int dim1, int dim2, int dim3)
        {
            var series = series_[0];
            // Find dim
            if (dim0 == -1)
                dim0 = series_.Length;
            var d = new[] { dim0, dim1, dim2, dim3 };
            // Lengths of all dims together
            var len = (long)series.Length * series[0].Length;
            // Validation: Make sure a max of 1 value is negative -1
            var negcnt = 0;
            foreach (var v in d)
                if (v <= 0) negcnt++;
            if (dim0 != series_.Length)
                throw new NotSupportedException("dim0 != series.Length");
            if (negcnt > 1) throw new Exception("Only 1 dim can have a -1 value.");
            // Needs to find dimention
            if (negcnt == 1)
            {
                // if it needs to find dim1
                if (dim1 == -1)
                {
                    var p = (long)dim2 * dim3;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim1 = (int)(len / p);
                }
                // if it needs to find dim2
                else if (dim2 == -1)
                {
                    var p = (long)dim1 * dim3;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim2 = (int)(len / p);
                }
                // if it needs to find dim3
                else if (dim3 == -1)
                {
                    var p = (long)dim1 * dim2;
                    // Validation
                    if (len % p != 0)
                        throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
                    // Get dim size
                    dim3 = (int)(len / p);
                }
                else throw new Exception("The only negative value can be -1.");
            }
            // Check if dimentions are dims for valid reshape
            else
            {
                var p = (long)dim1 * dim2 * dim3;
                // Validation
                if (len != p)
                    throw new Exception($"Cannot reshape with these dimensions ({string.Join(", ", d)}).");
            }
            // Faster less indexing
            if (dim1 == series.Length)
                return series_.Select(si => si.Select(s => ReshapeSub(s, dim2, dim3)).ToArray()).ToArray();
            // Reshape
            return series_.Select(s => ReshapeSub(s, dim1, dim2, dim3)).ToArray();
        }
        /// <summary>
        /// Sub Function: Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        static T[][] ReshapeSub<T>(T[] series, int dim0, int dim1)
        {
            var ix = 0;
            // Initialize Array
            var ot = new T[dim0][];
            // Loop output dim0
            for (int i = 0; i < dim0; i++)
            {
                var o = ot[i] = new T[dim1];
                // Loop output dim1
                for(int j = 0; j < dim1; j++)
                    o[j] = series[ix++];
            }
            // Validation
            if (ix != series.Length) throw new Exception("ix != series.Count");
            return ot;
        }
        /// <summary>
        /// Sub Function: Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        static T[][] ReshapeSub<T>(T[][][] series, int dim0, int dim1)
        {
            // Initialize Array
            var outs = new T[dim0][];
            // Series dim0, dim1, dim2 Respectivly
            int x = 0, y = 0, z = -1;  
            var sub = series[0];
            var s2 = sub[0];
            // Loop output dim0
            for (int i = 0; i < dim0; i++)
            {
                var o = outs[i] = new T[dim1];
                // Loop output dim1
                for (int j = 0; j < dim1; j++)
                {
                    // If size is bigger move to next
                    if (++z == s2.Length)
                    {
                        z = 0;
                        // If size is bigger move to next
                        if (++y == sub.Length)
                        {
                            y = 0;
                            sub = series[++x];
                        }
                        s2 = sub[y];
                    }
                    o[j] = s2[z];
                }
            }
            // Validation
            if (++x != series.Length || ++y != sub.Length || ++z != s2.Length)
                throw new Exception($"x({x}) != series.Length({series.Length}) || y({y}) != sub.Length({sub.Length}) || z({z}) != s2.Length({s2.Length})");
            return outs;
        }
        /// <summary>
        /// Sub Function: Reshape Array Dimensions to new Dimentions
        /// </summary>
        /// <typeparam name="T">Type of series</typeparam>
        /// <param name="series">Values to Reshape</param>
        /// <param name="dim0">First Dimention output</param>
        /// <param name="dim1">Second Dimention output</param>
        /// <param name="dim2">Third Dimention output</param>
        /// <returns>New Reshaped Array according to dim values</returns>
        static T[][][] ReshapeSub<T>(T[][] series, int dim0, int dim1, int dim2)
        {
            // Initialize Array
            var ot = new T[dim0][][];
            var x = 0;      // series dim0 index
            var y = -1;     // series dim1 index
            var sub = series[0];
            // Loop output dim0
            for (int i = 0; i < dim0; i++)
            {
                var oi = ot[i] = new T[dim1][];
                // Loop output dim1
                for (int j = 0; j < dim1; j++)
                {
                    var o = oi[j] = new T[dim2];
                    // Loop output dim2
                    for (int k = 0; k < dim2; k++)
                    {
                        // If size is bigger move to next
                        if (++y == sub.Length)
                        {
                            y = 0;
                            sub = series[++x];
                        }
                        o[k] = sub[y];
                    }
                }
            }
            // Validation
            if (x != series.Length - 1 || y != sub.Length - 1)
                throw new Exception($"x({x}) != series.Length - 1({series.Length - 1}) || y({y}) != sub.Length - 1({sub.Length - 1})");
            return ot;
        }

        /// <summary>
        /// Matrix Multiplication
        /// </summary>
        /// <param name="a">Array 'a'</param>
        /// <param name="b">Array 'a'</param>
        /// <returns>Output of Matrix Multiplication</returns>
        public static float[][] Matmul(float[][] a, float[][] b)
        {
            // Validation
            if (a[0].Length != b.Length) throw new Exception("a[0].Length != b.Length");
            // Matrix Multiplication
            return MatmulSub(a, b);
        }
        /// <summary>
        /// Matrix Multiplication
        /// </summary>
        /// <param name="a">Array 'a'</param>
        /// <param name="b">Array 'a'</param>
        /// <returns>Output of Matrix Multiplication</returns>
        public static float[][][] Matmul(float[][][] a, float[][][] b)
        {
            // Validation
            if (a.Length != b.Length) throw new Exception("a.Length != b.Length");
            if (a[0][0].Length != b[0].Length) throw new Exception("a[0][0].Length != b[0].Length");
            // Matrix Multiplication
            return a.Zip(b, MatmulSub).ToArray();
        }
        /// <summary>
        /// Matrix Multiplication
        /// </summary>
        /// <param name="a">Array 'a'</param>
        /// <param name="b">Array 'a'</param>
        /// <returns>Output of Matrix Multiplication</returns>
        public static float[][][][] Matmul(float[][][][] a, float[][][][] b)
        {
            // Validation
            if (a.Length != b.Length) throw new Exception("a.Length != b.Length");
            if (a[0].Length != b[0].Length) throw new Exception("a[0].Length != b[0].Length");
            if (a[0][0][0].Length != b[0][0].Length) throw new Exception("a[0][0][0].Length != b[0][0].Length");
            // Matrix Multiplication
            return a.Zip(b, MatmulSub).ToArray();
        }

        /// <summary>
        /// Sub Function (No Validation): Matrix Multiplication
        /// </summary>
        /// <param name="a">Array 'a'</param>
        /// <param name="b">Array 'a'</param>
        /// <returns>Output of Matrix Multiplication</returns>
        static float[][] MatmulSub(float[][] a, float[][] b)
        {
            var n = a[0].Length;
            var p = b[0].Length;
            var c = new float[a.Length][];
            // Loop a dim0
            for (int i = 0; i < a.Length; i++)
            {
                var ai = a[i];
                var ci = c[i] = new float[p];
                // Loop a dim1
                for (int k = 0; k < n; k++)
                {
                    var aik = ai[k];
                    var bk = b[k];
                    // Loop b dim1
                    for (int j = 0; j < p; j++)
                        ci[j] += aik * bk[j];
                }
            }
            return c;
        }
        /// <summary>
        /// Sub Function (No Validation): Matrix Multiplication
        /// </summary>
        /// <param name="a">Array 'a'</param>
        /// <param name="b">Array 'a'</param>
        /// <returns>Output of Matrix Multiplication</returns>
        static float[][][] MatmulSub(float[][][] a, float[][][] b)
            => a.Zip(b, MatmulSub).ToArray();

        /// <summary>
        /// The Product of double values
        /// </summary>
        /// <param name="values">Values to get the Product from</param>
        /// <returns>The Product of values</returns>
        public static double Product(IEnumerable<double> values)
        {
            var prod = 1d;
            foreach (var v in values)
                prod *= v;
            return prod;
        }

        static Random _rand = new Random();

        /// <summary>
        /// Shuffle Items in List
        /// </summary>
        /// <typeparam name="T">Type of List</typeparam>
        /// <param name="list">Values to Shuffle</param>
        /// <returns>Mutated list</returns>
        public static IList<T> Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            // Look backwards through list
            while (n > 1)
            {
                n--;
                int k = _rand.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
            return list;
        }

        /// <summary>
        /// Normalize
        /// </summary>
        /// <param name="values">Values to Normalize</param>
        /// <returns>New Array of Normalized Values</returns>
        public static List<float> Norm(float[] values)
        {
            var min = values.Min();
            var max = values.Max();
            if (max == min) throw new Exception("max == min: All the values are the same");
            var div = max - min;
            // Normalize
            return values.Select(v => Math.Min(Math.Max((v - min) / div, 0), 1)).ToList();
        }

        /// <summary>
        /// Modulus for floating point numbers. 
        /// Group Floating point values into an int group
        /// </summary>
        /// <param name="values">Values to use</param>
        /// <param name="div">Divisor that defines the group</param>
        /// <param name="maxval">Max value to return</param>
        /// <returns>Integer values of the groups</returns>
        public static int[] Modulo(List<float> values, float div, int maxval)
            => values.Select(v =>
            {
                // Floor
                var n = (int)(v / div);    
                // + Round
                n += (int)Math.Round((v - (n * div)) / div);
                // Bound to 0 -> maxval
                return Math.Min(Math.Max(n, 0), maxval);
            }).ToArray();

        /// <summary>
        /// Create an Array of Zeros
        /// </summary>
        /// <param name="dim">Dimention of Output Array</param>
        /// <returns>Array of Zeros as a value</returns>
        public static float[] Zeros(int dim)
            => new float[dim];
        /// <summary>
        /// Create an Array of Zeros
        /// </summary>
        /// <param name="dim0">Dimention (0) of Output Array</param>
        /// <param name="dim1">Dimention (1) of Output Array</param>
        /// <returns>Array of Zeros as a value</returns>
        public static float[][] Zeros(int dim0, int dim1)
        {
            var outs = new float[dim0][];
            for (int i = 0; i < dim0; i++)
                outs[i] = new float[dim1];
            return outs;
        }
        /// <summary>
        /// Create an Array of Zeros
        /// </summary>
        /// <param name="dim0">Dimention (0) of Output Array</param>
        /// <param name="dim1">Dimention (1) of Output Array</param>
        /// <param name="dim2">Dimention (2) of Output Array</param>
        /// <returns>Array of Zeros as a value</returns>
        public static float[][][] Zeros(int dim0, int dim1, int dim2)
        {
            var outs = new float[dim0][][];
            for (int i = 0; i < dim0; i++)
                outs[i] = Zeros(dim1, dim2);
            return outs;
        }

        /// <summary>
        /// Create an Array of Ones
        /// </summary>
        /// <param name="dim">Dimention of Output Array</param>
        /// <returns>Array of ones as a value</returns>
        public static float[] Ones(int dim)
        {
            var outs = new float[dim];
            for (int i = 0; i < dim; i++)
                outs[i] = 1;
            return outs;
        }
        /// <summary>
        /// Create an Array of Ones
        /// </summary>
        /// <param name="dim0">Dimention (0) of Output Array</param>
        /// <param name="dim1">Dimention (1) of Output Array</param>
        /// <returns>Array of ones as a value</returns>
        public static float[][] Ones(int dim0, int dim1)
        {
            var outs = new float[dim0][];
            for (int i = 0; i < dim0; i++)
                outs[i] = Ones(dim1);
            return outs;
        }
        /// <summary>
        /// Create an Array of Ones
        /// </summary>
        /// <param name="dim0">Dimention (0) of Output Array</param>
        /// <param name="dim1">Dimention (1) of Output Array</param>
        /// <param name="dim2">Dimention (2) of Output Array</param>
        /// <returns>Array of ones as a value</returns>
        public static float[][][] Ones(int dim0, int dim1, int dim2)
        {
            var outs = new float[dim0][][];
            for (int i = 0; i < dim0; i++)
                outs[i] = Ones(dim1, dim2);
            return outs;
        }

    /// <summary>
    /// Create an Array that all elements are filled with 'value'
    /// </summary>
    /// <param name="value">Value to fill all elements in Array</param>
    /// <param name="dim">Dimention of Output Array</param>
    /// <returns>Array that all elements are filled with 'value'</returns>
    public static T[] Fill<T>(int dim, T value)
    {
        var outs = new T[dim];
        for (int i = 0; i < dim; i++)
            outs[i] = value;
        return outs;
    }
    /// <summary>
    /// Create an Array that all elements are filled with 'value'
    /// </summary>
    /// <param name="value">Value to fill all elements in Array</param>
    /// <param name="dim0">Dimention (0) of Output Array</param>
    /// <param name="dim1">Dimention (1) of Output Array</param>
    /// <returns>Array that all elements are filled with 'value'</returns>
    public static T[][] Fill<T>(int dim0, int dim1, T value)
    {
        var outs = new T[dim0][];
        for (int i = 0; i < dim0; i++)
            outs[i] = Fill(dim1, value);
        return outs;
    }
    /// <summary>
    /// Create an Array that all elements are filled with 'value'
    /// </summary>
    /// <param name="value">Value to fill all elements in Array</param>
    /// <param name="dim0">Dimention (0) of Output Array</param>
    /// <param name="dim1">Dimention (1) of Output Array</param>
    /// <param name="dim2">Dimention (2) of Output Array</param>
    /// <returns>Array that all elements are filled with 'value'</returns>
    public static T[][][] Fill<T>(int dim0, int dim1, int dim2, T value)
    {
        var outs = new T[dim0][][];
        for (int i = 0; i < dim0; i++)
            outs[i] = Fill(dim1, dim2, value);
        return outs;
    }
    /// <summary>
    /// Create an Array that all elements are filled with 'value'
    /// </summary>
    /// <param name="value">Value to fill all elements in Array</param>
    /// <param name="dim0">Dimention (0) of Output Array</param>
    /// <param name="dim1">Dimention (1) of Output Array</param>
    /// <param name="dim2">Dimention (2) of Output Array</param>
    /// <param name="dim3">Dimention (3) of Output Array</param>
    /// <returns>Array that all elements are filled with 'value'</returns>
    public static T[][][][] Fill<T>(int dim0, int dim1, int dim2, int dim3, T value)
    {
        var outs = new T[dim0][][][];
        for (int i = 0; i < dim0; i++)
            outs[i] = Fill(dim1, dim2, dim3, value);
        return outs;
    }

    //public static void PlaySound(string path, double volume = 0.50)
    //{
    //    var player = new MediaPlayer();
    //    player.Volume = volume;
    //    player.Open(new Uri(path, UriKind.Relative));
    //    player.Play();
    //}
}
