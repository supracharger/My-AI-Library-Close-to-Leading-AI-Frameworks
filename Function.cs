using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Functions with Defined Backwards Passes
    /// </summary>
    public static class Function
    {
        /// <summary>
        /// Transpose first 2 dimenstions. Can be float[2d-3d]
        /// </summary>
        public static class T
        {
            /// <summary>
            /// Transpose first 2 dimenstions. float[2d]
            /// </summary>
            /// <param name="x">input</param>
            /// <returns>Transposed Dimentions</returns>
            public static Tensor<float[][]> Fwd(Tensor<float[][]> x)
            {
                var y = Fwd(x._data, out object ctx);
                return x.AddNode(y, ctx, Bck, false);
            }
            /// <summary>
            /// Transpose first 2 dimenstions. float[3d]
            /// </summary>
            /// <param name="x">input</param>
            /// <returns>Transposed Dimentions</returns>
            public static Tensor<float[][][]> Fwd(Tensor<float[][][]> x)
            {
                var y = Fwd(x._data, out object ctx);
                return x.AddNode(y, ctx, Bck_3dim, false);
            }
            /// <summary>
            /// Transpose first 2 dimenstions.
            /// </summary>
            /// <param name="values">input</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Transposed Dimentions</returns>
            public static S[][] Fwd<S>(S[][] values, out object ctx)
            {
                var y = Utilz.TransposeArray(values);
                ctx = new[] { y.Length, y[0].Length };
                return y;
            }

            /// <summary>
            /// Inlined Backwards Function 2d
            /// </summary>
            /// <param name="y">Output Node of forwards pass</param>
            static void Bck(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined Backwards Function 3d
            /// </summary>
            /// <param name="y">Output Node of forwards pass</param>
            static void Bck_3dim(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Backwards Function
            /// </summary>
            /// <typeparam name="S">Type of values</typeparam>
            /// <param name="values">Output gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Transposed Gradient</returns>
            public static S[][] Bck<S>(S[][] values, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 2)
                    throw new Exception($"ctx.Length({ctx.Length}) != 2");
                // Validate Shape
                if (values.Length != ctx[0] || values[0].Length != ctx[1])
                    throw new Exception($"Invalid gradient shape: got ({values.Length}, {values[0].Length}) " +
                                $"it should be ({ctx[0]}, {ctx[1]}).");
                // Transpose
                return Utilz.TransposeArray(values);
            }

        }

        /// <summary>
        /// Reshape Dimensions to another set of Dimentions
        /// </summary>
        public static class Reshape
        {
            /// <summary>
            /// float[1d] -> float[2d]
            /// </summary>
            /// <param name="x">values to Reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <returns>float[2d]</returns>
            public static Tensor<float[][]> Fwd(Tensor<float[]> x, int dim0, int dim1)
            {
                var y = Fwd(x._data, dim0, dim1, out object ctx);
                return x.AddNode(y, ctx, Bck1d_2d, false);
            }
            /// <summary>
            /// float[2d] -> float[1d]
            /// </summary>
            /// <param name="x">values to reshape</param>
            /// <param name="dim0">Dimention</param>
            /// <returns>float[1d]</returns>
            public static Tensor<float[]> Fwd(Tensor<float[][]> x, int dim0)
            {
                var y = Fwd(x._data, dim0, out object ctx);
                return x.AddNode(y, ctx, Bck2d_1d, false);
            }
            /// <summary>
            /// float[2d] -> float[3d]
            /// </summary>
            /// <param name="x">values to reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <param name="dim2">Third dimention</param>
            /// <returns>float[3d]</returns>
            public static Tensor<float[][][]> Fwd(Tensor<float[][]> x, int dim0, int dim1, int dim2)
            {
                var y = Fwd(x._data, dim0, dim1, dim2, out object ctx);
                return x.AddNode(y, ctx, Bck2d_3d, false);
            }
            /// <summary>
            /// float[3d] -> float[2d]
            /// </summary>
            /// <param name="x">values to reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <returns>float[2d]</returns>
            public static Tensor<float[][]> Fwd(Tensor<float[][][]> x, int dim0, int dim1)
            {
                var y = Fwd(x._data, dim0, dim1, out object ctx);
                return x.AddNode(y, ctx, Bck3d_2d, false);
            }
            /// <summary>
            /// float[3d] -> float[4d]
            /// </summary>
            /// <param name="x">values to reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <param name="dim2">Third dimention</param>
            /// <param name="dim3">Forth dimention</param>
            /// <returns>float[4d]</returns>
            public static Tensor<float[][][][]> Fwd(Tensor<float[][][]> x, int dim0, int dim1, int dim2, int dim3)
            {
                var y = Fwd(x._data, dim0, dim1, dim2, dim3, out object ctx);
                return x.AddNode(y, ctx, Bck3d_4d, false);
            }
            /// <summary>
            /// float[4d] -> float[3d]
            /// </summary>
            /// <param name="x">values to reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <param name="dim2">Third dimention</param>
            /// <returns>float[3d]</returns>
            public static Tensor<float[][][]> Fwd(Tensor<float[][][][]> x, int dim0, int dim1, int dim2)
            {
                var y = Fwd(x._data, dim0, dim1, dim2, out object ctx);
                return x.AddNode(y, ctx, Bck4d_3d, false);
            }

            /// <summary>
            /// float[1d] -> float[2d]
            /// </summary>
            /// <typeparam name="T">Type of series</typeparam>
            /// <param name="series">values to reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>float[2d]</returns>
            public static T[][] Fwd<T>(T[] series, int dim0, int dim1, out object ctx)
            {
                var y = Utilz.Reshape(series, dim0, dim1);
                ctx = new[] { series.Length, y.Length, y[0].Length };
                return y;
            }
            /// <summary>
            /// float[2d] -> float[1d]
            /// </summary>
            /// <typeparam name="T">Type of series</typeparam>
            /// <param name="series">values to reshape</param>
            /// <param name="dim">Dimention</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>float[1d]</returns>
            public static T[] Fwd<T>(T[][] series, int dim, out object ctx)
            {
                var y = Utilz.Reshape(series, dim);
                ctx = new[] { series.Length, series[0].Length, y.Length };
                return y;
            }
            /// <summary>
            /// float[2d] -> float[3d]
            /// </summary>
            /// <typeparam name="T">Type of series</typeparam>
            /// <param name="series">series to reshape</param>
            /// <param name="dim0">First Dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <param name="dim2">Third dimention</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>float[3d]</returns>
            public static T[][][] Fwd<T>(T[][] series, int dim0, int dim1, int dim2, out object ctx)
            {
                var y = Utilz.Reshape(series, dim0, dim1, dim2);
                ctx = new[] { series.Length, series[0].Length, y.Length, y[0].Length, y[0][0].Length };
                return y;
            }
            /// <summary>
            /// float[3d] -> float[2d]
            /// </summary>
            /// <typeparam name="T">Type of series</typeparam>
            /// <param name="series">Series to reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <param name="ctx">Third dimention</param>
            /// <returns>float[2d]</returns>
            public static T[][] Fwd<T>(T[][][] series, int dim0, int dim1, out object ctx)
            {
                var y = Utilz.Reshape(series, dim0, dim1);
                ctx = new[] { series.Length, series[0].Length, series[0][0].Length, 
                                y.Length, y[0].Length };
                return y;
            }
            /// <summary>
            /// float[3d] -> float[4d]
            /// </summary>
            /// <typeparam name="T">Type of Series</typeparam>
            /// <param name="series">Series to reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <param name="dim2">Third dimention</param>
            /// <param name="dim3">Fourth dimention</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>float[4d]</returns>
            public static T[][][][] Fwd<T>(T[][][] series, int dim0, int dim1, int dim2, int dim3, out object ctx)
            {
                var y = Utilz.Reshape(series, dim0, dim1, dim2, dim3);
                ctx = new[] { series.Length, series[0].Length, series[0][0].Length,
                                y.Length, y[0].Length, y[0][0].Length, y[0][0][0].Length };
                return y;
            }
            /// <summary>
            /// float[4d] -> float[3d]
            /// </summary>
            /// <typeparam name="T">Type of series</typeparam>
            /// <param name="series">Series to reshape</param>
            /// <param name="dim0">First dimention</param>
            /// <param name="dim1">Second dimention</param>
            /// <param name="dim2">Third dimention</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>float[3d]</returns>
            public static T[][][] Fwd<T>(T[][][][] series, int dim0, int dim1, int dim2, out object ctx)
            {
                var y = Utilz.Reshape(series, dim0, dim1, dim2);
                ctx = new[] { series.Length, series[0].Length, series[0][0].Length, series[0][0][0].Length, 
                                y.Length, y[0].Length, y[0][0].Length };
                return y;
            }

            /// <summary>
            /// Backwards: Forwards pass float[1d] -> float[2d]
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>float[1d]</returns>
            public static T[] Bck1d_2d<T>(T[][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 3)
                    throw new Exception($"ctx.Length({ctx.Length}) != 3");
                // Validate shape
                if (grad.Length != ctx[1] || grad[0].Length != ctx[2])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}) " +
                            $"it should be ({ctx[1]}, {ctx[2]}).");
                // Reshape
                return Utilz.Reshape(grad, ctx[0]);
            }
            /// <summary>
            /// Backwards: Forwards pass float[2d] -> float[1d]
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>float[2d]</returns>
            public static T[][] Bck2d_1d<T>(T[] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 3)
                    throw new Exception($"ctx.Length({ctx.Length}) != 3");
                // Validate shape
                if (grad.Length != ctx[2])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}) " +
                            $"it should be ({ctx[2]}).");
                // Reshape
                return Utilz.Reshape(grad, ctx[0], ctx[1]);
            }
            /// <summary>
            /// Backwards: Forwards pass float[2d] -> float[3d]
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>float[2d]</returns>
            public static T[][] Bck2d_3d<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 5)
                    throw new Exception($"ctx.Length({ctx.Length}) != 5");
                // Validate shape
                if (grad.Length != ctx[2] || grad[0].Length != ctx[3] || grad[0][0].Length != ctx[4])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                            $"it should be ({ctx[2]}, {ctx[3]}, {ctx[4]}).");
                // Reshape
                return Utilz.Reshape(grad, ctx[0], ctx[1]);
            }
            /// <summary>
            /// Backwards: Forwards pass float[4d] -> float[3d]
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>float[4d]</returns>
            public static T[][][][] Bck4d_3d<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 7)
                    throw new Exception($"ctx.Length({ctx.Length}) != 7");
                // Validate shape
                if (grad.Length != ctx[4] || grad[0].Length != ctx[5] || grad[0][0].Length != ctx[6])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                            $"it should be ({ctx[4]}, {ctx[5]}, {ctx[6]}).");
                // Reshape
                return Utilz.Reshape(grad, ctx[0], ctx[1], ctx[2], ctx[3]);
            }
            /// <summary>
            /// Backwards: Forwards pass float[3d] -> float[2d]
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>float[3d]</returns>
            public static T[][][] Bck3d_2d<T>(T[][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 5)
                    throw new Exception($"ctx.Length({ctx.Length}) != 5");
                // Validate shape
                if (grad.Length != ctx[3] || grad[0].Length != ctx[4])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}) " +
                            $"it should be ({ctx[3]}, {ctx[4]}).");
                // Reshape
                return Utilz.Reshape(grad, ctx[0], ctx[1], ctx[2]);
            }
            /// <summary>
            /// Backwards: Forwards pass float[3d] -> float[4d]
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>float[3d]</returns>
            public static T[][][] Bck3d_4d<T>(T[][][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 7)
                    throw new Exception($"ctx.Length({ctx.Length}) != 7");
                // Validate shape
                if (grad.Length != ctx[3] || grad[0].Length != ctx[4] ||
                    grad[0][0].Length != ctx[5] || grad[0][0][0].Length != ctx[6])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}, {grad[0][0][0].Length}) " +
                            $"it should be ({ctx[3]}, {ctx[4]}, {ctx[5]}, {ctx[6]}).");
                // Reshape
                return Utilz.Reshape(grad, ctx[0], ctx[1], ctx[2]);
            }
            
            /// <summary>
            /// Inlined Backwards pass for Tensor
            /// </summary>
            /// <param name="y">Output of forward pass</param>
            static void Bck1d_2d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck1d_2d((float[][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined Backwards pass for Tensor
            /// </summary>
            /// <param name="y">Output of forward pass</param>
            static void Bck2d_1d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck2d_1d((float[])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined Backwards pass for Tensor
            /// </summary>
            /// <param name="y">Output of forward pass</param>
            static void Bck2d_3d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck2d_3d((float[][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined Backwards pass for Tensor
            /// </summary>
            /// <param name="y">Output of forward pass</param>
            static void Bck4d_3d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck4d_3d((float[][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined Backwards pass for Tensor
            /// </summary>
            /// <param name="y">Output of forward pass</param>
            static void Bck3d_2d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck3d_2d((float[][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined Backwards pass for Tensor
            /// </summary>
            /// <param name="y">Output of forward pass</param>
            static void Bck3d_4d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck3d_4d((float[][][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
        }

        /// <summary>
        /// Takes the Bi-Square Root of the series.
        /// Negative values can be used & will return a negative Square Root value
        /// float[1d - 4d]
        /// </summary>
        public static class Sqrt
        {
            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <returns>The same shape</returns>
            public static Tensor<float[]> Fwd(Tensor<float[]> x)
            {
                var y = Fwd(x._data, out object ctx);
                return x.AddNode(y, ctx, Bck_d1, false);
            }
            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <returns>The same shape</returns>
            public static Tensor<float[][]> Fwd(Tensor<float[][]> x)
            {
                var y = Fwd(x._data, out object ctx);
                return x.AddNode(y, ctx, Bck_d2, false);
            }
            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <returns>The same shape</returns>
            public static Tensor<float[][][]> Fwd(Tensor<float[][][]> x)
            {
                var y = Fwd(x._data, out object ctx);
                return x.AddNode(y, ctx, Bck_d3, false);
            }
            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <returns>The same shape as 'x'</returns>
            public static Tensor<float[][][][]> Fwd(Tensor<float[][][][]> x)
            {
                var y = Fwd(x._data, out object ctx);
                return x.AddNode(y, ctx, Bck_d4, false);
            }
            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>The same shape as 'x'</returns>
            public static float[] Fwd(float[] x, out object ctx)
            {
                ctx = x;
                return FwdSub(x);
            }
            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>The same shape as 'x'</returns>
            public static float[][] Fwd(float[][] x, out object ctx)
            {
                ctx = x;
                return x.Select(FwdSub).ToArray();
            }
            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>The same shape as 'x'</returns>
            public static float[][][] Fwd(float[][][] x, out object ctx)
            {
                ctx = x;
                return x.Select(xi => xi.Select(FwdSub).ToArray()).ToArray();
            }
            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>The same shape as 'x'</returns>
            public static float[][][][] Fwd(float[][][][] x, out object ctx)
            {
                ctx = x;
                return x.Select(xi => xi.Select(xj => xj.Select(FwdSub).ToArray()).ToArray()).ToArray();
            }

            /// <summary>
            /// Bi-Square Root of x
            /// </summary>
            /// <param name="x">values to be Bi-Square Rooted</param>
            /// <returns>The same shape as 'x'</returns>
            static float[] FwdSub(float[] x)
                => x.Select(v => ((float)Math.Sqrt(Math.Abs(v) + 1) - 1) * Math.Sign(v)).ToArray();

            /// <summary>
            /// Inlined backwards pass for tensor
            /// </summary>
            /// <param name="y">output Node from forwards pass</param>
            static void Bck_d1(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined backwards pass for tensor
            /// </summary>
            /// <param name="y">output Node from forwards pass</param>
            static void Bck_d2(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined backwards pass for tensor
            /// </summary>
            /// <param name="y">output Node from forwards pass</param>
            static void Bck_d3(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inlined backwards pass for tensor
            /// </summary>
            /// <param name="y">output Node from forwards pass</param>
            static void Bck_d4(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[][][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }

            /// <summary>
            /// Inlined backwards pass
            /// </summary>
            /// <param name="gradout">Output Gradient</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Input Gradient</returns>
            public static float[] Bck(float[] gradout, object ctx)
            {
                var x = (float[])ctx;
                // Validate Shape
                if (gradout.Length != x.Length)
                    throw new Exception($"gradout.Length({gradout.Length}) != x.Length({x.Length})");
                // Get Derivative of Bi-Sqrt
                return BckSub(gradout, x);
            }
            /// <summary>
            /// Inlined backwards pass
            /// </summary>
            /// <param name="gradout">Output Gradient</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Input Gradient</returns>
            public static float[][] Bck(float[][] gradout, object ctx)
            {
                var x = (float[][])ctx;
                // Validate Shape
                if (gradout.Length != x.Length || gradout[0].Length != x[0].Length)
                    throw new Exception($"gradout.Length({gradout.Length}, {gradout[0].Length}) != x.Length({x.Length}, {x[0].Length})");
                // Get Derivative of Bi-Sqrt
                return gradout.Zip(x, BckSub).ToArray();
            }
            /// <summary>
            /// Inlined backwards pass
            /// </summary>
            /// <param name="gradout">Output Gradient</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Input Gradient</returns>
            public static float[][][] Bck(float[][][] gradout, object ctx)
            {
                var x = (float[][][])ctx;
                // Validate Shape
                if (gradout.Length != x.Length || gradout[0].Length != x[0].Length || gradout[0][0].Length != x[0][0].Length)
                    throw new Exception($"gradout.Length({gradout.Length}, {gradout[0].Length}, {gradout[0][0].Length}) != " +
                            $"x.Length({x.Length}, {x[0].Length}, {x[0][0].Length})");
                // Get Derivative of Bi-Sqrt
                return gradout.Zip(x, BckSub).ToArray();
            }
            /// <summary>
            /// Inlined backwards pass
            /// </summary>
            /// <param name="gradout">Output Gradient</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Input Gradient</returns>
            public static float[][][][] Bck(float[][][][] gradout, object ctx)
            {
                var x = (float[][][][])ctx;
                // Validate Shape
                if (gradout.Length != x.Length || gradout[0].Length != x[0].Length || 
                    gradout[0][0].Length != x[0][0].Length || gradout[0][0][0].Length != x[0][0][0].Length)
                    throw new Exception($"gradout.Length({gradout.Length}, {gradout[0].Length}, {gradout[0][0].Length}, {gradout[0][0][0].Length}) != " +
                            $"x.Length({x.Length}, {x[0].Length}, {x[0][0].Length}, {x[0][0][0].Length})");
                // Get Derivative of Bi-Sqrt
                return gradout.Zip(x, BckSub).ToArray();
            }

            /// <summary>
            /// Get Derivative of Bi-Sqrt
            /// </summary>
            /// <param name="g">Output Gradient</param>
            /// <param name="x">Input to forwards pass</param>
            /// <returns>Input Gradient</returns>
            static float[] BckSub(float[] g, float[] x)
                => g.Zip(x, (gi, xi) => gi / (2 * (float)Math.Sqrt(Math.Abs(xi) + 1))).ToArray();
            /// <summary>
            /// Get Derivative of Bi-Sqrt
            /// </summary>
            /// <param name="g">Output Gradient</param>
            /// <param name="x">Input to forwards pass</param>
            /// <returns>Input Gradient</returns>
            static float[][] BckSub(float[][] g, float[][] x)
                => g.Zip(x, BckSub).ToArray();
            /// <summary>
            /// Get Derivative of Bi-Sqrt
            /// </summary>
            /// <param name="g">Output Gradient</param>
            /// <param name="x">Input to forwards pass</param>
            /// <returns>Input Gradient</returns>
            static float[][][] BckSub(float[][][] g, float[][][] x)
                => g.Zip(x, BckSub).ToArray();
        }

        /// <summary>
        /// Matrix Multiplication float[3d - 4d]
        /// </summary>
        public static class Matmul
        {
            /// <summary>
            /// Matrix Multiplication float[3d]
            /// </summary>
            /// <param name="a">values a</param>
            /// <param name="b">values b</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>float[3d]</returns>
            public static float[][][] Fwd(float[][][] a, float[][][] b, out object ctx)
            {
                ctx = new[] { a, b };
                return Utilz.Matmul(a, b);
            }
            /// <summary>
            /// Matrix Multiplication float[4d]
            /// </summary>
            /// <param name="a">values a</param>
            /// <param name="b">values b</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>float[4d]</returns>
            public static float[][][][] Fwd(float[][][][] a, float[][][][] b, out object ctx)
            {
                ctx = new[] { a, b };
                return Utilz.Matmul(a, b);
            }

            /// <summary>
            /// Matrix Multiplication Backward pass float[3d]
            /// </summary>
            /// <param name="gradout">Output gradient</param>
            /// <param name="gradA">Returned Input gradient of 'a'</param>
            /// <param name="gradB">Returned Input gradient of 'b'</param>
            /// <param name="ctx">Context for backwards pass</param>
            public static void Bck(float[][][] gradout, out float[][][] gradA, out float[][][] gradB, object ctx)
            {
                var vs = (float[][][][])ctx;
                var a = vs[0];
                var b = vs[1];
                // Validate Shape
                if (gradout.Length != a.Length || gradout[0].Length != a[0].Length || gradout[0][0].Length != b[0][0].Length)
                    throw new Exception($"Invalid gradient shape: got ({gradout.Length}, {gradout[0].Length}, {gradout[0][0].Length}) should be ({a.Length}, {a[0].Length}, {b[0][0].Length}).");
                // Input gradient of 'a'
                gradA = Utilz.Matmul(gradout, Utilz.Transpose(b, 1, 2));
                // Input gradient of 'b'
                gradB = Utilz.Matmul(Utilz.Transpose(a, 1, 2), gradout);
            }
            /// <summary>
            /// Matrix Multiplication Backward pass float[4d]
            /// </summary>
            /// <param name="gradout">Output gradient</param>
            /// <param name="gradA">Returned Input gradient of 'a'</param>
            /// <param name="gradB">Returned Input gradient of 'b'</param>
            /// <param name="ctx">Context for backwards pass</param>
            public static void Bck(float[][][][] gradout, out float[][][][] gradA, out float[][][][] gradB, object ctx)
            {
                var vs = (float[][][][][])ctx;
                var a = vs[0];
                var b = vs[1];
                // Validate Shape
                if (gradout.Length != a.Length || gradout[0].Length != a[0].Length ||
                    gradout[0][0].Length != a[0][0].Length || gradout[0][0][0].Length != b[0][0][0].Length)
                    throw new Exception($"Invalid gradient shape: got ({gradout.Length}, {gradout[0].Length}, {gradout[0][0].Length}, {gradout[0][0][0].Length}) " +
                            $"it should be ({a.Length}, {a[0].Length}, {a[0][0].Length}, {b[0][0][0].Length}).");
                // Input gradient of 'a'
                gradA = Utilz.Matmul(gradout, Utilz.Transpose(b, 2, 3));
                // Input gradient of 'b'
                gradB = Utilz.Matmul(Utilz.Transpose(a, 2, 3), gradout);
            }
        }

        /// <summary>
        /// Takes the first number of values on a dimention. The number of values is of size 'take.'
        /// Inputs can be float[2d - 3d]
        /// </summary>
        public static class Take
        {
            /// <summary>
            /// Takes the first number of values on a dimention. The number of values is of size 'take.'
            /// </summary>
            /// <param name="x">Series to modify</param>
            /// <param name="take">The size to keep on dimention</param>
            /// <param name="dim">The dim the operation works on</param>
            /// <returns>Shape is the same except on dim: length is size 'take'</returns>
            public static Tensor<float[][]> Fwd(Tensor<float[][]> x, int take, int dim)
            {
                object ctx;
                float[][] y;
                AutoGrad.BckF bckF;
                var values = x._data;
                switch (dim)
                {
                    // dim0
                    case 0:
                        // Validation
                        if (values.Length < take)
                            throw new Exception($"values.Length({values.Length}) < take({take})");
                        // Context
                        ctx = new[] { dim, values.Length - take, take, values[0].Length };
                        // Forward Pass
                        y = values.Take(take).ToArray();
                        // Backwards Function
                        bckF = Bck2d_dim0;
                        break;
                    // dim1
                    case 1:
                        // Validation
                        if (values[0].Length < take)
                            throw new Exception($"values[0].Length({values[0].Length}) < take({take})");
                        // Context
                        ctx = new[] { dim, values[0].Length - take, values.Length, take };
                        // Forward Pass
                        y = values.Select(s => s.Take(take).ToArray()).ToArray();
                        // Backwards Function
                        bckF = Bck2d_dim1;
                        break;
                    default:
                        throw new Exception("dim < 0 || dim > 1");
                }
                return x.AddNode(y, ctx, bckF, false);
            }
            /// <summary>
            /// Takes the first number of values on a dimention. The number of values is of size 'take.'
            /// </summary>
            /// <param name="x">Series to modify</param>
            /// <param name="take">The size to keep on dimention</param>
            /// <param name="dim">The dim the operation works on</param>
            /// <returns>Shape is the same except on dim: length is size 'take'</returns>
            public static Tensor<float[][][]> Fwd(Tensor<float[][][]> x, int take, int dim)
            {
                object ctx;
                float[][][] y;
                AutoGrad.BckF bckF;
                var values = x._data;
                switch (dim)
                {
                    // dim0
                    case 0:
                        // Validation
                        if (values.Length < take)
                            throw new Exception($"values.Length({values.Length}) < take({take})");
                        // Context
                        ctx = new[] { dim, values.Length - take, take, values[0].Length, values[0][0].Length };
                        // Forwards Pass
                        y = values.Take(take).ToArray();
                        // Backwards Function
                        bckF = Bck3d_dim0;
                        break;
                    // dim1
                    case 1:
                        // Validation
                        if (values[0].Length < take)
                            throw new Exception($"values[0].Length({values[0].Length}) < take({take})");
                        // Context
                        ctx = new[] { dim, values[0].Length - take, values.Length, take, values[0][0].Length };
                        // Forwards Pass
                        y = values.Select(s => s.Take(take).ToArray()).ToArray();
                        // Backwards Function
                        bckF = Bck3d_dim1;
                        break;
                    // dim2
                    case 2:
                        // Validation
                        if (values[0][0].Length < take)
                            throw new Exception($"values[0][0].Length({values[0][0].Length}) < take({take})");
                        // Context
                        ctx = new[] { dim, values[0][0].Length - take, values.Length, values[0].Length, take };
                        // Forwards Pass
                        y = values.Select(s => s.Select(l => l.Take(take).ToArray()).ToArray()).ToArray();
                        // Backwards Function
                        bckF = Bck3d_dim2;
                        break;
                    default:
                        throw new Exception("dim < 0 || dim > 2");
                }
                return x.AddNode(y, ctx, bckF, false);
            }
            /// <summary>
            /// Takes the first number of values on a dimention. The number of values is of size 'take.'
            /// </summary>
            /// <typeparam name="T">Type of values.</typeparam>
            /// <param name="values">Series to modify</param>
            /// <param name="take">The size to keep on dimention</param>
            /// <param name="dim">The dim the operation works on</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Shape is the same except on dim: length is size 'take'</returns>
            public static T[][][] Fwd<T>(T[][][] values, int take, int dim, out object ctx)
            {
                // dim0
                if (dim == 0)
                {
                    // Validation
                    if (values.Length < take)
                        throw new Exception($"values.Length({values.Length}) < take({take})");
                    // Context
                    ctx = new[] { dim, values.Length - take, take, values[0].Length, values[0][0].Length };
                    // Forwards Pass
                    return values.Take(take).ToArray();
                }
                // dim1
                else if (dim == 1)
                {
                    // Validation
                    if (values[0].Length < take)
                        throw new Exception($"values[0].Length({values[0].Length}) < take({take})");
                    // Context
                    ctx = new[] { dim, values[0].Length - take, values.Length, take, values[0][0].Length };
                    // Forwards Pass
                    return values.Select(s => s.Take(take).ToArray()).ToArray();
                }
                // dim2
                else if (dim == 2)
                {
                    // Validation
                    if (values[0][0].Length < take)
                        throw new Exception($"values[0][0].Length({values[0][0].Length}) < take({take})");
                    // Context
                    ctx = new[] { dim, values[0][0].Length - take, values.Length, values[0].Length, take };
                    // Forwards Pass
                    return values.Select(s => s.Select(l => l.Take(take).ToArray()).ToArray()).ToArray();
                }
                else
                    throw new Exception("dim < 0 || dim > 2");
            }
            /// <summary>
            /// float[2d]: Takes the first number of values on a dimention. The number of values is of size 'take.'
            /// </summary>
            /// <typeparam name="T">Type of values.</typeparam>
            /// <param name="values">Series to modify</param>
            /// <param name="take">The size to keep on dimention</param>
            /// <param name="dim">The dim the operation works on</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Shape is the same except on dim: length is size 'take'</returns>
            public static T[][] Fwd<T>(T[][] values, int take, int dim, out object ctx)
            => Fwd(new[] { values }, take, dim + 1, out ctx)[0];

            /// <summary>
            /// Inside Tensor Backwards pass
            /// </summary>
            /// <param name="y">Output node from forward pass</param>
            static void Bck3d_dim0(AutoGrad.BackwardNode y)
            {
                var ctx = (int[])y._ctx;
                var grad = (float[][][])y._grad;
                // Validate Context size
                if (ctx.Length != 5)
                    throw new Exception($"ctx.Length({ctx.Length}) != 5");
                // Validate Shape
                if (grad.Length != ctx[2] || grad[0].Length != ctx[3] || grad[0][0].Length != ctx[4])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                                $"it should be ({ctx[2]}, {ctx[3]}, {ctx[4]}).");
                // Padding
                var pad = ctx[1];
                var back = new float[grad[0][0].Length];
                var zeros = new bool[grad[0].Length].Select(no => back).ToArray();
                // Backwards Pass
                var ingrad = grad.Concat(new bool[pad].Select(no => zeros)).ToArray();
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Tensor Backwards pass
            /// </summary>
            /// <param name="y">Output node from forward pass</param>
            static void Bck3d_dim1(AutoGrad.BackwardNode y)
            {
                var ctx = (int[])y._ctx;
                var grad = (float[][][])y._grad;
                // Validate Context size
                if (ctx.Length != 5)
                    throw new Exception($"ctx.Length({ctx.Length}) != 5");
                // Validate Shape
                if (grad.Length != ctx[2] || grad[0].Length != ctx[3] || grad[0][0].Length != ctx[4])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                                $"it should be ({ctx[2]}, {ctx[3]}, {ctx[4]}).");
                // Padding
                var pad = ctx[1];
                var back = new float[grad[0][0].Length];
                var zeros = new bool[pad].Select(no => back).ToArray();
                // Backwards Pass
                var ingrad = grad.Select(g => g.Concat(zeros).ToArray()).ToArray();
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Tensor Backwards pass
            /// </summary>
            /// <param name="y">Output node from forward pass</param>
            static void Bck3d_dim2(AutoGrad.BackwardNode y)
            {
                var ctx = (int[])y._ctx;
                var grad = (float[][][])y._grad;
                // Validate Context size
                if (ctx.Length != 5)
                    throw new Exception($"ctx.Length({ctx.Length}) != 5");
                // Validate Shape
                if (grad.Length != ctx[2] || grad[0].Length != ctx[3] || grad[0][0].Length != ctx[4])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                                $"it should be ({ctx[2]}, {ctx[3]}, {ctx[4]}).");
                // Padding
                var pad = ctx[1];
                var back = new float[pad];
                // Backwards Pass
                var ingrad = grad.Select(gi => gi.Select(g => g.Concat(back).ToArray()).ToArray()).ToArray();
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Tensor Backwards pass
            /// </summary>
            /// <param name="y">Output node from forward pass</param>
            static void Bck2d_dim0(AutoGrad.BackwardNode y)
            {
                var ctx = (int[])y._ctx;
                var grad = (float[][])y._grad;
                // Validate Context size
                if (ctx.Length != 4)
                    throw new Exception($"ctx.Length({ctx.Length}) != 4");
                // Validate Shape
                if (grad.Length != ctx[2] || grad[0].Length != ctx[3])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}) " +
                                $"it should be ({ctx[2]}, {ctx[3]}).");
                // Padding
                var pad = ctx[1];
                var back = new float[grad[0].Length];
                var zeros = new bool[pad].Select(no => back).ToArray();
                // Backwards Pass
                var ingrad = grad.Concat(zeros).ToArray();
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Tensor Backwards pass
            /// </summary>
            /// <param name="y">Output node from forward pass</param>
            static void Bck2d_dim1(AutoGrad.BackwardNode y)
            {
                var ctx = (int[])y._ctx;
                var grad = (float[][])y._grad;
                // Validate Context size
                if (ctx.Length != 4)
                    throw new Exception($"ctx.Length({ctx.Length}) != 4");
                // Validate Shape
                if (grad.Length != ctx[2] || grad[0].Length != ctx[3])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}) " +
                                $"it should be ({ctx[2]}, {ctx[3]}).");
                // Padding
                var pad = ctx[1];
                var back = new float[pad];
                // Backwards Pass
                var ingrad = grad.Select(g => g.Concat(back).ToArray()).ToArray();
                y.GradSum(ingrad);
            }

            /// <summary>
            /// Inside Backwards pass
            /// </summary>
            /// <typeparam name="T">Type of 'grads'</typeparam>
            /// <param name="grads">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input array in forwards pass</returns>
            public static T[][][] Bck<T>(T[][][] grads, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 5)
                    throw new Exception($"ctx.Length({ctx.Length}) != 5");
                // Validate Shape
                if (grads.Length != ctx[2] || grads[0].Length != ctx[3] || grads[0][0].Length != ctx[4])
                    throw new Exception($"Invalid gradient shape: got ({grads.Length}, {grads[0].Length}, {grads[0][0].Length}) " +
                                $"it should be ({ctx[2]}, {ctx[3]}, {ctx[4]}).");
                var dim = ctx[0];
                var pad = ctx[1];
                if (dim < 2)
                {
                    // Padding
                    var back = new T[grads[0][0].Length];
                    // dim0
                    if (dim == 0)
                    {
                        // Padding
                        var zeros = new bool[grads[0].Length].Select(no => back).ToArray();
                        // Backwards Pass
                        return grads.Concat(new bool[pad].Select(no => zeros)).ToArray();
                    }
                    // dim1
                    else
                    {
                        // Padding
                        var zeros = new bool[pad].Select(no => back).ToArray();
                        // Backwards Pass
                        return grads.Select(g => g.Concat(zeros).ToArray()).ToArray();
                    }
                }
                // dim2
                else
                {
                    // Padding
                    var back = new T[pad];
                    // Backwards Pass
                    return grads.Select(gi => gi.Select(g => g.Concat(back).ToArray()).ToArray()).ToArray();
                }
            }
            /// <summary>
            /// Inside Backwards pass
            /// </summary>
            /// <typeparam name="T">Type of 'grads'</typeparam>
            /// <param name="grads">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input array in forwards pass</returns>
            public static T[][] Bck<T>(T[][] grads, object ctx_)
            => Bck(new[] { grads }, ctx_)[0];
        }

        /// <summary>
        /// Flattens two dimensions into one.
        /// Input can be float[2d-3d]
        /// </summary>
        public static class Flatten
        {
            /// <summary>
            /// Tensor(float[2d]): Flattens two dimensions into one.
            /// </summary>
            /// <param name="x">Series to flatten</param>
            /// <returns>Tensor(float[1d])</returns>
            public static Tensor<float[]> Fwd(Tensor<float[][]> x)
            {
                var values = x._data;
                // Forwards Pass
                var y = values.SelectMany(s => s).ToArray();
                // Context
                var ctx = new[] { values.Length, values[0].Length};
                // Add Tensor to Graph
                return x.AddNode(y, ctx, Bck2d, false);
            }
            /// <summary>
            /// Tensor(float[3d]): Flattens two dimensions into one.
            /// </summary>
            /// <param name="x">Series to flatten</param>
            /// <param name="dim">Dimention index to flatten on</param>
            /// <returns>Tensor(float[2d])</returns>
            public static Tensor<float[][]> Fwd(Tensor<float[][][]> x, int dim)
            {
                float[][] y;
                var values = x._data;
                // dim0
                if (dim == 0)
                    // Forwards Pass
                    y = values.SelectMany(s => s).ToArray();
                // dim1
                else if (dim == 1)
                    // Forwards Pass
                    y = values.Select(xi => xi.SelectMany(s => s).ToArray()).ToArray();
                else throw new Exception("dim < 0 || dim > 1");
                // Context
                var ctx = new[] { values.Length, values[0].Length, values[0][0].Length, y.Length, y[0].Length };
                // Add Tensor to Graph
                return x.AddNode(y, ctx, Bck3d, false);
            }
            /// <summary>
            /// T[3d]: Flattens two dimensions into one.
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Series to flatten</param>
            /// <param name="dim">Dimention index to flatten on</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>T[2d]</returns>
            public static T[][] Fwd<T>(T[][][] values, int dim, out object ctx)
            {
                if (dim == 0)
                    return Fwd_d0(values, out ctx);
                else if (dim == 1)
                    return Fwd_d1(values, out ctx);
                else throw new Exception("dim < 0 || dim > 1");
            }
            /// <summary>
            /// T[3d]: On dimention index 0: Flattens two dimensions into one.
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Series to flatten</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Tensor(T[2d])</returns>
            public static T[][] Fwd_d0<T>(T[][][] values, out object ctx)
            {
                ctx = new[] { 0, values.Length, values[0].Length, values[0][0].Length };
                return values.SelectMany(l => l).ToArray();
            }
            /// <summary>
            /// T[3d]: On dimention index 1: Flattens two dimensions into one.
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Series to flatten</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Tensor(T[2d])</returns>
            public static T[][] Fwd_d1<T>(T[][][] values, out object ctx)
            {
                ctx = new[] { 1, values.Length, values[0].Length, values[0][0].Length };
                return values.Select(l => l.SelectMany(s => s).ToArray()).ToArray();
            }

            /// <summary>
            /// Inside Backwards Function. Input to forward pass is 2d.
            /// </summary>
            /// <param name="y">Output node from forward pass</param>
            static void Bck2d(AutoGrad.BackwardNode y)
            {
                var ctx = (int[])y._ctx;
                // Validate Context size
                if (ctx.Length != 2)
                    throw new Exception($"ctx.Length({ctx.Length}) != 2");
                var dim0 = ctx[0];
                var dim1 = ctx[1];
                var grad = (float[])y._grad;
                // Validate Shape
                if (grad.Length != dim0 * dim1)
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}) " +
                                    $"it should be ({dim0 * dim1}).");
                // Backwards pass
                var ingrad = Utilz.Reshape(grad, dim0, dim1);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Function. Input to forward pass is 3d.
            /// </summary>
            /// <param name="y">Output node from forward pass</param>
            static void Bck3d(AutoGrad.BackwardNode y)
            {
                var ctx = (int[])y._ctx;
                // Validate Context size
                if (ctx.Length != 5)
                    throw new Exception($"ctx.Length({ctx.Length}) != 5");
                var grad = (float[][])y._grad;
                // Validate Shape
                if (grad.Length != ctx[3] || grad[0].Length != ctx[4])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}) " +
                                    $"it should be ({ctx[3]}, {ctx[4]}).");
                // Backwards pass
                var ingrad = Utilz.Reshape(grad, ctx[0], ctx[1], ctx[2]);
                y.GradSum(ingrad);
            }

            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <typeparam name="T">Type of 'grads'</typeparam>
            /// <param name="grads">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input array to forwards pass</returns>
            public static T[][][] Bck<T>(T[][] grads, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 4)
                    throw new Exception($"ctx.Length({ctx.Length}) != 4");
                var dim = ctx[0];
                // dim0
                if (dim == 0)
                    return Bck_d0(grads, ctx_);
                // dim1
                else if (dim == 1)
                    return Bck_d1(grads, ctx_);
                throw new Exception("There was a unknown error.");
            }
            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <typeparam name="T">Type of 'grads'</typeparam>
            /// <param name="grads">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input array to forwards pass</returns>
            public static T[][][] Bck_d0<T>(T[][] grads, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 4)
                    throw new Exception($"ctx.Length({ctx.Length}) != 4");
                // Make sure it is the same dim as forwards pass
                if (ctx[0] != 0) throw new Exception($"You should be using Bwd_d{ctx[0]}() backwards function.");
                // Validate Shape
                if (grads.Length != ctx[1] * ctx[2] || grads[0].Length != ctx[3])
                    throw new Exception($"Invalid gradient shape: got ({grads.Length}, {grads[0].Length}) " +
                                    $"it should be ({ctx[1] * ctx[2]}, {ctx[3]}).");
                // Backwards pass
                return Utilz.Reshape(grads, ctx[1], ctx[2]);
            }
            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <typeparam name="T">Type of 'grads'</typeparam>
            /// <param name="grads">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input array to forwards pass</returns>
            public static T[][][] Bck_d1<T>(T[][] grads, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 4)
                    throw new Exception($"ctx.Length({ctx.Length}) != 4");
                // Make sure it is the same dim as forwards pass
                if (ctx[0] != 1) throw new Exception($"You should be using Bwd_d{ctx[0]}() backwards function.");
                var dim1 = ctx[2];
                var dim2 = ctx[3];
                // Validate Shape
                if (grads.Length != ctx[1] || grads[0].Length != dim1 * dim2)
                    throw new Exception($"Invalid gradient shape: got ({grads.Length}, {grads[0].Length}) " +
                                    $"it should be ({ctx[1]}, {dim1 * dim2}).");
                // Backwards pass
                return Utilz.Reshape(grads, grads.Length, dim1, dim2);
            }
        }

        /// <summary>
        /// Concatenates values on specific dimension.
        /// Inputs can be float[2d - 3d]
        /// </summary>
        public static class Cat
        {
            /// <summary>
            /// Concatenates Tensors on specific dimension.
            /// </summary>
            /// <param name="dim">The dimention to concatenate Tensors on</param>
            /// <param name="nodes">The Tensors to concatenate</param>
            /// <returns>Same shape as single node except on 'dim' the length is summed.</returns>
            public static Tensor<float[][]> Fwd(int dim, params Tensor<float[][]>[] nodes)
            {
                // Validation
                if (nodes.Length < 2)
                    throw new Exception($"There needs to be at least 2 nodes: got ({nodes.Length}).");
                object ctx;
                float[][] y;
                AutoGrad.BckF bckF;
                // Values. Make sure first dim is nodes.Count
                var xxv = new[] { nodes.Select(n => n._data).ToArray() };
                xxv = Utilz.TransposeArray(xxv);
                switch (dim)
                {
                    // dim0
                    case 0:
                        // Forwards pass
                        y = Fwd_d1(xxv, out ctx, -1)[0];
                        // Backwards pass
                        bckF = Bck_dim0_2d;
                        break;
                    // dim1
                    case 1:
                        // Forwards pass
                        y = Fwd_d2(xxv, out ctx, -1)[0];
                        // Backwards pass
                        bckF = Bck_dim1_2d;
                        break;
                    default:
                        throw new Exception("dim < 0 || dim > 1");
                }
                // Add Consolidation node: Combines multiple nodes into one
                return Tensor<float[][]>.AddConsolidate(y, ctx, bckF, false, nodes);
            }
            /// <summary>
            /// Concatenates Tensors on specific dimension.
            /// </summary>
            /// <param name="dim">The dimention to concatenate Tensors on</param>
            /// <param name="nodes">The Tensors to concatenate</param>
            /// <returns>Same shape as single node except on 'dim' the length is summed.</returns>
            public static Tensor<float[][][]> Fwd(int dim, params Tensor<float[][][]>[] nodes)
            {
                // Validation
                if (nodes.Length < 2)
                    throw new Exception($"There needs to be at least 2 nodes: got ({nodes.Length}).");
                object ctx;
                float[][][] y;
                AutoGrad.BckF bckF;
                // Get Nodes values
                var xxv = nodes.Select(n => n._data).ToArray();
                switch (dim)
                {
                    // dim0
                    case 0:
                        // Forwards pass
                        y = Fwd_d0(xxv, out ctx);
                        // Backwards pass
                        bckF = Bck_dim0_3d;
                        break;
                    // dim1
                    case 1:
                        // Forwards pass
                        y = Fwd_d1(xxv, out ctx);
                        // Backwards pass
                        bckF = Bck_dim1_3d;
                        break;
                    // dim2
                    case 2:
                        // Forwards pass
                        y = Fwd_d2(xxv, out ctx);
                        // Backwards pass
                        bckF = Bck_dim2_3d;
                        break;
                    default:
                        throw new Exception("dim < 0 || dim > 2");
                }
                // Add Consolidation node: Combines multiple nodes into one
                return Tensor<float[][][]>.AddConsolidate(y, ctx, bckF, false, nodes);
            }
            /// <summary>
            /// Concatenates Arrays on specific dimension.
            /// </summary>
            /// <typeparam name="T">Type of 'nodes'</typeparam>
            /// <param name="nodes">The Arrays to concatenate</param>
            /// <param name="dim">The dimention to concatenate Arrays on</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="dm">Do NOT modify</param>
            /// <returns>Same shape as single array except on 'dim' the length is summed.</returns>
            public static T[][][] Fwd<T>(T[][][][] nodes, int dim, out object ctx, int dm = 0)
            {
                switch (dim)
                {
                    // dim0
                    case 0:
                        return Fwd_d0(nodes, out ctx, dm);
                    // dim1
                    case 1:
                        return Fwd_d1(nodes, out ctx, dm);
                    // dim2
                    case 2:
                        return Fwd_d2(nodes, out ctx, dm);
                }
                throw new Exception("dim < 0 || dim > 2");
            }

            /// <summary>
            /// Concatenates Arrays on dimension index (0).
            /// </summary>
            /// <typeparam name="T">Type of 'nodes'</typeparam>
            /// <param name="nodes">The Arrays to concatenate</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="dm">Do NOT modify</param>
            /// <returns>Same shape as single array except on 'dim' the length is summed.</returns>
            public static T[][][] Fwd_d0<T>(T[][][][] nodes, out object ctx, int dm = 0)
            {
                // Validation
                if (nodes.Length < 2)
                    throw new Exception("There are no values to concatenate! nodes.Length < 2");
                // Make sure other dims are of same length
                var d = new[] { nodes[0][0].Length, nodes[0][0][0].Length };
                foreach (var nd in nodes)
                {
                    if (nd[0].Length != d[0])
                        throw new Exception($"Incompatable shape at dim {1 + dm}.");
                    if (nd[0][0].Length != d[1])
                        throw new Exception($"Incompatable shape at dim {2 + dm}.");
                }
                // Forwards Pass
                var y = nodes.SelectMany(n => n).ToArray();
                // Context
                ctx = new[] { 0, y.Length, y[0].Length, y[0][0].Length }
                        .Concat(nodes.Select(nd => nd.Length)).ToArray();
                return y;
            }
            /// <summary>
            /// Concatenates Arrays on dimension index (1).
            /// </summary>
            /// <typeparam name="T">Type of 'nodes'</typeparam>
            /// <param name="nodes">The Arrays to concatenate</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="dm">Do NOT modify</param>
            /// <returns>Same shape as single array except on 'dim' the length is summed.</returns>
            public static T[][][] Fwd_d1<T>(T[][][][] nodes, out object ctx, int dm = 0)
            {
                // Validation
                if (nodes.Length < 2)
                    throw new Exception("There are no values to concatenate! nodes.Length < 2");
                // Make sure other dims are of same length
                var d = new[] { nodes[0].Length, nodes[0][0][0].Length };
                foreach (var nd in nodes.Skip(1))
                {
                    if (nd.Length != d[0])
                        throw new Exception($"Incompatable shape at dim {dm}.");
                    if (nd[0][0].Length != d[1])
                        throw new Exception($"Incompatable shape at dim {dm + 2}.");
                }
                // Forwards Pass
                var y = Enumerable.Range(0, nodes[0].Length).Select(i =>
                        nodes.SelectMany(nd => nd[i]).ToArray()).ToArray();
                // Context
                ctx = new[] { 1, y.Length, y[0].Length, y[0][0].Length }
                        .Concat(nodes.Select(nd => nd[0].Length)).ToArray();
                return y;
            }
            /// <summary>
            /// Concatenates Arrays on dimension index (2).
            /// </summary>
            /// <typeparam name="T">Type of 'nodes'</typeparam>
            /// <param name="nodes">The Arrays to concatenate</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="dm">Do NOT modify</param>
            /// <returns>Same shape as single array except on 'dim' the length is summed.</returns>
            public static T[][][] Fwd_d2<T>(T[][][][] nodes, out object ctx, int dm = 0)
            {
                // Validation
                if (nodes.Length < 2)
                    throw new Exception("There are no values to concatenate! nodes.Length < 2");
                // Make sure other dims are of same length
                var d = new[] { nodes[0].Length, nodes[0][0].Length };
                foreach (var nd in nodes.Skip(1))
                {
                    if (nd.Length != d[0])
                        throw new Exception($"Incompatable shape at dim {dm}.");
                    if (nd[0].Length != d[1])
                        throw new Exception($"Incompatable shape at dim {dm + 1}.");
                }
                // Forwards Pass
                var d1 = Enumerable.Range(0, nodes[0][0].Length).ToArray();
                var y = Enumerable.Range(0, nodes[0].Length).Select(i => d1.Select(j =>
                        nodes.SelectMany(nd => nd[i][j]).ToArray()).ToArray()).ToArray();
                // Context
                ctx = new[] { 2, y.Length, y[0].Length, y[0][0].Length }
                        .Concat(nodes.Select(nd => nd[0][0].Length)).ToArray();
                return y;
            }

            /// <summary>
            /// Backwards Pass
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Shape[nodes length][each shape...]</returns>
            public static T[][][][] Bck<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 6)
                    throw new Exception("This is not the right function for backward.");
                switch (ctx[0])
                {
                    case 0:     // dim0
                        return Bck_d0(grad, ctx_);
                    case 1:     // dim1
                        return Bck_d1(grad, ctx_);
                    case 2:     // dim2
                        return Bck_d2(grad, ctx_);
                }
                throw new Exception("There was a unknown error.");
            }

            /// <summary>
            /// Backwards Pass on dimention index (0)
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Shape[nodes length][each shape...]</returns>
            public static T[][][][] Bck_d0<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 6)
                    throw new Exception("This is not the right function for backward.");
                // Make sure it is the same dimention as forward pass
                if (ctx[0] != 0)
                    throw new Exception($"You should be using Bck_d{ctx[0]}() backward function.");
                // Validate Shape
                if (grad.Length != ctx[1] || grad[0].Length != ctx[2] || grad[0][0].Length != ctx[3])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                                        $"it should be ({ctx[1]}, {ctx[2]}, {ctx[3]}).");
                // Backwards Pass
                return Bck_sub(grad, ctx.Skip(4).ToArray());
            }
            /// <summary>
            /// Backwards Pass on dimention index (1)
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Shape[nodes length][each shape...]</returns>
            public static T[][][][] Bck_d1<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 6)
                    throw new Exception("This is not the right function for backward.");
                // Make sure it is the same dimention as forward pass
                if (ctx[0] != 1)
                    throw new Exception($"You should be using Bck_d{ctx[0]}() backward function.");
                // Validate Shape
                if (grad.Length != ctx[1] || grad[0].Length != ctx[2] || grad[0][0].Length != ctx[3])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                                        $"it should be ({ctx[1]}, {ctx[2]}, {ctx[3]}).");
                // Get Origonal Lengths on dim
                var len = ctx.Skip(4).ToArray();
                // Backwards Pass
                var ingrad = len.Select(ln => grad.Select(no => new T[ln][]).ToArray()).ToArray();
                // Loop dim0
                for (int i = 0; i < grad.Length; i++)
                {
                    // Make sure the size of nodes is on dim0
                    var nodes = Bck_sub(grad[i], len);
                    for (int j = 0; j < len.Length; j++)
                        ingrad[j][i] = nodes[j];
                }
                return ingrad;
            }
            /// <summary>
            /// Backwards Pass on dimention index (2)
            /// </summary>
            /// <typeparam name="T">Type of grad</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Shape[nodes length][each shape...]</returns>
            public static T[][][][] Bck_d2<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 6)
                    throw new Exception("This is not the right function for backward.");
                // Make sure it is the same dimention as forward pass
                if (ctx[0] != 2)
                    throw new Exception($"You should be using Bck_d{ctx[0]}() backward function.");
                // Validate Shape
                if (grad.Length != ctx[1] || grad[0].Length != ctx[2] || grad[0][0].Length != ctx[3])
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                                        $"it should be ({ctx[1]}, {ctx[2]}, {ctx[3]}).");
                // Get Origonal Lengths on dim
                var len = ctx.Skip(4).ToArray();
                // Backwards Pass
                var ingrad = len.Select(ln => grad.Select(gi => gi.Select(no => new T[ln])
                             .ToArray()).ToArray()).ToArray();
                // Loop dim0
                for (int i = 0; i < grad.Length; i++)
                {
                    var gi = grad[i];
                    // Loop dim1
                    for (int j = 0; j < gi.Length; j++)
                    {
                        var g = gi[j];
                        var start = 0;
                        // Loop each concatenated node
                        for (int k = 0; k < len.Length; k++)
                        {
                            var ln = len[k];
                            var end = start + ln;
                            var ing = ingrad[k][i][j];  // in grad
                            // get node values
                            for (int l = start; l < end; l++)
                                ing[l - start] = g[l];
                            // shift to next node
                            start += ln;
                        }
                    }
                }
                return ingrad;
            }

            /// <summary>
            /// Inside Backwards Function on dim index (0) & Tensor(float[2d])
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_dim0_2d(AutoGrad.BackwardNode y)
            {
                var grad = new[] { (float[][])y._grad };
                var ingrad = Bck_d1(grad, y._ctx);
                y.GradSum(Utilz.TransposeArray(ingrad)[0]);
            }
            /// <summary>
            /// Inside Backwards Function on dim index (1) & Tensor(float[2d])
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_dim1_2d(AutoGrad.BackwardNode y)
            {
                var grad = new[] { (float[][])y._grad };
                var ingrad = Bck_d2(grad, y._ctx);
                y.GradSum(Utilz.TransposeArray(ingrad)[0]);
            }
            /// <summary>
            /// Inside Backwards Function on dim index (0) & Tensor(float[3d])
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_dim0_3d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck_d0((float[][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Function on dim index (1) & Tensor(float[3d])
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_dim1_3d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck_d1((float[][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Function on dim index (2) & Tensor(float[3d])
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_dim2_3d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck_d2((float[][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }

            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Concatenated values</param>
            /// <param name="lens">length of each node</param>
            /// <returns>Shape T[node.Count]...</returns>
            static T[][] Bck_sub<T>(T[] values, int[] lens)
            {
                var start = 0;
                var ar = lens.Select(ln => new T[ln]).ToArray();
                // Loop each node
                for (int j = 0; j < lens.Length; j++)
                {
                    var ln = lens[j];
                    var end = start + ln;
                    var arj = ar[j];
                    // get each nodes values
                    for (int i = start; i < end; i++)
                        arj[i - start] = values[i];
                    // Shift to next node
                    start += ln;
                }
                return ar;
            }
        }
        
        /// <summary>
        /// Repeats values on Specific Dimentions.
        /// Inputs can be float[1d - 4d]
        /// </summary>
        public static class Repeat
        {
            /// <summary>
            /// Repeats Tensor on Specific Dimention
            /// </summary>
            /// <param name="x">Tensor to repeat</param>
            /// <param name="dim0">Repeat multiplicative size</param>
            /// <returns>Shape[x.length * dim0]</returns>
            public static Tensor<float[]> Fwd(Tensor<float[]> x, int dim0)
            {
                var y = Fwd(x._data, dim0, out object ctx);
                return x.AddNode(y, ctx, Bck_1d, false);
            }
            /// <summary>
            /// Repeats Tensor on Specific Dimentions
            /// </summary>
            /// <param name="x">Tensor to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat multiplicative size</param>
            /// <param name="dim1">Dimention 1: Repeat multiplicative size</param>
            /// <returns>Shape[x.length * dim0][x[0].length * dim1]</returns>
            public static Tensor<float[][]> Fwd(Tensor<float[][]> x, int dim0, int dim1 = 1)
            {
                var y = Fwd(x._data, dim0, dim1, out object ctx);
                return x.AddNode(y, ctx, Bck_2d, false);
            }
            /// <summary>
            /// Repeats Tensor on Specific Dimentions
            /// </summary>
            /// <param name="x">Tensor to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat multiplicative size</param>
            /// <param name="dim1">Dimention 1: Repeat multiplicative size</param>
            /// <param name="dim2">Dimention 2: Repeat multiplicative size</param>
            /// <returns>Shape[x.length*dim0][x[0].length*dim1][x[0][0].length*dim2]</returns>
            public static Tensor<float[][][]> Fwd(Tensor<float[][][]> x, int dim0, int dim1 = 1, int dim2 = 1)
            {
                var y = Fwd(x._data, dim0, dim1, dim2, out object ctx);
                return x.AddNode(y, ctx, Bck_3d, false);
            }
            /// <summary>
            /// Repeats Tensor on Specific Dimentions
            /// </summary>
            /// <param name="x">Tensor to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat multiplicative size</param>
            /// <param name="dim1">Dimention 1: Repeat multiplicative size</param>
            /// <param name="dim2">Dimention 2: Repeat multiplicative size</param>
            /// <param name="dim3">Dimention 3: Repeat multiplicative size</param>
            /// <returns>Shape[x.length*dim0][x[0].length*dim1][x[0][0].length*dim2][x[0][0][0].length*dim3]</returns>
            public static Tensor<float[][][][]> Fwd(Tensor<float[][][][]> x, int dim0, int dim1 = 1, int dim2 = 1, int dim3 = 1)
            {
                var y = Fwd(x._data, dim0, dim1, dim2, dim3, out object ctx);
                return x.AddNode(y, ctx, Bck_4d, false);
            }
            /// <summary>
            /// Repeats Array on Specific Dimentions
            /// </summary>
            /// <param name="values">Array to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat multiplicative size</param>
            /// <param name="dim1">Dimention 1: Repeat multiplicative size</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Shape[x.length * dim0][x[0].length * dim1]</returns>
            public static T[][] Fwd<T>(T[][] values, int dim0, int dim1, out object ctx)
            {
                var d = new[] { dim0, dim1 };
                // Validate dims >= 1
                if (d.Any(v => v <= 0))
                    throw new Exception($"At least 1 of the dims <= 0: got ({string.Join(", ", d)}).");
                // Context
                ctx = new[] { 1, dim0, dim1, values.Length, values[0].Length };
                // Forwards Pass
                return FwdSub(values, dim0, dim1);
            }
            /// <summary>
            /// Repeats Array on Specific Dimentions
            /// </summary>
            /// <param name="values">Array to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat multiplicative size</param>
            /// <param name="dim1">Dimention 1: Repeat multiplicative size</param>
            /// <param name="dim2">Dimention 2: Repeat multiplicative size</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Shape[x.length*dim0][x[0].length*dim1][x[0][0].length*dim2]</returns>
            public static T[][][] Fwd<T>(T[][][] values, int dim0, int dim1, int dim2, out object ctx)
            {
                var d = new[] { dim0, dim1, dim2 };
                // Validate dims >= 1
                if (d.Any(v => v <= 0))
                    throw new Exception($"At least 1 of the dims <= 0: got ({string.Join(", ", d)}).");
                // Context
                ctx = new[] { 2, dim0, dim1, dim2, values.Length, values[0].Length, values[0][0].Length };
                // Forwards Pass
                return FwdSub(values, dim0, dim1, dim2);
            }
            /// <summary>
            /// Repeats Array on Specific Dimentions
            /// </summary>
            /// <param name="values">Array to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat multiplicative size</param>
            /// <param name="dim1">Dimention 1: Repeat multiplicative size</param>
            /// <param name="dim2">Dimention 2: Repeat multiplicative size</param>
            /// <param name="dim3">Dimention 3: Repeat multiplicative size</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Shape[x.length*dim0][x[0].length*dim1][x[0][0].length*dim2][x[0][0][0].length*dim3]</returns>
            public static T[][][][] Fwd<T>(T[][][][] values, int dim0, int dim1, int dim2, int dim3, out object ctx)
            {
                var d = new[] { dim0, dim1, dim2, dim3 };
                // Validate dims >= 1
                if (d.Any(v => v <= 0))
                    throw new Exception($"At least 1 of the dims <= 0: got ({string.Join(", ", d)}).");
                // Context
                ctx = new[] { 3, dim0, dim1, dim2, dim3, values.Length, values[0].Length, values[0][0].Length, values[0][0][0].Length };
                // Forwards Pass
                return FwdSub(values, dim0, dim1, dim2, dim3);
            }
            /// <summary>
            /// Repeats Array on Specific Dimention
            /// </summary>
            /// <param name="values">Array to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat multiplicative size</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <returns>Shape[x.length*dim0]</returns>
            public static T[] Fwd<T>(T[] values, int dim0, out object ctx)
            {
                // Validate dims >= 1
                if (dim0 <= 0)
                    throw new Exception($"At least 1 of the dims <= 0: got ({dim0}).");
                // Context
                ctx = new[] { 0, dim0, values.Length };
                // Forwards Pass
                return FwdSub(values, dim0);
            }

            /// <summary>
            /// Inside Backwards Pass Function
            /// </summary>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input into forwards pass</returns>
            public static float[] Bck(float[] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 3)
                    throw new Exception($"ctx.Length({ctx.Length}) != 3");
                // Make sure it is using the correct backwards function from forwards function
                if (ctx[0] != 0)
                    throw new Exception($"ctx[0]({ctx[0]}) != 0");
                var dim0 = ctx[1];
                var len0 = ctx[2];
                // Validate Shape
                if (grad.Length != dim0 * len0)
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}) " +
                            $"it should be ({dim0 * len0}).");
                // Backwards Pass
                var ingrad = new float[len0];
                BckSub(grad, ingrad, len0, dim0);
                var div = 1 / (float)Math.Sqrt(dim0);
                Module.MultiplyInPlace(ingrad, div);
                return ingrad;
            }
            /// <summary>
            /// Inside Backwards Pass Function
            /// </summary>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input into forwards pass</returns>
            public static float[][] Bck(float[][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 5)
                    throw new Exception($"ctx.Length({ctx.Length}) != 5");
                // Make sure it is using the correct backwards function from forwards function
                if (ctx[0] != 1)
                    throw new Exception($"ctx[0]({ctx[0]}) != 1");
                var dim0 = ctx[1];
                var dim1 = ctx[2];
                var len0 = ctx[3];
                var len1 = ctx[4];
                // Validate Shape
                if (grad.Length != dim0 * len0 || grad[0].Length != dim1 * len1)
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}) " +
                            $"it should be ({dim0 * len0}, {dim1 * len1}).");
                // Backwards Pass
                var ingrad = new bool[len0].Select(no => new float[len1]).ToArray();
                BckSub(grad, ingrad, len0, len1, dim0, dim1);
                var div = 1 / (float)Math.Sqrt((double)dim0 * dim1);
                Module.MultiplyInPlace(ingrad, div);
                return ingrad;
            }
            /// <summary>
            /// Inside Backwards Pass Function
            /// </summary>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input into forwards pass</returns>
            public static float[][][] Bck(float[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 7)
                    throw new Exception($"ctx.Length({ctx.Length}) != 7");
                // Make sure it is using the correct backwards function from forwards function
                if (ctx[0] != 2)
                    throw new Exception($"ctx[0]({ctx[0]}) != 2");
                var dim0 = ctx[1];
                var dim1 = ctx[2];
                var dim2 = ctx[3];
                var len0 = ctx[4];
                var len1 = ctx[5];
                var len2 = ctx[6];
                // Validate Shape
                if (grad.Length != dim0 * len0 || grad[0].Length != dim1 * len1 || grad[0][0].Length != dim2 * len2)
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                            $"it should be ({dim0 * len0}, {dim1 * len1}, {dim2 * len2}).");
                // Backwards Pass
                var ingrad = new bool[len0].Select(no => new bool[len1].Select(n2 => new float[len2]).ToArray()).ToArray();
                BckSub(grad, ingrad, len0, len1, len2, dim0, dim1, dim2);
                var div = 1 / (float)Math.Sqrt((double)dim0 * dim1 * dim2);
                Module.MultiplyInPlace(ingrad, div);
                return ingrad;
            }
            /// <summary>
            /// Inside Backwards Pass Function
            /// </summary>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Same shape as input into forwards pass</returns>
            public static float[][][][] Bck(float[][][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length != 9)
                    throw new Exception($"ctx.Length({ctx.Length}) != 9");
                // Make sure it is using the correct backwards function from forwards function
                if (ctx[0] != 3)
                    throw new Exception($"ctx[0]({ctx[0]}) != 3");
                var dim0 = ctx[1];
                var dim1 = ctx[2];
                var dim2 = ctx[3];
                var dim3 = ctx[4];
                var len0 = ctx[5];
                var len1 = ctx[6];
                var len2 = ctx[7];
                var len3 = ctx[8];
                // Validate Shape
                if (grad.Length != dim0 * len0 || grad[0].Length != dim1 * len1 || 
                    grad[0][0].Length != dim2 * len2 || grad[0][0][0].Length != dim3 * len3)
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}, {grad[0][0][0].Length}) " +
                            $"it should be ({dim0 * len0}, {dim1 * len1}, {dim2 * len2}, {dim3 * len3}).");
                // Backwards Pass
                var ingrad = new bool[len0].Select(no => new bool[len1].Select(n2 => 
                            new bool[len2].Select(n => new float[len3]).ToArray()).ToArray()).ToArray();
                BckSub(grad, ingrad, len0, len1, len2, len3, dim0, dim1, dim2, dim3);
                var div = 1 / (float)Math.Sqrt((double)dim0 * dim1 * dim2 * dim3);
                Module.MultiplyInPlace(ingrad, div);
                return ingrad;
            }

            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_1d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_2d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_3d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <param name="y">Output node from Forwards pass</param>
            static void Bck_4d(AutoGrad.BackwardNode y)
            {
                var ingrad = Bck((float[][][][])y._grad, y._ctx);
                y.GradSum(ingrad);
            }

            /// <summary>
            /// Forwards Actual Function
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Series to repeat</param>
            /// <param name="dim0">Repeat size</param>
            /// <returns>Shape[values.length*dim0]</returns>
            static T[] FwdSub<T>(T[] values, int dim0)
            {
                // No need to calculate
                if (dim0 == 1) return values;
                // New Dimention size
                var y = new T[values.Length * dim0];
                // Distributes values evenly
                for (int i = 0; i < values.Length; i++)
                {
                    var iii = i * dim0;
                    var v = values[i];
                    // Loop repeat size
                    for (int di = 0; di < dim0; di++)
                        y[iii + di] = v;
                }
                return y;
            }
            /// <summary>
            /// Forwards Actual Function
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Series to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat size</param>
            /// <param name="dim1">Dimention 1: Repeat size</param>
            /// <returns>Shape[values.length*dim0][values[0].length*dim1]</returns>
            static T[][] FwdSub<T>(T[][] values, int dim0, int dim1)
            {
                // Dimention size without repeat on this dimention
                var vals = new T[values.Length][];
                // Get values
                for (int i = 0; i < values.Length; i++)
                    vals[i] = FwdSub(values[i], dim1);
                // No need to calculate
                if (dim0 == 1) return vals;
                // New Dimention size
                var y = new T[values.Length * dim0][];
                // Distributes values evenly
                for (int i = 0; i < values.Length; i++)
                {
                    var iii = i * dim0;
                    // Loop repeat size
                    for (int di = 0; di < dim0; di++)
                        y[iii + di] = vals[i];
                }
                return y;
            }
            /// <summary>
            /// Forwards Actual Function
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Series to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat size</param>
            /// <param name="dim1">Dimention 1: Repeat size</param>
            /// <param name="dim2">Dimention 2: Repeat size</param>
            /// <returns>Shape[values.length*dim0][values[0].length*dim1][values[0][0].length*dim2]</returns>
            static T[][][] FwdSub<T>(T[][][] values, int dim0, int dim1, int dim2)
            {
                // Dimention size without repeat on this dimention
                var vals = new T[values.Length][][];
                // Get values
                for (int i = 0; i < values.Length; i++)
                    vals[i] = FwdSub(values[i], dim1, dim2);
                // No need to calculate
                if (dim0 == 1) return vals;
                // New Dimention size
                var y = new T[values.Length * dim0][][];
                // Distributes values evenly
                for (int i = 0; i < values.Length; i++)
                {
                    var iii = i * dim0;
                    // Loop repeat size
                    for (int di = 0; di < dim0; di++)
                        y[iii + di] = vals[i];
                }
                return y;
            }
            /// <summary>
            /// Forwards Actual Function
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Series to repeat</param>
            /// <param name="dim0">Dimention 0: Repeat size</param>
            /// <param name="dim1">Dimention 1: Repeat size</param>
            /// <param name="dim2">Dimention 2: Repeat size</param>
            /// <param name="dim3">Dimention 3: Repeat size</param>
            /// <returns>Shape[values.length*dim0][values[0].length*dim1][values[0][0].length*dim2][values[0][0][0].length*dim3]</returns>
            static T[][][][] FwdSub<T>(T[][][][] values, int dim0, int dim1, int dim2, int dim3)
            {
                // Dimention size without repeat on this dimention
                var vals = new T[values.Length][][][];
                // Get values
                for (int i = 0; i < values.Length; i++)
                    vals[i] = FwdSub(values[i], dim1, dim2, dim3);
                // No need to calculate
                if (dim0 == 1) return vals;
                // New Dimention size
                var y = new T[values.Length * dim0][][][];
                // Distributes values evenly
                for (int i = 0; i < values.Length; i++)
                {
                    var iii = i * dim0;
                    // Loop repeat size
                    for (int di = 0; di < dim0; di++)
                        y[iii + di] = vals[i];
                }
                return y;
            }

            /// <summary>
            /// Actual Backwards Pass
            /// </summary>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ingrad">Input Gradient</param>
            /// <param name="len">Origonal Length</param>
            /// <param name="dim">Repeat Size</param>
            static void BckSub(float[] grad, float[] ingrad, int len, int dim)
            {
                // Loop origonal input len
                for (int i = 0; i < len; i++)
                {
                    var iii = i * dim;
                    var ig = 0f;
                    // Loop repeat size
                    for (int di = 0; di < dim; di++)
                        ig += grad[iii + di];
                    ingrad[i] += ig;
                }
            }
            /// <summary>
            /// Actual Backwards Pass
            /// </summary>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ingrad">Input Gradient</param>
            /// <param name="len0">Origonal Length 0</param>
            /// <param name="len1">Origonal Length 1</param>
            /// <param name="dim0">Dimention 0: Repeat Size</param>
            /// <param name="dim1">Dimention 1: Repeat Size</param>
            static void BckSub(float[][] grad, float[][] ingrad, int len0, int len1, int dim0, int dim1)
            {
                // Loop origonal input len
                for (int i = 0; i < len0; i++)
                {
                    var iii = i * dim0;
                    var ig = ingrad[i];
                    // Loop repeat size
                    for (int di = 0; di < dim0; di++)
                        BckSub(grad[iii + di], ig, len1, dim1);
                }
            }
            /// <summary>
            /// Actual Backwards Pass
            /// </summary>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ingrad">Input Gradient</param>
            /// <param name="len0">Origonal Length 0</param>
            /// <param name="len1">Origonal Length 1</param>
            /// <param name="len2">Origonal Length 2</param>
            /// <param name="dim0">Dimention 0: Repeat Size</param>
            /// <param name="dim1">Dimention 1: Repeat Size</param>
            /// <param name="dim2">Dimention 2: Repeat Size</param>
            static void BckSub(float[][][] grad, float[][][] ingrad, int len0, int len1, int len2, int dim0, int dim1, int dim2)
            {
                // Loop origonal input len
                for (int i = 0; i < len0; i++)
                {
                    var iii = i * dim0;
                    var ig = ingrad[i];
                    // Loop repeat size
                    for (int di = 0; di < dim0; di++)
                        BckSub(grad[iii + di], ig, len1, len2, dim1, dim2);
                }
            }
            /// <summary>
            /// Actual Backwards Pass
            /// </summary>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ingrad">Input Gradient</param>
            /// <param name="len0">Origonal Length 0</param>
            /// <param name="len1">Origonal Length 1</param>
            /// <param name="len2">Origonal Length 2</param>
            /// <param name="len3">Origonal Length 3</param>
            /// <param name="dim0">Dimention 0: Repeat Size</param>
            /// <param name="dim1">Dimention 1: Repeat Size</param>
            /// <param name="dim2">Dimention 2: Repeat Size</param>
            /// <param name="dim3">Dimention 3: Repeat Size</param>
            static void BckSub(float[][][][] grad, float[][][][] ingrad, int len0, int len1, int len2, int len3, int dim0, int dim1, int dim2, int dim3)
            {
                // Loop origonal input len
                for (int i = 0; i < len0; i++)
                {
                    var iii = i * dim0;
                    var ig = ingrad[i];
                    // Loop repeat size
                    for (int di = 0; di < dim0; di++)
                        BckSub(grad[iii + di], ig, len1, len2, len3, dim1, dim2, dim3);
                }
            }
        }

        /// <summary>
        /// Splits values on a specific dimention to multiple values.
        /// Inputs: float[1d - 4d]
        /// </summary>
        public static class Split
        {
            /// <summary>
            /// Splits values on a specific dimention to multiple values.
            /// </summary>
            /// <param name="x">Series to split</param>
            /// <param name="dim">Dimention index to split on</param>
            /// <param name="lengths">The length for each splitted value</param>
            /// <returns>Splitted values on certain dimention</returns>
            public static NodeArray<float[][], float[][]> Fwd(Tensor<float[][]> x, int dim, params int[] lengths)
            {
                object ctx;
                AutoGrad.BckF bckF;
                float[][][] y;
                // dim0
                if (dim == 0)
                {
                    // Forwards pass
                    y = Fwd_dim0(x._data, out ctx, lengths);
                    // Backwards pass
                    bckF = Bck_dim0_2d;
                }
                // dim1
                else if (dim == 1)
                {
                    // Forwards pass
                    y = Fwd_dim1(x._data, out ctx, lengths);
                    // Backwards pass
                    bckF = Bck_dim1_2d;
                }
                else
                    throw new Exception("dim < 0 || dim > 1");
                // Branch from one node to multiple nodes
                return x.AddBranch(ctx, bckF, false, y);
            }
            /// <summary>
            /// Splits values on a specific dimention to multiple values.
            /// </summary>
            /// <param name="x">Series to split</param>
            /// <param name="dim">Dimention index to split on</param>
            /// <param name="lengths">The length for each splitted value</param>
            /// <returns>Splitted values on certain dimention</returns>
            public static NodeArray<float[][][], float[][][]> Fwd(Tensor<float[][][]> x, int dim, params int[] lengths)
            {
                object ctx;
                float[][][][] y;
                AutoGrad.BckF bckF;
                // dim1
                if (dim == 1)
                {
                    // Forwards pass
                    y = Fwd_dim1(x._data, out ctx, lengths);
                    // Backwards pass
                    bckF = Bck_dim1_3d;
                }
                // dim2
                else if (dim == 2)
                {
                    // Forwards pass
                    y = Fwd_dim2(x._data, out ctx, lengths);
                    // Backwards pass
                    bckF = Bck_dim2_3d;
                }
                // dim0
                else if (dim == 0)
                {
                    // Forwards pass
                    y = Fwd_dim0(x._data, out ctx, lengths);
                    // Backwards pass
                    bckF = Bck_dim0_3d;
                }
                else
                    throw new Exception("dim < 0 || dim > 2");
                // Branch from one node to multiple nodes
                return x.AddBranch(ctx, bckF, false, y);
            }
            /// <summary>
            /// Splits values on a specific dimention to multiple values.
            /// </summary>
            /// <param name="x">Series to split</param>
            /// <param name="dim">Dimention index to split on</param>
            /// <param name="lengths">The length for each splitted value</param>
            /// <returns>Splitted values on certain dimention</returns>
            public static NodeArray<float[][][][], float[][][][]> Fwd(Tensor<float[][][][]> x, int dim, params int[] lengths)
            {
                object ctx;
                float[][][][][] y;
                AutoGrad.BckF bckF;
                // dim1
                if (dim == 1)
                {
                    // Forwards pass
                    y = Fwd_dim1(x._data, out ctx, lengths);
                    // Backwards pass
                    bckF = Bck_dim1_4d;
                }
                // dim2
                else if (dim == 2)
                {
                    // Forwards pass
                    y = Fwd_dim2(x._data, out ctx, lengths);
                    // Backwards pass
                    bckF = Bck_dim2_4d;
                }
                // dim0
                else if (dim == 0)
                {
                    // Forwards pass
                    y = Fwd_dim0(x._data, out ctx, lengths);
                    // Backwards pass
                    bckF = Bck_dim0_4d;
                }
                // dim3
                else if (dim == 3)
                    throw new NotSupportedException("dim == 3");
                else
                    throw new Exception("dim < 0 || dim > 3");
                // Branch from one node to multiple nodes
                return x.AddBranch(ctx, bckF, false, y);
            }

            /// <summary>
            /// Splits values on a specific dimention to multiple values.
            /// </summary>
            /// <param name="values">Series to split</param>
            /// <param name="dim">Dimention index to split on</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="lengths">The length for each splitted value</param>
            /// <returns>Splitted values on certain dimention</returns>
            public static T[][][] Fwd<T>(T[][] values, int dim, out object ctx, params int[] lengths)
            {
                // dim1
                if (dim == 1)
                    return Fwd_dim1(values, out ctx, lengths);
                // dim0
                else if (dim == 0)
                    return Fwd_dim0(values, out ctx, lengths);
                throw new Exception("dim < 0 || dim > 1");
            }
            /// <summary>
            /// Splits values on a specific dimention to multiple values.
            /// </summary>
            /// <param name="values">Series to split</param>
            /// <param name="dim">Dimention index to split on</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="lengths">The length for each splitted value</param>
            /// <returns>Splitted values on certain dimention</returns>
            public static T[][][][] Fwd<T>(T[][][] values, int dim, out object ctx, params int[] lengths)
            {
                // dim1
                if (dim == 1)
                    return Fwd_dim1(values, out ctx, lengths);
                // dim0
                else if (dim == 0)
                    return Fwd_dim0(values, out ctx, lengths);
                // dim2
                else if (dim == 2)
                    return Fwd_dim2(values, out ctx, lengths);
                throw new Exception("dim < 0 || dim > 2");
            }

            /// <summary>
            /// Splits values on dimention index (0) to multiple values.
            /// </summary>
            /// <param name="values">Series to split</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="lengths">The length for each splitted value</param>
            /// <returns>Splitted values on certain dimention</returns
            public static T[][][] Fwd_dim0<T>(T[][] values, out object ctx, params int[] lengths)
            {
                // Validate
                if (lengths.Length < 2)
                    throw new Exception($"lengths.Length({lengths.Length}) < 2");
                // Make sure all the split lengths combined equal the dim length
                var len = lengths.Sum();
                if (len != values.Length)
                    throw new Exception($"The sum of the lengths is ({len}). It should equal " +
                            $"values at dim 0 ({values.Length}).");
                // Forward Pass
                var y = Fwd_sub(values, lengths);
                // Context
                ctx = new[] { 0, y.Length, y[0].Length, y[0][0].Length }
                        .Concat(lengths).ToArray();
                return y;
            }
            /// <summary>
            /// Splits values on dimention index (1) to multiple values.
            /// </summary>
            /// <param name="values">Series to split</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="lengths">The length for each splitted value</param>
            /// <returns>Splitted values on certain dimention</returns
            public static T[][][] Fwd_dim1<T>(T[][] values, out object ctx, params int[] lengths)
            {
                // Validate
                if (lengths.Length < 2)
                    throw new Exception($"lengths.Length({lengths.Length}) < 2");
                // Make sure all the split lengths combined equal the dim length
                var len = lengths.Sum();
                if (len != values[0].Length)
                    throw new Exception($"The sum of the lengths is ({len}). It should equal " +
                            $"values at dim 1 ({values[0].Length}).");
                // Forward Pass
                var y = lengths.Select(ln => new T[values.Length][]).ToArray();
                // Loop batch
                for (int b = 0; b < values.Length; b++)
                {
                    var o = Fwd_sub(values[b], lengths);
                    // y first dimention is the number of splits
                    for (int i = 0; i < lengths.Length; i++)
                        y[i][b] = o[i];
                }
                // Context
                ctx = new[] { 1, y.Length, y[0].Length, y[0][0].Length }
                        .Concat(lengths).ToArray();
                return y;
            }
            /// <summary>
            /// Splits values on dimention index (2) to multiple values.
            /// </summary>
            /// <param name="values">Series to split</param>
            /// <param name="ctx">Context for backwards pass</param>
            /// <param name="lengths">The length for each splitted value</param>
            /// <returns>Splitted values on certain dimention</returns
            public static T[][][][] Fwd_dim2<T>(T[][][] values, out object ctx, params int[] lengths)
            {
                // Validate
                if (lengths.Length < 2)
                    throw new Exception($"lengths.Length({lengths.Length}) < 2");
                // Make sure all the split lengths combined equal the dim length
                var len = lengths.Sum();
                if (len != values[0][0].Length)
                    throw new Exception($"The sum of the lengths is ({len}). It should equal " +
                            $"values at dim 1 ({values[0][0].Length}).");
                var dim1 = values[0].Length;
                // Forward Pass
                var y = lengths.Select(ln => new bool[values.Length].Select(no => new T[dim1][]).ToArray()).ToArray();
                // Loop batch
                for (int b = 0; b < values.Length; b++)
                {
                    var vb = values[b];
                    // Loop dim1
                    for (int j = 0; j < dim1; j++)
                    {
                        var o = Fwd_sub(vb[j], lengths);
                        // y first dimention is the number of splits
                        for (int i = 0; i < lengths.Length; i++)
                            y[i][b][j] = o[i];
                    }
                }
                // Context
                ctx = new[] { 2, y.Length, y[0].Length, y[0][0].Length, y[0][0][0].Length }
                        .Concat(lengths).ToArray();
                return y;
            }

            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <typeparam name="T">Type of 'grad'</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Backwards Pass Context</param>
            /// <returns>Input Gradient</returns>
            public static T[][] Bck<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 6)
                    throw new Exception($"ctx.Length({ctx.Length}) < 6");
                var dim = ctx[0];
                // dim1
                if (dim == 1)
                    return Bck_dim1(grad, ctx_);
                // dim0
                else if (dim == 0)
                    return Bck_dim0(grad, ctx_);
                throw new Exception("Unknown Error: dim < 0 || dim > 1");
            }
            /// <summary>
            /// Inside Backwards Function
            /// </summary>
            /// <typeparam name="T">Type of 'grad'</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Backwards Pass Context</param>
            /// <returns>Input Gradient</returns>
            public static T[][][] Bck<T>(T[][][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 6)
                    throw new Exception($"ctx.Length({ctx.Length}) < 6");
                var dim = ctx[0];
                // dim1
                if (dim == 1)
                    return Bck_dim1(grad, ctx_);
                // dim0
                else if (dim == 0)
                    return Bck_dim0(grad, ctx_);
                // dim2
                else if (dim == 2)
                    return Bck_dim2(grad, ctx_);
                throw new Exception("Unknown Error: dim < 0 || dim > 2");
            }

            /// <summary>
            /// Inside Backwards Tensor Function
            /// </summary>
            /// <param name="y_">Output node of Forwards pass</param>
            static void Bck_dim0_2d(AutoGrad.BackwardNode y_)
            {
                var y = (NodeArray<float[][], float[][]>)y_;
                // Output Gradient
                var outgrad = y._nodes.Select(n => (float[][])n._grad).ToArray();
                // If requiresGrad=True make sure it has a gradient
                y._grad = outgrad;
                // Backwards Pass
                var ingrad = Bck_dim0(outgrad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Tensor Function
            /// </summary>
            /// <param name="y_">Output node of Forwards pass</param>
            static void Bck_dim0_3d(AutoGrad.BackwardNode y_)
            {
                var y = (NodeArray<float[][][], float[][][]>)y_;
                // Output Gradient
                var outgrad = y._nodes.Select(n => (float[][][])n._grad).ToArray();
                // If requiresGrad=True make sure it has a gradient
                y._grad = outgrad;
                // Backwards Pass
                var ingrad = Bck_dim0(outgrad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Tensor Function
            /// </summary>
            /// <param name="y_">Output node of Forwards pass</param>
            static void Bck_dim0_4d(AutoGrad.BackwardNode y_)
            {
                var y = (NodeArray<float[][][][], float[][][][]>)y_;
                // Output Gradient
                var outgrad = y._nodes.Select(n => (float[][][][])n._grad).ToArray();
                // If requiresGrad=True make sure it has a gradient
                y._grad = outgrad;
                // Backwards Pass
                var ingrad = Bck_dim0(outgrad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Tensor Function
            /// </summary>
            /// <param name="y_">Output node of Forwards pass</param>
            static void Bck_dim1_2d(AutoGrad.BackwardNode y_)
            {
                var y = (NodeArray<float[][], float[][]>)y_;
                // Output Gradient
                var outgrad = y._nodes.Select(n => (float[][])n._grad).ToArray();
                // If requiresGrad=True make sure it has a gradient
                y._grad = outgrad;
                // Backwards Pass
                var ingrad = Bck_dim1(outgrad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Tensor Function
            /// </summary>
            /// <param name="y_">Output node of Forwards pass</param>
            static void Bck_dim1_3d(AutoGrad.BackwardNode y_)
            {
                var y = (NodeArray<float[][][], float[][][]>)y_;
                // Output Gradient
                var outgrad = y._nodes.Select(n => (float[][][])n._grad).ToArray();
                // If requiresGrad=True make sure it has a gradient
                y._grad = outgrad;
                // Backwards Pass
                var ingrad = Bck_dim1(outgrad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Tensor Function
            /// </summary>
            /// <param name="y_">Output node of Forwards pass</param>
            static void Bck_dim1_4d(AutoGrad.BackwardNode y_)
            {
                var y = (NodeArray<float[][][][], float[][][][]>)y_;
                // Output Gradient
                var outgrad = y._nodes.Select(n => (float[][][][])n._grad).ToArray();
                // If requiresGrad=True make sure it has a gradient
                y._grad = outgrad;
                // Backwards Pass
                var ingrad = Bck_dim1(outgrad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Tensor Function
            /// </summary>
            /// <param name="y_">Output node of Forwards pass</param>
            static void Bck_dim2_3d(AutoGrad.BackwardNode y_)
            {
                var y = (NodeArray<float[][][], float[][][]>)y_;
                // Output Gradient
                var outgrad = y._nodes.Select(n => (float[][][])n._grad).ToArray();
                // If requiresGrad=True make sure it has a gradient
                y._grad = outgrad;
                // Backwards Pass
                var ingrad = Bck_dim2(outgrad, y._ctx);
                y.GradSum(ingrad);
            }
            /// <summary>
            /// Inside Backwards Tensor Function
            /// </summary>
            /// <param name="y_">Output node of Forwards pass</param>
            static void Bck_dim2_4d(AutoGrad.BackwardNode y_)
            {
                var y = (NodeArray<float[][][][], float[][][][]>)y_;
                // Output Gradient
                var outgrad = y._nodes.Select(n => (float[][][][])n._grad).ToArray();
                // If requiresGrad=True make sure it has a gradient
                y._grad = outgrad;
                // Backwards Pass
                var ingrad = Bck_dim2(outgrad, y._ctx);
                y.GradSum(ingrad);
            }
            
            /// <summary>
            /// Inside Backwards Function on dimention index (0)
            /// </summary>
            /// <typeparam name="T">Type of 'grad'</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Input Gradient</returns>
            public static T[][] Bck_dim0<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 6)
                    throw new Exception($"ctx.Length({ctx.Length}) < 6");
                var nodelen = ctx.Length - 4;
                var dim2 = ctx[3];
                // Make sure it has the correct backwards function from forwards pass
                if (ctx[0] != 0)
                    throw new Exception($"ctx[0]({ctx[0]}) != 0: you should be using Bck_dim{ctx[0]}() instead.");
                // Validate Node Length
                if (nodelen != ctx[1])
                    throw new Exception($"nodelen({nodelen}) != ctx[1]({ctx[1]})");
                // Validate Node Length in gradient
                if (grad.Length != nodelen)
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, dim, {grad[0][0].Length}) " +
                            $"it should be ({nodelen} ...). Wrong number of nodes.");
                // Validate Shape for each branch
                for (int i = 0; i < nodelen; i++)
                {
                    var g = grad[i];
                    if (g.Length != ctx[i + 4] || g[0].Length != dim2)
                        throw new Exception($"Invalid gradient at node index ({i}): got ({nodelen}, {g.Length}, {g[0].Length}) " +
                                $"it should be ({nodelen}, {ctx[i + 4]}, {dim2}");
                }
                // Backwards Pass
                return grad.SelectMany(g => g).ToArray();
            }
            /// <summary>
            /// Inside Backwards Function on dimention index (1)
            /// </summary>
            /// <typeparam name="T">Type of 'grad'</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Input Gradient</returns>
            public static T[][] Bck_dim1<T>(T[][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 6)
                    throw new Exception($"ctx.Length({ctx.Length}) < 6");
                var nodelen = ctx.Length - 4;
                var dim1 = ctx[2];
                // Make sure it has the correct backwards function from forwards pass
                if (ctx[0] != 1)
                    throw new Exception($"ctx[0]({ctx[0]}) != 1: you should be using Bck_dim{ctx[0]}() instead.");
                // Validate Node Length
                if (nodelen != ctx[1])
                    throw new Exception($"nodelen({nodelen}) != ctx[1]({ctx[1]})");
                // Validate Node Length in gradient
                if (grad.Length != nodelen)
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, dim) " +
                            $"it should be ({nodelen} ...). Wrong number of nodes.");
                // Validate Shape for each branch
                for (int i = 0; i < nodelen; i++)
                {
                    var g = grad[i];
                    if (g.Length != dim1 || g[0].Length != ctx[i + 4])
                        throw new Exception($"Invalid gradient at node index ({i}): got ({nodelen}, {g.Length}, {g[0].Length}) " +
                                $"it should be ({nodelen}, {dim1}, {ctx[i + 4]}");
                }
                // Backwards Pass
                return Enumerable.Range(0, dim1).Select(b =>
                        grad.SelectMany(g => g[b]).ToArray()).ToArray();
            }
            /// <summary>
            /// Inside Backwards Function on dimention index (2)
            /// </summary>
            /// <typeparam name="T">Type of 'grad'</typeparam>
            /// <param name="grad">Output Gradient</param>
            /// <param name="ctx_">Context for backwards pass</param>
            /// <returns>Input Gradient</returns>
            public static T[][][] Bck_dim2<T>(T[][][][] grad, object ctx_)
            {
                var ctx = (int[])ctx_;
                // Validate Context size
                if (ctx.Length < 7)
                    throw new Exception($"ctx.Length({ctx.Length}) < 7");
                var nodelen = ctx.Length - 5;
                var dim1 = ctx[2];
                var dim2 = ctx[3];
                // Make sure it has the correct backwards function from forwards pass
                if (ctx[0] != 2)
                    throw new Exception($"ctx[0]({ctx[0]}) != 2: you should be using Bck_dim{ctx[0]}() instead.");
                // Validate Node Length
                if (nodelen != ctx[1])
                    throw new Exception($"nodelen({nodelen}) != ctx[1]({ctx[1]})");
                // Validate Node Length in gradient
                if (grad.Length != nodelen)
                    throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}, dim) " +
                            $"it should be ({nodelen} ...). Wrong number of nodes.");
                // Validate Shape for each branch
                for (int i = 0; i < nodelen; i++)
                {
                    var g = grad[i];
                    if (g.Length != dim1  || g[0].Length != dim2 || g[0][0].Length != ctx[i + 5])
                        throw new Exception($"Invalid gradient at node index ({i}): got ({nodelen}, {g.Length}, {g[0].Length}, {g[0][0].Length}) " +
                                $"it should be ({nodelen}, {dim1}, {dim2}, {ctx[i + 5]}");
                }
                // Backwards Pass
                return Enumerable.Range(0, dim1).Select(b =>
                        Enumerable.Range(0, dim2).Select(i =>
                        grad.SelectMany(g => g[b][i]).ToArray()).ToArray()).ToArray();
            }

            /// <summary>
            /// Actual Forwards Pass
            /// </summary>
            /// <typeparam name="T">Type of 'values'</typeparam>
            /// <param name="values">Series to Split</param>
            /// <param name="lengths">Length of each one to split</param>
            /// <returns>Shape[nodes length][lengths[i]]</returns>
            static T[][] Fwd_sub<T>(T[] values, int[] lengths)
            {
                var start = 0;
                // Number of nodes
                var outs = new T[lengths.Length][];
                // Loop each node
                for (int i = 0; i < lengths.Length; i++)
                {
                    var ln = lengths[i];
                    var end = start + ln;
                    var o = outs[i] = new T[ln];
                    // Nodes values
                    for (int j = start; j < end; j++)
                        o[j - start] = values[j];
                    // Shift to next node
                    start += ln;
                }
                return outs;
            }
        }
    }
}
