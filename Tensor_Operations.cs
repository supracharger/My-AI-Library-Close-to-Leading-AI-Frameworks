using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    // Rediculus Amout of operations for Tensor!
    public partial class Tensor<T>
    {
        // Sum Operations ..............................................
        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
        {
            if (left is Tensor<float[]> lf)
            {
                var rf = (Tensor<float[]>)(object)right;
                var values = Sum(SameShape(lf._data, rf._data), rf._data);
                var l2 = lf.AddNode(values, null, Transfer1, false);
                var r2 = rf.AddNode(values, null, Transfer1, false);
                return (Tensor<T>)(object)Tensor<float[]>.AddConsolidate(
                        values, null, BackwardConsolidated1, false, l2, r2);
            }
            else if (left is Tensor<float[][]> l2d)
            {
                var r2d = (Tensor<float[][]>)(object)right;
                var values = Sum(SameShape(l2d._data, r2d._data), r2d._data);
                var l2 = l2d.AddNode(values, null, Transfer2, false);
                var r2 = r2d.AddNode(values, null, Transfer2, false);
                return (Tensor<T>)(object)Tensor<float[][]>.AddConsolidate(
                        values, null, BackwardConsolidated2, false, l2, r2);
            }
            else if (left is Tensor<float[][][]> l3d)
            {
                var r3d = (Tensor<float[][][]>)(object)right;
                var values = Sum(SameShape(l3d._data, r3d._data), r3d._data);
                var l2 = l3d.AddNode(values, null, Transfer3, false);
                var r2 = r3d.AddNode(values, null, Transfer3, false);
                return (Tensor<T>)(object)Tensor<float[][][]>.AddConsolidate(
                        values, null, BackwardConsolidated3, false, l2, r2);
            }
            else if (left is Tensor<float[][][][]> l4d)
            {
                var r4d = (Tensor<float[][][][]>)(object)right;
                var values = Sum(SameShape(l4d._data, r4d._data), r4d._data);
                var l2 = l4d.AddNode(values, null, Transfer4, false);
                var r2 = r4d.AddNode(values, null, Transfer4, false);
                return (Tensor<T>)(object)Tensor<float[][][][]>.AddConsolidate(
                        values, null, BackwardConsolidated4, false, l2, r2);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator +(Tensor<T> left, double n_)
        {
            var n = (float)n_;
            if (left is Tensor<float[]> f1)
            {
                var values = Sum(f1._data, n);
                return (Tensor<T>)(object)f1.AddNode(values, null, Transfer1, false);
            }
            else if (left is Tensor<float[][]> f2)
            {
                var values = Sum(f2._data, n);
                return (Tensor<T>)(object)f2.AddNode(values, null, Transfer2, false);
            }
            else if (left is Tensor<float[][][]> f3)
            {
                var values = Sum(f3._data, n);
                return (Tensor<T>)(object)f3.AddNode(values, null, Transfer3, false);
            }
            else if (left is Tensor<float[][][][]> f4)
            {
                var values = Sum(f4._data, n);
                return (Tensor<T>)(object)f4.AddNode(values, null, Transfer4, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator +(double n, Tensor<T> right)
            => right + n;
        public static Tensor<T> operator +(Tensor<T> left, float[] right)
        {
            if (left is Tensor<float[]> f)
            {
                SameShape(f._data, right);
                var y = Sum(f._data, right);
                return (Tensor<T>)(object)f.AddNode(y, null, Transfer1, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator +(Tensor<T> left, float[][] right)
        {
            if (left is Tensor<float[][]> f)
            {
                SameShape(f._data, right);
                var y = Sum(f._data, right);
                return (Tensor<T>)(object)f.AddNode(y, null, Transfer2, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator +(Tensor<T> left, float[][][] right)
        {
            if (left is Tensor<float[][][]> f)
            {
                SameShape(f._data, right);
                var y = Sum(f._data, right);
                return (Tensor<T>)(object)f.AddNode(y, null, Transfer3, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator +(Tensor<T> left, float[][][][] right)
        {
            if (left is Tensor<float[][][][]> f)
            {
                SameShape(f._data, right);
                var y = Sum(f._data, right);
                return (Tensor<T>)(object)f.AddNode(y, null, Transfer4, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator +(float[] left, Tensor<T> right)
            => right + left;
        public static Tensor<T> operator +(float[][] left, Tensor<T> right)
            => right + left;
        public static Tensor<T> operator +(float[][][] left, Tensor<T> right)
            => right + left;
        public static Tensor<T> operator +(float[][][][] left, Tensor<T> right)
            => right + left;

        // Subtract Operations ..............................................
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
        {
            if (left is Tensor<float[]> lf1)
            {
                var r = (Tensor<float[]>)(object)right;
                var values = Subtract(SameShape(lf1._data, r._data), r._data);
                AutoGrad.BckF BackFuncRight = y => y.GradSum(Multiply((float[])y._grad, -1));
                var l2 = lf1.AddNode(values, null, Transfer1, false);
                var r2 = r.AddNode(values, null, BackFuncRight, false);
                return (Tensor<T>)(object)Tensor<float[]>.AddConsolidate(values, null, BackwardConsolidated1, false, l2, r2);
            }
            else if (left is Tensor<float[][]> lf2)
            {
                var r = (Tensor<float[][]>)(object)right;
                var values = Subtract(SameShape(lf2._data, r._data), r._data);
                AutoGrad.BckF BackFuncRight = y => y.GradSum(Multiply((float[][])y._grad, -1));
                var l2 = lf2.AddNode(values, null, Transfer2, false);
                var r2 = r.AddNode(values, null, BackFuncRight, false);
                return (Tensor<T>)(object)Tensor<float[][]>.AddConsolidate(values, null, BackwardConsolidated2, false, l2, r2);
            }
            else if (left is Tensor<float[][][]> lf3)
            {
                var r = (Tensor<float[][][]>)(object)right;
                var values = Subtract(SameShape(lf3._data, r._data), r._data);
                AutoGrad.BckF BackFuncRight = y => y.GradSum(Multiply((float[][][])y._grad, -1));
                var l2 = lf3.AddNode(values, null, Transfer3, false);
                var r2 = r.AddNode(values, null, BackFuncRight, false);
                return (Tensor<T>)(object)Tensor<float[][][]>.AddConsolidate(values, null, BackwardConsolidated3, false, l2, r2);
            }
            else if (left is Tensor<float[][][][]> lf4)
            {
                var r = (Tensor<float[][][][]>)(object)right;
                var values = Subtract(SameShape(lf4._data, r._data), r._data);
                AutoGrad.BckF BackFuncRight = y => y.GradSum(Multiply((float[][][][])y._grad, -1));
                var l2 = lf4.AddNode(values, null, Transfer4, false);
                var r2 = r.AddNode(values, null, BackFuncRight, false);
                return (Tensor<T>)(object)Tensor<float[][][][]>.AddConsolidate(values, null, BackwardConsolidated4, false, l2, r2);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator -(Tensor<T> left, double n_)
        {
            var n = (float)n_;
            if (left is Tensor<float[]> f1)
            {
                var values = Subtract(f1._data, n);
                return (Tensor<T>)(object)f1.AddNode(values, null, Transfer1, false);
            }
            else if (left is Tensor<float[][]> f2)
            {
                var values = Subtract(f2._data, n);
                return (Tensor<T>)(object)f2.AddNode(values, null, Transfer2, false);
            }
            else if (left is Tensor<float[][][]> f3)
            {
                var values = Subtract(f3._data, n);
                return (Tensor<T>)(object)f3.AddNode(values, null, Transfer3, false);
            }
            else if (left is Tensor<float[][][][]> f4)
            {
                var values = Subtract(f4._data, n);
                return (Tensor<T>)(object)f4.AddNode(values, null, Transfer4, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator -(double left_, Tensor<T> right)
        {
            var left = (float)left_;
            if (right is Tensor<float[]> f1)
            {
                var values = Subtract(left, f1._data);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[])y._grad, y), -1));
                return (Tensor<T>)(object)f1.AddNode(values, null, BackFunc, false);
            }
            else if (right is Tensor<float[][]> f2)
            {
                var values = Subtract(left, f2._data);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][])y._grad, y), -1));
                return (Tensor<T>)(object)f2.AddNode(values, null, BackFunc, false);
            }
            else if (right is Tensor<float[][][]> f3)
            {
                var values = Subtract(left, f3._data);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][][])y._grad, y), -1));
                return (Tensor<T>)(object)f3.AddNode(values, null, BackFunc, false);
            }
            else if (right is Tensor<float[][][][]> f4)
            {
                var values = Subtract(left, f4._data);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][][][])y._grad, y), -1));
                return (Tensor<T>)(object)f4.AddNode(values, null, BackFunc, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator -(Tensor<T> left, float[] right)
        {
            if (left is Tensor<float[]> f)
            {
                SameShape(f._data, right);
                var y = Subtract(f._data, right);
                return (Tensor<T>)(object)f.AddNode(y, null, Transfer1, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator -(Tensor<T> left, float[][] right)
        {
            if (left is Tensor<float[][]> f)
            {
                SameShape(f._data, right);
                var y = Subtract(f._data, right);
                return (Tensor<T>)(object)f.AddNode(y, null, Transfer2, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator -(Tensor<T> left, float[][][] right)
        {
            if (left is Tensor<float[][][]> f)
            {
                SameShape(f._data, right);
                var y = Subtract(f._data, right);
                return (Tensor<T>)(object)f.AddNode(y, null, Transfer3, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator -(Tensor<T> left, float[][][][] right)
        {
            if (left is Tensor<float[][][][]> f)
            {
                SameShape(f._data, right);
                var y = Subtract(f._data, right);
                return (Tensor<T>)(object)f.AddNode(y, null, Transfer4, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator -(float[] left, Tensor<T> right)
        {
            if (right is Tensor<float[]> f)
            {
                SameShape(left, f._data);
                var y = Subtract(left, f._data);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Multiply(SameShape((float[])yy._grad, y), -1));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator -(float[][] left, Tensor<T> right)
        {
            if (right is Tensor<float[][]> f)
            {
                SameShape(left, f._data);
                var y = Subtract(left, f._data);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Multiply(SameShape((float[][])yy._grad, y), -1));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator -(float[][][] left, Tensor<T> right)
        {
            if (right is Tensor<float[][][]> f)
            {
                SameShape(left, f._data);
                var y = Subtract(left, f._data);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Multiply(SameShape((float[][][])yy._grad, y), -1));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator -(float[][][][] left, Tensor<T> right)
        {
            if (right is Tensor<float[][][][]> f)
            {
                SameShape(left, f._data);
                var y = Subtract(left, f._data);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Multiply(SameShape((float[][][][])yy._grad, y), -1));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }

        // Multiply Operations ..............................................
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
        {
            if (left is Tensor<float[]> l1)
            {
                var r = (Tensor<float[]>)(object)right;
                var leftval = l1._data;
                var rightval = r._data;
                SameShape(leftval, rightval);
                var values = Multiply(leftval, rightval);
                AutoGrad.BckF BackFuncleft = y => y.GradSum(Multiply((float[])y._grad, rightval));
                AutoGrad.BckF BackFuncright = y => y.GradSum(Multiply(leftval, (float[])y._grad));
                var l2 = l1.AddNode(values, null, BackFuncleft, false);
                var r2 = r.AddNode(values, null, BackFuncright, false);
                return (Tensor<T>)(object)Tensor<float[]>.AddConsolidate(values, null, BackwardConsolidated1, false, l2, r2);
            }
            else if (left is Tensor<float[][]> lf2)
            {
                var r = (Tensor<float[][]>)(object)right;
                var leftval = lf2._data;
                var rightval = r._data;
                SameShape(leftval, rightval);
                var values = Multiply(leftval, rightval);
                AutoGrad.BckF BackFuncleft = y => y.GradSum(Multiply((float[][])y._grad, rightval));
                AutoGrad.BckF BackFuncright = y => y.GradSum(Multiply(leftval, (float[][])y._grad));
                var l2 = lf2.AddNode(values, null, BackFuncleft, false);
                var r2 = r.AddNode(values, null, BackFuncright, false);
                return (Tensor<T>)(object)Tensor<float[][]>.AddConsolidate(values, null, BackwardConsolidated2, false, l2, r2);
            }
            else if (left is Tensor<float[][][]> l3)
            {
                var r = (Tensor<float[][][]>)(object)right;
                var leftval = l3._data;
                var rightval = r._data;
                SameShape(leftval, rightval);
                var values = Multiply(leftval, rightval);
                AutoGrad.BckF BackFuncleft = y => y.GradSum(Multiply((float[][][])y._grad, rightval));
                AutoGrad.BckF BackFuncright = y => y.GradSum(Multiply(leftval, (float[][][])y._grad));
                var l2 = l3.AddNode(values, null, BackFuncleft, false);
                var r2 = r.AddNode(values, null, BackFuncright, false);
                return (Tensor<T>)(object)Tensor<float[][][]>.AddConsolidate(values, null, BackwardConsolidated3, false, l2, r2);
            }
            else if (left is Tensor<float[][][][]> l4)
            {
                var r = (Tensor<float[][][][]>)(object)right;
                var leftval = l4._data;
                var rightval = r._data;
                SameShape(leftval, rightval);
                var values = Multiply(leftval, rightval);
                AutoGrad.BckF BackFuncleft = y => y.GradSum(Multiply((float[][][][])y._grad, rightval));
                AutoGrad.BckF BackFuncright = y => y.GradSum(Multiply(leftval, (float[][][][])y._grad));
                var l2 = l4.AddNode(values, null, BackFuncleft, false);
                var r2 = r.AddNode(values, null, BackFuncright, false);
                return (Tensor<T>)(object)Tensor<float[][][][]>.AddConsolidate(values, null, BackwardConsolidated4, false, l2, r2);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator *(Tensor<T> left, double right_)
        {
            var right = (float)right_;
            if (left is Tensor<float[]> f1)
            {
                var values = Multiply(f1._data, right);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[])y._grad, y), right));
                return (Tensor<T>)(object)f1.AddNode(values, null, BackFunc, false);
            }
            else if (left is Tensor<float[][]> f2)
            {
                var values = Multiply(f2._data, right);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][])y._grad, y), right));
                return (Tensor<T>)(object)f2.AddNode(values, null, BackFunc, false);
            }
            else if (left is Tensor<float[][][]> f3)
            {
                var values = Multiply(f3._data, right);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][][])y._grad, y), right));
                return (Tensor<T>)(object)f3.AddNode(values, null, BackFunc, false);
            }
            else if (left is Tensor<float[][][][]> f4)
            {
                var values = Multiply(f4._data, right);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][][][])y._grad, y), right));
                return (Tensor<T>)(object)f4.AddNode(values, null, BackFunc, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator *(double left, Tensor<T> right)
            => right * left;
        public static Tensor<T> operator *(Tensor<T> left, float[] right)
        {
            if (left is Tensor<float[]> f)
            {
                SameShape(f._data, right);
                var y = Multiply(f._data, right);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Multiply(SameShape((float[])yy._grad, right), right));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator *(Tensor<T> left, float[][] right)
        {
            if (left is Tensor<float[][]> f)
            {
                SameShape(f._data, right);
                var y = Multiply(f._data, right);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Multiply(SameShape((float[][])yy._grad, right), right));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator *(Tensor<T> left, float[][][] right)
        {
            if (left is Tensor<float[][][]> f)
            {
                SameShape(f._data, right);
                var y = Multiply(f._data, right);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Multiply(SameShape((float[][][])yy._grad, right), right));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator *(Tensor<T> left, float[][][][] right)
        {
            if (left is Tensor<float[][][][]> f)
            {
                SameShape(f._data, right);
                var y = Multiply(f._data, right);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Multiply(SameShape((float[][][][])yy._grad, right), right));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator *(float[] left, Tensor<T> right)
            => right * left;
        public static Tensor<T> operator *(float[][] left, Tensor<T> right)
            => right * left;
        public static Tensor<T> operator *(float[][][] left, Tensor<T> right)
            => right * left;
        public static Tensor<T> operator *(float[][][][] left, Tensor<T> right)
            => right * left;

        // Divide Operations ..............................................
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right)
        {
            if (left is Tensor<float[]> l1)
            {
                var r = (Tensor<float[]>)(object)right;
                var leftval = l1._data;
                var rightval = r._data;
                SameShape(leftval, rightval);
                var values = Divide(leftval, rightval);
                AutoGrad.BckF BackFuncleft = y => y.GradSum(Divide((float[])y._grad, rightval));
                AutoGrad.BckF BackFuncright = y => y.GradSum(Divide(leftval, (float[])y._grad));
                var l2 = l1.AddNode(values, null, BackFuncleft, false);
                var r2 = r.AddNode(values, null, BackFuncright, false);
                return (Tensor<T>)(object)Tensor<float[]>.AddConsolidate(values, null, BackwardConsolidated1, false, l2, r2);
            }
            else if (left is Tensor<float[][]> lf2)
            {
                var r = (Tensor<float[][]>)(object)right;
                var leftval = lf2._data;
                var rightval = r._data;
                SameShape(leftval, rightval);
                var values = Divide(leftval, rightval);
                AutoGrad.BckF BackFuncleft = y => y.GradSum(Divide((float[][])y._grad, rightval));
                AutoGrad.BckF BackFuncright = y => y.GradSum(Divide(leftval, (float[][])y._grad));
                var l2 = lf2.AddNode(values, null, BackFuncleft, false);
                var r2 = r.AddNode(values, null, BackFuncright, false);
                return (Tensor<T>)(object)Tensor<float[][]>.AddConsolidate(values, null, BackwardConsolidated2, false, l2, r2);
            }
            else if (left is Tensor<float[][][]> l3)
            {
                var r = (Tensor<float[][][]>)(object)right;
                var leftval = l3._data;
                var rightval = r._data;
                SameShape(leftval, rightval);
                var values = Divide(leftval, rightval);
                AutoGrad.BckF BackFuncleft = y => y.GradSum(Divide((float[][][])y._grad, rightval));
                AutoGrad.BckF BackFuncright = y => y.GradSum(Divide(leftval, (float[][][])y._grad));
                var l2 = l3.AddNode(values, null, BackFuncleft, false);
                var r2 = r.AddNode(values, null, BackFuncright, false);
                return (Tensor<T>)(object)Tensor<float[][][]>.AddConsolidate(values, null, BackwardConsolidated3, false, l2, r2);
            }
            else if (left is Tensor<float[][][][]> l4)
            {
                var r = (Tensor<float[][][][]>)(object)right;
                var leftval = l4._data;
                var rightval = r._data;
                SameShape(leftval, rightval);
                var values = Divide(leftval, rightval);
                AutoGrad.BckF BackFuncleft = y => y.GradSum(Divide((float[][][][])y._grad, rightval));
                AutoGrad.BckF BackFuncright = y => y.GradSum(Divide(leftval, (float[][][][])y._grad));
                var l2 = l4.AddNode(values, null, BackFuncleft, false);
                var r2 = r.AddNode(values, null, BackFuncright, false);
                return (Tensor<T>)(object)Tensor<float[][][][]>.AddConsolidate(values, null, BackwardConsolidated4, false, l2, r2);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator /(Tensor<T> left, double right_)
        {
            var right = (float)right_;
            if (left is Tensor<float[]> f1)
            {
                var values = Divide(f1._data, right);
                AutoGrad.BckF BackFunc = y => y.GradSum(Divide(SameShape((float[])y._grad, y), right));
                return (Tensor<T>)(object)f1.AddNode(values, null, BackFunc, false);
            }
            else if (left is Tensor<float[][]> f2)
            {
                var values = Divide(f2._data, right);
                AutoGrad.BckF BackFunc = y => y.GradSum(Divide(SameShape((float[][])y._grad, y), right));
                return (Tensor<T>)(object)f2.AddNode(values, null, BackFunc, false);
            }
            else if (left is Tensor<float[][][]> f3)
            {
                var values = Divide(f3._data, right);
                AutoGrad.BckF BackFunc = y => y.GradSum(Divide(SameShape((float[][][])y._grad, y), right));
                return (Tensor<T>)(object)f3.AddNode(values, null, BackFunc, false);
            }
            else if (left is Tensor<float[][][][]> f4)
            {
                var values = Divide(f4._data, right);
                AutoGrad.BckF BackFunc = y => y.GradSum(Divide(SameShape((float[][][][])y._grad, y), right));
                return (Tensor<T>)(object)f4.AddNode(values, null, BackFunc, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator /(double left_, Tensor<T> right)
        {
            var left = (float)left_;
            if (right is Tensor<float[]> f1)
            {
                var values = Divide(left, f1._data);
                AutoGrad.BckF BackFunc = y => y.GradSum(Divide(left, SameShape((float[])y._grad, y)));
                return (Tensor<T>)(object)f1.AddNode(values, null, BackFunc, false);
            }
            else if (right is Tensor<float[][]> f2)
            {
                var values = Divide(left, f2._data);
                AutoGrad.BckF BackFunc = y => y.GradSum(Divide(left, SameShape((float[][])y._grad, y)));
                return (Tensor<T>)(object)f2.AddNode(values, null, BackFunc, false);
            }
            else if (right is Tensor<float[][][]> f3)
            {
                var values = Divide(left, f3._data);
                AutoGrad.BckF BackFunc = y => y.GradSum(Divide(left, SameShape((float[][][])y._grad, y)));
                return (Tensor<T>)(object)f3.AddNode(values, null, BackFunc, false);
            }
            else if (right is Tensor<float[][][][]> f4)
            {
                var values = Divide(left, f4._data);
                AutoGrad.BckF BackFunc = y => y.GradSum(Divide(left, SameShape((float[][][][])y._grad, y)));
                return (Tensor<T>)(object)f4.AddNode(values, null, BackFunc, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }
        public static Tensor<T> operator /(Tensor<T> left, float[] right)
        {
            if (left is Tensor<float[]> f)
            {
                SameShape(f._data, right);
                var y = Divide(f._data, right);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Divide((float[])yy._grad, right));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator /(Tensor<T> left, float[][] right)
        {
            if (left is Tensor<float[][]> f)
            {
                SameShape(f._data, right);
                var y = Divide(f._data, right);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Divide((float[][])yy._grad, right));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator /(Tensor<T> left, float[][][] right)
        {
            if (left is Tensor<float[][][]> f)
            {
                SameShape(f._data, right);
                var y = Divide(f._data, right);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Divide((float[][][])yy._grad, right));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator /(Tensor<T> left, float[][][][] right)
        {
            if (left is Tensor<float[][][][]> f)
            {
                SameShape(f._data, right);
                var y = Divide(f._data, right);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Divide((float[][][][])yy._grad, right));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator /(float[] left, Tensor<T> right)
        {
            if (right is Tensor<float[]> f)
            {
                SameShape(left, f._data);
                var y = Divide(left, f._data);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Divide(left, (float[])yy._grad));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator /(float[][] left, Tensor<T> right)
        {
            if (right is Tensor<float[][]> f)
            {
                SameShape(left, f._data);
                var y = Divide(left, f._data);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Divide(left, (float[][])yy._grad));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator /(float[][][] left, Tensor<T> right)
        {
            if (right is Tensor<float[][][]> f)
            {
                SameShape(left, f._data);
                var y = Divide(left, f._data);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Divide(left, (float[][][])yy._grad));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }
        public static Tensor<T> operator /(float[][][][] left, Tensor<T> right)
        {
            if (right is Tensor<float[][][][]> f)
            {
                SameShape(left, f._data);
                var y = Divide(left, f._data);
                AutoGrad.BckF BackFunc = yy => yy.GradSum(Divide(left, (float[][][][])yy._grad));
                return (Tensor<T>)(object)f.AddNode(y, null, BackFunc, false);
            }
            throw new NotSupportedException($"Tensor<{typeof(T)}> & {right.GetType()}");
        }

        /// <summary>
        /// Means all dimension in a single Tensor(float).
        /// </summary>
        /// <returns>Means all dimension in a single Tensor(float)</returns>
        public Tensor<float> Sum()
        {
            if (this is Tensor<float[]> f1)
            {
                var x = f1._data;
                var values = x.Sum();
                var N = x.Length;
                AutoGrad.BckF BackFunc = y =>
                {
                    var ingrad = Utilz.Fill(N, (float)y._grad);
                    y.GradSum(ingrad);
                };
                return f1.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][]> f2)
            {
                var x = f2._data;
                var values = x.Select(s => s.Sum()).Sum();
                var lens = new[] { x.Length, x[0].Length };
                AutoGrad.BckF BackFunc = y =>
                {
                    var ingrad = Utilz.Fill(lens[0], lens[1], (float)y._grad);
                    y.GradSum(ingrad);
                };
                return f2.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][][]> f3)
            {
                var x = f3._data;
                var values = x.Select(xi => xi.Select(s => s.Sum()).Sum()).Sum();
                var lens = new[] { x.Length, x[0].Length, x[0][0].Length };
                AutoGrad.BckF BackFunc = y =>
                {
                    var ingrad = Utilz.Fill(lens[0], lens[1], lens[2], (float)y._grad);
                    y.GradSum(ingrad);
                };
                return f3.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][][][]> f4)
            {
                var x = f4._data;
                var values = x.Select(xi => xi.Select(xj => xj.Select(s => s.Sum()).Sum()).Sum()).Sum();
                var lens = new[] { x.Length, x[0].Length, x[0][0].Length, x[0][0][0].Length };
                AutoGrad.BckF BackFunc = y =>
                {
                    var ingrad = Utilz.Fill(lens[0], lens[1], lens[2], lens[3], (float)y._grad);
                    y.GradSum(ingrad);
                };
                return f4.AddNode(values, null, BackFunc, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }

        /// <summary>
        /// Means all dimension in a single Tensor(float).
        /// </summary>
        /// <returns>Means all dimension in a single Tensor(float)</returns>
        public Tensor<float> Mean()
        {
            if (this is Tensor<float[]> f1)
            {
                var x = f1._data;
                var values = x.Average();
                var N = x.Length;
                AutoGrad.BckF BackFunc = y =>
                {
                    var grad = ((float)y._grad) / N;
                    var ingrad = Utilz.Fill(N, grad);
                    y.GradSum(ingrad);
                };
                return f1.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][]> f2)
            {
                var x = f2._data;
                var values = x.Select(s => s.Average()).Average();
                var lens = new[] { x.Length, x[0].Length };
                AutoGrad.BckF BackFunc = y =>
                {
                    var dim0 = lens[0];
                    var dim1 = lens[1];
                    var N = (long)dim0 * dim1;
                    var grad = ((float)y._grad) / N;
                    var ingrad = Utilz.Fill(dim0, dim1, grad);
                    y.GradSum(ingrad);
                };
                return f2.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][][]> f3)
            {
                var x = f3._data;
                var values = x.Select(xi => xi.Select(s => s.Average()).Average()).Average();
                var lens = new[] { x.Length, x[0].Length, x[0][0].Length };
                AutoGrad.BckF BackFunc = y =>
                {
                    var dim0 = lens[0];
                    var dim1 = lens[1];
                    var dim2 = lens[2];
                    var N = (long)dim0 * dim1 * dim2;
                    var grad = ((float)y._grad) / N;
                    var ingrad = Utilz.Fill(dim0, dim1, dim2, grad);
                    y.GradSum(ingrad);
                };
                return f3.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][][][]> f4)
            {
                var x = f4._data;
                var values = x.Select(xi => xi.Select(xj => xj.Select(s => s.Average()).Average()).Average()).Average();
                var lens = new[] { x.Length, x[0].Length, x[0][0].Length, x[0][0][0].Length };
                AutoGrad.BckF BackFunc = y =>
                {
                    var dim0 = lens[0];
                    var dim1 = lens[1];
                    var dim2 = lens[2];
                    var dim3 = lens[3];
                    var N = (long)dim0 * dim1 * dim2 * dim3;
                    var grad = ((float)y._grad) / N;
                    var ingrad = Utilz.Fill(dim0, dim1, dim2, dim3, grad);
                    y.GradSum(ingrad);
                };
                return f4.AddNode(values, null, BackFunc, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }

        /// <summary>
        /// Get the Absolute value of the Tensor.
        /// </summary>
        /// <returns>Absolute value of the Tensor</returns>
        public Tensor<T> Abs()
        {
            if (this is Tensor<float[]> f1)
            {
                var x = f1._data;
                var values = Abs(x);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[])y._grad, x), Sign(x)));
                return (Tensor<T>)(object)f1.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][]> f2)
            {
                var x = f2._data;
                var values = Abs(x);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][])y._grad, x), Sign(x)));
                return (Tensor<T>)(object)f2.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][][]> f3)
            {
                var x = f3._data;
                var values = Abs(x);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][][])y._grad, x), Sign(x)));
                return (Tensor<T>)(object)f3.AddNode(values, null, BackFunc, false);
            }
            else if (this is Tensor<float[][][][]> f4)
            {
                var x = f4._data;
                var values = Abs(x);
                AutoGrad.BckF BackFunc = y => y.GradSum(Multiply(SameShape((float[][][][])y._grad, x), Sign(x)));
                return (Tensor<T>)(object)f4.AddNode(values, null, BackFunc, false);
            }
            throw new NotSupportedException($"Not supported Type: {typeof(T)}.");
        }

        /// <summary>
        /// Transfer gradient to parent node.
        /// </summary>
        /// <param name="y">Output node of the forward pass</param>
        static void Transfer1(AutoGrad.BackwardNode y)
            => y.GradSum(SameShape((float[])y._grad, y));
        /// <summary>
        /// Transfer gradient to parent node.
        /// </summary>
        /// <param name="y">Output node of the forward pass</param>
        static void Transfer2(AutoGrad.BackwardNode y)
            => y.GradSum(SameShape((float[][])y._grad, y));
        /// <summary>
        /// Transfer gradient to parent node.
        /// </summary>
        /// <param name="y">Output node of the forward pass</param>
        static void Transfer3(AutoGrad.BackwardNode y)
            => y.GradSum(SameShape((float[][][])y._grad, y));
        /// <summary>
        /// Transfer gradient to parent node.
        /// </summary>
        /// <param name="y">Output node of the forward pass</param>
        static void Transfer4(AutoGrad.BackwardNode y)
            => y.GradSum(SameShape((float[][][][])y._grad, y));

        /// <summary>
        /// Transfer Gradient to consolidated nodes.
        /// </summary>
        /// <param name="y_">Output node of forward pass.</param>
        static void BackwardConsolidated1(AutoGrad.BackwardNode y_)
        {
            var y = (NodeArray<float[], float[]>)y_;
            var grad = SameShape((float[])y._grad, y);
            y.GradSum(y._nodes.Select(n => grad).ToArray());
        }
        /// <summary>
        /// Transfer Gradient to consolidated nodes.
        /// </summary>
        /// <param name="y_">Output node of forward pass.</param>
        static void BackwardConsolidated2(AutoGrad.BackwardNode y_)
        {
            var y = (NodeArray<float[][], float[][]>)y_;
            var grad = SameShape((float[][])y._grad, y);
            y.GradSum(y._nodes.Select(n => grad).ToArray());
        }
        /// <summary>
        /// Transfer Gradient to consolidated nodes.
        /// </summary>
        /// <param name="y_">Output node of forward pass.</param>
        static void BackwardConsolidated3(AutoGrad.BackwardNode y_)
        {
            var y = (NodeArray<float[][][], float[][][]>)y_;
            var grad = SameShape((float[][][])y._grad, y);
            y.GradSum(y._nodes.Select(n => grad).ToArray());
        }
        /// <summary>
        /// Transfer Gradient to consolidated nodes.
        /// </summary>
        /// <param name="y_">Output node of forward pass.</param>
        static void BackwardConsolidated4(AutoGrad.BackwardNode y_)
        {
            var y = (NodeArray<float[][][][], float[][][][]>)y_;
            var grad = SameShape((float[][][][])y._grad, y);
            y.GradSum(y._nodes.Select(n => grad).ToArray());
        }

        // Sum Operations ..............................................
        public static float[] Sum(float[] a, float[] b)
            => a.Zip(b, (x, y) => x + y).ToArray();
        public static float[][] Sum(float[][] a, float[][] b)
            => a.Zip(b, Sum).ToArray();
        public static float[][][] Sum(float[][][] a, float[][][] b)
            => a.Zip(b, Sum).ToArray();
        public static float[][][][] Sum(float[][][][] a, float[][][][] b)
            => a.Zip(b, Sum).ToArray();
        public static float[] Sum(float[] a, float b)
            => a.Select(v => v + b).ToArray();
        public static float[][] Sum(float[][] a, float b)
            => a.Select(v => Sum(v, b)).ToArray();
        public static float[][][] Sum(float[][][] a, float b)
            => a.Select(v => Sum(v, b)).ToArray();
        public static float[][][][] Sum(float[][][][] a, float b)
            => a.Select(v => Sum(v, b)).ToArray();

        // Subtract Operations ..............................................
        public static float[] Subtract(float[] a, float[] b)
            => a.Zip(b, (x, y) => x - y).ToArray();
        public static float[][] Subtract(float[][] a, float[][] b)
            => a.Zip(b, Subtract).ToArray();
        public static float[][][] Subtract(float[][][] a, float[][][] b)
            => a.Zip(b, Subtract).ToArray();
        public static float[][][][] Subtract(float[][][][] a, float[][][][] b)
            => a.Zip(b, Subtract).ToArray();
        public static float[] Subtract(float[] a, float b)
            => a.Select(aa => aa - b).ToArray();
        public static float[][] Subtract(float[][] a, float b)
            => a.Select(aa => Subtract(aa, b)).ToArray();
        public static float[][][] Subtract(float[][][] a, float b)
            => a.Select(aa => Subtract(aa, b)).ToArray();
        public static float[][][][] Subtract(float[][][][] a, float b)
            => a.Select(aa => Subtract(aa, b)).ToArray();
        public static float[] Subtract(float a, float[] b)
            => b.Select(bb => a - bb).ToArray();
        public static float[][] Subtract(float a, float[][] b)
            => b.Select(bb => Subtract(a, bb)).ToArray();
        public static float[][][] Subtract(float a, float[][][] b)
            => b.Select(bb => Subtract(a, bb)).ToArray();
        public static float[][][][] Subtract(float a, float[][][][] b)
            => b.Select(bb => Subtract(a, bb)).ToArray();

        // Multiply Operations ..............................................
        public static float[] Multiply(float[] a, float[] b)
            => a.Zip(b, (x, y) => x * y).ToArray();
        public static float[][] Multiply(float[][] a, float[][] b)
            => a.Zip(b, Multiply).ToArray();
        public static float[][][] Multiply(float[][][] a, float[][][] b)
            => a.Zip(b, Multiply).ToArray();
        public static float[][][][] Multiply(float[][][][] a, float[][][][] b)
            => a.Zip(b, Multiply).ToArray();
        public static float[] Multiply(float[] a, float b)
            => a.Select(aa => aa * b).ToArray();
        public static float[][] Multiply(float[][] a, float b)
            => a.Select(aa => Multiply(aa, b)).ToArray();
        public static float[][][] Multiply(float[][][] a, float b)
            => a.Select(aa => Multiply(aa, b)).ToArray();
        public static float[][][][] Multiply(float[][][][] a, float b)
            => a.Select(aa => Multiply(aa, b)).ToArray();

        // Divide Operations ..............................................
        public static float[] Divide(float[] a, float[] b)
            => a.Zip(b, (x, y) => x / y).ToArray();
        public static float[][] Divide(float[][] a, float[][] b)
            => a.Zip(b, Divide).ToArray();
        public static float[][][] Divide(float[][][] a, float[][][] b)
            => a.Zip(b, Divide).ToArray();
        public static float[][][][] Divide(float[][][][] a, float[][][][] b)
            => a.Zip(b, Divide).ToArray();
        public static float[] Divide(float[] a, float b)
            => a.Select(aa => aa / b).ToArray();
        public static float[][] Divide(float[][] a, float b)
            => a.Select(aa => Divide(aa, b)).ToArray();
        public static float[][][] Divide(float[][][] a, float b)
            => a.Select(aa => Divide(aa, b)).ToArray();
        public static float[][][][] Divide(float[][][][] a, float b)
            => a.Select(aa => Divide(aa, b)).ToArray();
        public static float[] Divide(float a, float[] b)
            => b.Select(bb => a / bb).ToArray();
        public static float[][] Divide(float a, float[][] b)
            => b.Select(bb => Divide(a, bb)).ToArray();
        public static float[][][] Divide(float a, float[][][] b)
            => b.Select(bb => Divide(a, bb)).ToArray();
        public static float[][][][] Divide(float a, float[][][][] b)
            => b.Select(bb => Divide(a, bb)).ToArray();

        // Sign Operations ..............................................
        public static float[] Sign(float[] a)
            => a.Select(ai => (float)Math.Sign(ai)).ToArray();
        public static float[][] Sign(float[][] a)
            => a.Select(Sign).ToArray();
        public static float[][][] Sign(float[][][] a)
            => a.Select(Sign).ToArray();
        public static float[][][][] Sign(float[][][][] a)
            => a.Select(Sign).ToArray();

        // Abs Operations ..............................................
        public static float[] Abs(float[] a)
            => a.Select(Math.Abs).ToArray();
        public static float[][] Abs(float[][] a)
            => a.Select(Abs).ToArray();
        public static float[][][] Abs(float[][][] a)
            => a.Select(Abs).ToArray();
        public static float[][][][] Abs(float[][][][] a)
            => a.Select(Abs).ToArray();
    }
}
