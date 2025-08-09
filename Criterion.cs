using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Loss Functions to use during training
    /// </summary>
    public static class Criterion
    {
        // Standard Gradient Descent Loss Funtion =============================================================
        /// <summary>
        /// Standard Gradient Descent Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value</returns>
        public static Tensor<float> Regular(Tensor<float[]> y, float[] labels)
        {
            // Validate Shape
            if (y._data.Length != labels.Length)
                throw new Exception($"Shape should be equal: got y({y._data.Length}) != labels({labels.Length})");
            // Criterion
            var grad = Regular(y._data, labels);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Node
            return y.AddNode(loss, grad, Backward1d, false);
        }
        /// <summary>
        /// Standard Gradient Descent Loss Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value</returns>
        public static Tensor<float> Regular(Tensor<float[][]> y, float[][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}) != labels({lab.Length}, {lab[0].Length})");
            // Criterion
            var grad = Regular(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Node
            return y.AddNode(loss, grad, Backward2d, false);
        }
        /// <summary>
        /// Standard Gradient Descent Loss Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value</returns>
        public static Tensor<float> Regular(Tensor<float[][][]> y, float[][][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length || yy[0][0].Length != lab[0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length})");
            // Criterion
            var grad = Regular(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward3d, false);
        }
        /// <summary>
        /// Standard Gradient Descent Loss Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value</returns>
        public static Tensor<float> Regular(Tensor<float[][][][]> y, float[][][][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length ||
                yy[0][0].Length != lab[0][0].Length || yy[0][0][0].Length != lab[0][0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}, {yy[0][0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length}, {lab[0][0][0].Length})");
            // Criterion
            var grad = Regular(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward4d, false);
        }
        // Tensor & Tensor .............
        /// <summary>
        /// Standard Gradient Descent Loss Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value</returns>
        public static Tensor<float> Regular(Tensor<float[]> y, Tensor<float[]> labels)
        {
            // Validate Shape
            if (y._data.Length != labels._data.Length)
                throw new Exception($"Shape should be equal: got y({y._data.Length}) != labels({labels._data.Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = Regular(y._data, labels._data);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward1d, false);
        }
        /// <summary>
        /// Standard Gradient Descent Loss Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value</returns>
        public static Tensor<float> Regular(Tensor<float[][]> y, Tensor<float[][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}) != labels({lab.Length}, {lab[0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = Regular(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward2d, false);
        }
        /// <summary>
        /// Standard Gradient Descent Loss Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value</returns>
        public static Tensor<float> Regular(Tensor<float[][][]> y, Tensor<float[][][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length || yy[0][0].Length != lab[0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = Regular(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward3d, false);
        }
        /// <summary>
        /// Standard Gradient Descent Loss Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value</returns>
        public static Tensor<float> Regular(Tensor<float[][][][]> y, Tensor<float[][][][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length || 
                yy[0][0].Length != lab[0][0].Length || yy[0][0][0].Length != lab[0][0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}, {yy[0][0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length}, {lab[0][0][0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = Regular(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward4d, false);
        }
        // Array & Array .............
        /// <summary>
        /// Standard Gradient Descent Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[] Regular(float[] y, float[] labels)
        {
            var grad = new float[labels.Length];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = y[i] - labels[i];
            return grad;
        }
        /// <summary>
        /// Standard Gradient Descent Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][] Regular(float[][] y, float[][] labels)
        {
            var grad = new float[labels.Length][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = Regular(y[i], labels[i]);
            return grad;
        }
        /// <summary>
        /// Standard Gradient Descent Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][][] Regular(float[][][] y, float[][][] labels)
        {
            var grad = new float[labels.Length][][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = Regular(y[i], labels[i]);
            return grad;
        }
        /// <summary>
        /// Standard Gradient Descent Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][][][] Regular(float[][][][] y, float[][][][] labels)
        {
            var grad = new float[labels.Length][][][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = Regular(y[i], labels[i]);
            return grad;
        }

        // Mean Squared Error Loss Funtion =============================================================
        /// <summary>
        /// Mean Squared Error Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> MSE(Tensor<float[]> y, float[] labels)
        {
            // Validate Shape
            if (y._data.Length != labels.Length)
                throw new Exception($"Shape should be equal: got y({y._data.Length}) != labels({labels.Length})");
            // Criterion
            var grad = MSE(y._data, labels);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward1d, false);
        }
        /// <summary>
        /// Mean Squared Error Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> MSE(Tensor<float[][]> y, float[][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}) != labels({lab.Length}, {lab[0].Length})");
            // Criterion
            var grad = MSE(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward2d, false);
        }
        /// <summary>
        /// Mean Squared Error Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> MSE(Tensor<float[][][]> y, float[][][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length || yy[0][0].Length != lab[0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length})");
            // Criterion
            var grad = MSE(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward3d, false);
        }
        /// <summary>
        /// Mean Squared Error Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> MSE(Tensor<float[][][][]> y, float[][][][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length ||
                yy[0][0].Length != lab[0][0].Length || yy[0][0][0].Length != lab[0][0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}, {yy[0][0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length}, {lab[0][0][0].Length})");
            // Criterion
            var grad = MSE(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward4d, false);
        }
        // Tensor & Tensor .............
        /// <summary>
        /// Mean Squared Error Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Tensor of Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> MSE(Tensor<float[]> y, Tensor<float[]> labels)
        {
            // Validate Shape
            if (y._data.Length != labels._data.Length)
                throw new Exception($"Shape should be equal: got y({y._data.Length}) != labels({labels._data.Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = MSE(y._data, labels._data);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward1d, false);
        }
        /// <summary>
        /// Mean Squared Error Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Tensor of Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> MSE(Tensor<float[][]> y, Tensor<float[][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}) != labels({lab.Length}, {lab[0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = MSE(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward2d, false);
        }
        /// <summary>
        /// Mean Squared Error Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Tensor of Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> MSE(Tensor<float[][][]> y, Tensor<float[][][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length || yy[0][0].Length != lab[0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = MSE(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward3d, false);
        }
        /// <summary>
        /// Mean Squared Error Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Tensor of Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> MSE(Tensor<float[][][][]> y, Tensor<float[][][][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length ||
                yy[0][0].Length != lab[0][0].Length || yy[0][0][0].Length != lab[0][0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}, {yy[0][0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length}, {lab[0][0][0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = MSE(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward4d, false);
        }
        // Array & Array .............
        /// <summary>
        /// Mean Squared Error Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[] MSE(float[] y, float[] labels)
        {
            // My version MSE
            // The problem with regular MSE is when the magnitude of the gradient is smaller than one, it makes the value very small
            var grad = new float[labels.Length];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = 3 * (y[i] - labels[i]);
            return grad;
        }
        /// <summary>
        /// Mean Squared Error Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][] MSE(float[][] y, float[][] labels)
        {
            var grad = new float[labels.Length][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = MSE(y[i], labels[i]);
            return grad;
        }
        /// <summary>
        /// Mean Squared Error Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][][] MSE(float[][][] y, float[][][] labels)
        {
            var grad = new float[labels.Length][][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = MSE(y[i], labels[i]);
            return grad;
        }
        /// <summary>
        /// Mean Squared Error Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][][][] MSE(float[][][][] y, float[][][][] labels)
        {
            var grad = new float[labels.Length][][][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = MSE(y[i], labels[i]);
            return grad;
        }

        // Conservative Loss Funtion =============================================================
        /// <summary>
        /// Conservative Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> Conservative(Tensor<float[]> y, float[] labels)
        {
            // Validate Shape
            if (y._data.Length != labels.Length)
                throw new Exception($"Shape should be equal: got y({y._data.Length}) != labels({labels.Length})");
            // Criterion
            var grad = Conservative(y._data, labels);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward1d, false);
        }
        /// <summary>
        /// Conservative Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> Conservative(Tensor<float[][]> y, float[][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}) != labels({lab.Length}, {lab[0].Length})");
            // Criterion
            var grad = Conservative(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward2d, false);
        }
        /// <summary>
        /// Conservative Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> Conservative(Tensor<float[][][]> y, float[][][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length || yy[0][0].Length != lab[0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length})");
            // Criterion
            var grad = Conservative(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward3d, false);
        }
        /// <summary>
        /// Conservative Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> Conservative(Tensor<float[][][][]> y, float[][][][] labels)
        {
            var yy = y._data;
            var lab = labels;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length ||
                yy[0][0].Length != lab[0][0].Length || yy[0][0][0].Length != lab[0][0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}, {yy[0][0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length}, {lab[0][0][0].Length})");
            // Criterion
            var grad = Conservative(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward4d, false);
        }
        // Tensor & Tensor .............
        /// <summary>
        /// Conservative Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Tensor of Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> Conservative(Tensor<float[]> y, Tensor<float[]> labels)
        {
            // Validate Shape
            if (y._data.Length != labels._data.Length)
                throw new Exception($"Shape should be equal: got y({y._data.Length}) != labels({labels._data.Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = Conservative(y._data, labels._data);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward1d, false);
        }
        // Tensor & Tensor .............
        /// <summary>
        /// Conservative Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Tensor of Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> Conservative(Tensor<float[][]> y, Tensor<float[][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}) != labels({lab.Length}, {lab[0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = Conservative(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward2d, false);
        }
        // Tensor & Tensor .............
        /// <summary>
        /// Conservative Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Tensor of Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> Conservative(Tensor<float[][][]> y, Tensor<float[][][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length || yy[0][0].Length != lab[0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = Conservative(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward3d, false);
        }
        // Tensor & Tensor .............
        /// <summary>
        /// Conservative Loss Funtion
        /// </summary>
        /// <param name="y">Tensor Output of Network</param>
        /// <param name="labels">Tensor of Target labels to reach</param>
        /// <returns>Single Loss Value as Tensor</returns>
        public static Tensor<float> Conservative(Tensor<float[][][][]> y, Tensor<float[][][][]> labels)
        {
            var yy = y._data;
            var lab = labels._data;
            // Validate Shape
            if (yy.Length != lab.Length || yy[0].Length != lab[0].Length ||
                yy[0][0].Length != lab[0][0].Length || yy[0][0][0].Length != lab[0][0][0].Length)
                throw new Exception($"Shape should be equal: got y({yy.Length}, {yy[0].Length}, {yy[0][0].Length}, {yy[0][0][0].Length}) != " +
                        $"labels({lab.Length}, {lab[0].Length}, {lab[0][0].Length}, {lab[0][0][0].Length})");
            // Labels should not require grad. It won't be used in the graph
            if (labels._requiresGrad)
                throw new Exception("Labels should not require a gradient.");
            // Criterion
            var grad = Conservative(yy, lab);
            // Loss
            var loss = GetError(grad).Average();
            // Add Leaf Tensor
            return y.AddNode(loss, grad, Backward4d, false);
        }
        // Array & Array .............
        /// <summary>
        /// Conservative Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[] Conservative(float[] y, float[] labels)
        {
            var grad = new float[labels.Length];
            for (int i = 0; i < labels.Length; i++)
            {
                var lab = labels[i];
                var yi = y[i];
                var g = yi - lab;
                // Pentalize for outputs greather than targets or not of the same sign
                if ((lab >= 0 && yi > lab) || (lab < 0 && yi < lab) || (yi >= 0) != (lab >= 0))
                    g *= 3;
                grad[i] = g;
            }
            return grad;
        }
        /// <summary>
        /// Conservative Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][] Conservative(float[][] y, float[][] labels)
        {
            var grad = new float[labels.Length][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = Conservative(y[i], labels[i]);
            return grad;
        }
        /// <summary>
        /// Conservative Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][][] Conservative(float[][][] y, float[][][] labels)
        {
            var grad = new float[labels.Length][][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = Conservative(y[i], labels[i]);
            return grad;
        }
        /// <summary>
        /// Conservative Gradient Funtion
        /// </summary>
        /// <param name="y">Output of Network</param>
        /// <param name="labels">Target labels to reach</param>
        /// <returns>New Array of Gradient values</returns>
        public static float[][][][] Conservative(float[][][][] y, float[][][][] labels)
        {
            var grad = new float[labels.Length][][][];
            for (int i = 0; i < labels.Length; i++)
                grad[i] = Conservative(y[i], labels[i]);
            return grad;
        }

        // Get Errors from Gradient =============================================================
        /// <summary>
        /// Gets Errors from Gradient Values
        /// </summary>
        /// <param name="grad">The Gradient</param>
        /// <returns>Errors for each Gradient at dimention (0)</returns>
        public static List<float> GetError(float[] grad)
        {
            var errors = new List<float>(grad.Length);
            for (int i = 0; i < grad.Length; i++)
                errors.Add(Math.Abs(grad[i]));
            return errors;
        }
        /// <summary>
        /// Gets Errors from Gradient Values
        /// </summary>
        /// <param name="grad">The Gradient</param>
        /// <returns>Errors for each Gradient at dimention (0)</returns>
        public static List<float> GetError(float[][] grad)
        {
            var errors = new List<float>(grad.Length);
            for (int i = 0; i < grad.Length; i++)
                errors.Add(GetError(grad[i]).Average());
            return errors;
        }
        /// <summary>
        /// Gets Errors from Gradient Values
        /// </summary>
        /// <param name="grad">The Gradient</param>
        /// <returns>Errors for each Gradient at dimention (0)</returns>
        public static List<float> GetError(float[][][] grad)
        {
            var errors = new List<float>(grad.Length);
            for (int i = 0; i < grad.Length; i++)
                errors.Add(GetError(grad[i]).Average());
            return errors;
        }
        /// <summary>
        /// Gets Errors from Gradient Values
        /// </summary>
        /// <param name="grad">The Gradient</param>
        /// <returns>Errors for each Gradient at dimention (0)</returns>
        public static List<float> GetError(float[][][][] grad)
        {
            var errors = new List<float>(grad.Length);
            for (int i = 0; i < grad.Length; i++)
                errors.Add(GetError(grad[i]).Average());
            return errors;
        }

        // Backwards Pass for Tensor =============================================================
        /// <summary>
        /// Inside Backward Function for Tensor
        /// </summary>
        /// <param name="y">Output Node of forward pass</param>
        static void Backward1d(AutoGrad.BackwardNode y)
        {
            var g = (float)y._grad;
            // Validation
            if (g != 1)
                throw new Exception("This node should be a leaf node.");
            // Tranfer Gradient to Parent Node
            y.GradSum((float[])y._ctx);
        }
        /// <summary>
        /// Inside Backward Function for Tensor
        /// </summary>
        /// <param name="y">Output Node of forward pass</param>
        static void Backward2d(AutoGrad.BackwardNode y)
        {
            var g = (float)y._grad;
            // Validation
            if (g != 1)
                throw new Exception("This node should be a leaf node.");
            // Tranfer Gradient to Parent Node
            y.GradSum((float[][])y._ctx);
        }
        /// <summary>
        /// Inside Backward Function for Tensor
        /// </summary>
        /// <param name="y">Output Node of forward pass</param>
        static void Backward3d(AutoGrad.BackwardNode y)
        {
            var g = (float)y._grad;
            // Validation
            if (g != 1)
                throw new Exception("This node should be a leaf node.");
            // Tranfer Gradient to Parent Node
            y.GradSum((float[][][])y._ctx);
        }
        /// <summary>
        /// Inside Backward Function for Tensor
        /// </summary>
        /// <param name="y">Output Node of forward pass</param>
        static void Backward4d(AutoGrad.BackwardNode y)
        {
            var g = (float)y._grad;
            // Validation
            if (g != 1)
                throw new Exception("This node should be a leaf node.");
            // Tranfer Gradient to Parent Node
            y.GradSum((float[][][][])y._ctx);
        }
    }
}
