using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Tensor object or node.
    /// This partial class contains Methods for the node.
    /// Generic class that inherits from BackwardNode.
    /// </summary>
    /// <typeparam name="T">Type of Data</typeparam>
    public partial class Tensor<T> : AutoGrad.BackwardNode
    {
        // The data of the Tensor
        public readonly T _data;

        /// <summary>
        /// Add Single node onto graph.
        /// </summary>
        /// <typeparam name="S">'input' type.</typeparam>
        /// <param name="input">The data of the input.</param>
        /// <param name="ctx">The context for the backwards pass.</param>
        /// <param name="backFunc">The Function to be used in the backwards pass.</param>
        /// <param name="requiresGrad">Set if this node needs a gradient</param>
        /// <returns>The added node to graph.</returns>
        public Tensor<S> AddNode<S>(S input, object ctx, AutoGrad.BckF backFunc, bool requiresGrad)
        {
            _connects++;    // Used in backwards pass
                            // Create node
            var node = new Tensor<S>(input, ctx, backFunc, _auto, this, requiresGrad || _requiresGrad);
            // Use for validation of graph
            if (AutoGrad._testing)
                _auto._nodes.Add(node);
            _auto._nodectr++;
            return node;
        }

        /// <summary>
        /// Combine multiple nodes into one single node.
        /// </summary>
        /// <typeparam name="S">Type of input.</typeparam>
        /// <typeparam name="V">Enclosing Type of inputted nodes.</typeparam>
        /// <param name="input">Data of the single node.</param>
        /// <param name="ctx">Context for the backwards pass.</param>
        /// <param name="backFunc">Backwards Function to be executed.</param>
        /// <param name="requiresGrad">If this node needs a gradient.</param>
        /// <param name="nodes">Nodes to be combined into one single node.</param>
        /// <returns>One single node of combined nodes</returns>
        public static NodeArray<S, V> AddConsolidate<S, V>(S input, object ctx, AutoGrad.BckF backFunc, bool requiresGrad, params Tensor<V>[] nodes)
        {
            // For backwards pass
            foreach (var n in nodes)
                n.IncConnects();
            // if gradient calc is required
            requiresGrad = requiresGrad || nodes.Any(n => n._requiresGrad);
            // Find best AutoGrad ref
            AutoGrad auto = null;
            foreach (var n in nodes)
                if (n._auto != null)
                {
                    auto = n._auto;
                    break;
                }
            // Group multiple nodes into one
            var consol = new NodeArray<S, V>(nodes, input, ctx, backFunc, auto, null, requiresGrad, false);
            // Graph Validation
            if (AutoGrad._testing)
                auto._nodes.Add(consol);
            auto._nodectr++;
            return consol;
        }

        /// <summary>
        /// If you want to return multiple nodes. One node to multiple that are encased into a new node.
        /// </summary>
        /// <typeparam name="S">The type of the 'inputs'</typeparam>
        /// <param name="ctx">Context for the backwards pass.</param>
        /// <param name="backFunc">The backwards function to be executed in the backwards pass.</param>
        /// <param name="requiresGrad">If needs gradient.</param>
        /// <param name="inputs">Each of the branch nodes data.</param>
        /// <returns>A single node that holds multiple nodes.</returns>
        public NodeArray<S, S> AddBranch<S>(object ctx, AutoGrad.BckF backFunc, bool requiresGrad, params S[] inputs)
        {
            _connects++;        // For backwards pass
                                // if needs grad
            requiresGrad = requiresGrad || _requiresGrad;
            // Multiple branches from one node. One node to multiple nodes.
            var branches = BranchNode<S>.NewSeries(_auto, requiresGrad, BranchNode<S>.Blank, inputs);
            // The signle node that holds multiple nodes.
            var col = new NodeArray<S, S>(branches, default(S), ctx, backFunc, _auto, this, requiresGrad, true);
            // For backwards pass
            col._connects += branches.Length;
            foreach (var b in branches)
                b._parent = col;
            // Graph validation.
            if (AutoGrad._testing)
            {
                _auto._nodes.Add(col);
                _auto._nodes.AddRange(branches);
            }
            _auto._nodectr += 1 + branches.Length;
            return col;
        }

        /// <summary>
        /// Get Single item value of node. Must be a Singleton dimension.
        /// </summary>
        /// <returns>float data of node</returns>
        public override float Item()
        {
            if (typeof(float) != typeof(T))
                throw new InvalidCastException("Value needs to be a Singleton dimension of type (float).");
            return (float)(object)_data;
        }

        /// <summary>
        /// The data of node as type object.
        /// </summary>
        /// <returns>The data of node as type object.</returns>
        public override object GetData()
            => _data;

        /// <summary>
        /// Try to return the shape of the node.
        /// </summary>
        /// <returns>Try to return the shape of the node.</returns>
        public override string ToString()
            => Shape();

        /// <summary>
        /// Try to return the shape of the node.
        /// </summary>
        /// <returns>As string: Try to return the shape of the node.</returns>
        public string Shape()
        {
            // If no data no shape
            if (_data == null)
                return base.ToString();
            // Get type of data & remove any array brackets[]
            var type = _data.GetType();
            var tx = type.ToString();
            tx = tx.Replace("System.", "");
            var pos = tx.IndexOf('[');
            if (pos >= 0)
                tx = tx.Substring(0, pos);
            // Get Shape
            int[] shape = null;
            if (typeof(float[][]) == type)
                shape = Shape((float[][])this);
            else if (typeof(float[][][]) == type)
                shape = Shape((float[][][])this);
            else if (typeof(float[]) == type)
                shape = Shape((float[])this);
            else if (typeof(float[][][][]) == type)
                shape = Shape((float[][][][])this);
            else if (typeof(int[]) == type)
                shape = Shape((int[])this);
            else if (typeof(int[][]) == type)
                shape = Shape((int[][])this);
            else if (typeof(float) == type)
                return ((float)this).ToString();
            // If unsupported shape
            if (shape == null)
                return base.ToString();
            // Return shape
            return $"{tx}({string.Join(", ", shape)})";
        }
        /// <summary>
        /// Return shape of 1d array.
        /// </summary>
        /// <typeparam name="S"></typeparam>
        /// <param name="values">Array to find the shape of.</param>
        /// <returns>The shape of array.</returns>
        static int[] Shape<S>(S[] values)
            => new[] { values.Length };
        /// <summary>
        /// Return shape of 2d array.
        /// </summary>
        /// <typeparam name="S"></typeparam>
        /// <param name="values">Array to find the shape of.</param>
        /// <returns>The shape of array.</returns>
        static int[] Shape<S>(S[][] values)
            => new[] { values.Length, values[0].Length };
        /// <summary>
        /// Return shape of 3d array.
        /// </summary>
        /// <typeparam name="S"></typeparam>
        /// <param name="values">Array to find the shape of.</param>
        /// <returns>The shape of array.</returns>
        static int[] Shape<S>(S[][][] values)
            => new[] { values.Length, values[0].Length, values[0][0].Length };
        /// <summary>
        /// Return shape of 4d array.
        /// </summary>
        /// <typeparam name="S"></typeparam>
        /// <param name="values">Array to find the shape of.</param>
        /// <returns>The shape of array.</returns>
        static int[] Shape<S>(S[][][][] values)
            => new[] { values.Length, values[0].Length, values[0][0].Length, values[0][0][0].Length };

        /// <summary>
        /// Cast node of type Tensor(object) into type Tensor(float[])
        /// </summary>
        /// <param name="node">Tensor to be casted.</param>
        /// <returns>Casted node.</returns>
        public static Tensor<float[]> CastFloat1d(Tensor<object> node)
            => node.AddNode((float[])node, null, BranchNode<object>.Transfer, node._requiresGrad);
        /// <summary>
        /// Cast node of type Tensor(object) into type Tensor(float[][])
        /// </summary>
        /// <param name="node">Tensor to be casted.</param>
        /// <returns>Casted node.</returns>
        public static Tensor<float[][]> CastFloat2d(Tensor<object> node)
            => node.AddNode((float[][])node, null, BranchNode<object>.Transfer, node._requiresGrad);
        /// <summary>
        /// Cast node of type Tensor(object) into type Tensor(float[][][])
        /// </summary>
        /// <param name="node">Tensor to be casted.</param>
        /// <returns>Casted node.</returns>
        public static Tensor<float[][][]> CastFloat3d(Tensor<object> node)
            => node.AddNode((float[][][])node, null, BranchNode<object>.Transfer, node._requiresGrad);
        /// <summary>
        /// Cast node of type Tensor(object) into type Tensor(float[][][][])
        /// </summary>
        /// <param name="node">Tensor to be casted.</param>
        /// <returns>Casted node.</returns>
        public static Tensor<float[][][][]> CastFloat4d(Tensor<object> node)
            => node.AddNode((float[][][][])node, null, BranchNode<object>.Transfer, node._requiresGrad);
        /// <summary>
        /// Cast node of type Tensor(object) into type Tensor(int[])
        /// </summary>
        /// <param name="node">Tensor to be casted.</param>
        /// <returns>Casted node.</returns>
        public static Tensor<int[]> CastInt1d(Tensor<object> node)
            => node.AddNode((int[])node, null, BranchNode<object>.Transfer, node._requiresGrad);
        /// <summary>
        /// Cast node of type Tensor(object) into type Tensor(int[][])
        /// </summary>
        /// <param name="node">Tensor to be casted.</param>
        /// <returns>Casted node.</returns>
        public static Tensor<int[][]> CastInt2d(Tensor<object> node)
            => node.AddNode((int[][])node, null, BranchNode<object>.Transfer, node._requiresGrad);

        /// <summary>
        /// Tensor object.
        /// </summary>
        /// <param name="input">Data of the Tensor</param>
        /// <param name="ctx">Context of the backwards pass.</param>
        /// <param name="backFunc">Backwards Function to be used in the backwards pass.</param>
        /// <param name="auto">Reference of the AutoGrad obj.</param>
        /// <param name="parent">The parent of this node.</param>
        /// <param name="requiresGrad">If this node needs a gradient.</param>
        public Tensor(T input, object ctx, AutoGrad.BckF backFunc, AutoGrad auto, AutoGrad.BackwardNode parent, bool requiresGrad)
            : base(ctx, backFunc, auto, parent, requiresGrad)
        {
            _data = input;
        }
    }
}
