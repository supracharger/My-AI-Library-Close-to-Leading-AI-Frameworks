using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /// <summary>
    /// Sometimes Nodes need to be consolidated into 1 node or multiple nodes need to be return.
    /// If you are combinging multiple nodes into 1 or branching out from 1 node, this class is used.
    /// </summary>
    /// <typeparam name="T">Type of node.</typeparam>
    /// <typeparam name="S">Type of consolidated or branching nodes.</typeparam>
    public class NodeArray<T, S> : Tensor<T>
    {
        // Nodes combined or branching nodes
        public readonly Tensor<S>[] _nodes;
        // if _nodes are branching nodes
        readonly bool _isbranch;

        /// <summary>
        /// For branch nodes you can simply use obj[index] to get node
        /// </summary>
        /// <param name="index">Index you want to use</param>
        /// <returns>Tensor at index.</returns>
        public Tensor<S> this[int index]
            => _nodes[index];

        /// <summary>
        /// Get Length of nodes. Usually for braching nodes.
        /// </summary>
        public int Length
            => _nodes.Length;

        /// <summary>
        /// Recusive Backwards Node. Should NOT be called externally.
        /// </summary>
        public override void BackNode()
        {
            // Recusive call of BackNode from superclass
            base.BackNode();
            // If consolidated node & time to transverse
            if (!_isbranch && _connects == 0)
            {
                // Recusive call through nodes
                foreach (var n in _nodes)
                    n.BackNode();
            }
        }

        /// <summary>
        /// Sum the gradient for next node or if grad of next node=null: assign grad.
        /// Exits if node or parent node does not need gradient to save calculations.
        /// </summary>
        /// <param name="grad">Grad to be transfered.</param>
        /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
        public override void GradSum(float[] grad, AutoGrad.BackwardNode node = null)
        {
            // Exit: no need to tranfer grad
            if (!_requiresGrad) return;
            // Else transfer grad if branch
            if (_isbranch) base.GradSum(grad);
            // Consolidated node should be at least a 2 dim array.
            else throw new Exception("There should be at least a 2 dim array.");
        }
        /// <summary>
        /// Sum the gradient for next node or if grad of next node=null: assign grad.
        /// Exits if node or parent node does not need gradient to save calculations.
        /// </summary>
        /// <param name="grad">Grad to be transfered.</param>
        /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
        public override void GradSum(float[][] grad, AutoGrad.BackwardNode node = null)
        {
            // Exit: no need to tranfer grad
            if (!_requiresGrad) return;
            // Else transfer grad if branch & Exit
            if (_isbranch)
            {
                base.GradSum(grad);
                return;
            }
            // Validate Shape
            if (grad.Length != _nodes.Length)
                throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}) " +
                            $"it should be ({_nodes.Length} ...).");
            // If Consolidated node: transfer to nodes that were consolidated
            for (int i = 0; i < _nodes.Length; i++)
                base.GradSum(grad[i], _nodes[i]);
        }
        /// <summary>
        /// Sum the gradient for next node or if grad of next node=null: assign grad.
        /// Exits if node or parent node does not need gradient to save calculations.
        /// </summary>
        /// <param name="grad">Grad to be transfered.</param>
        /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
        public override void GradSum(float[][][] grad, AutoGrad.BackwardNode node = null)
        {
            // Exit: no need to tranfer grad
            if (!_requiresGrad) return;
            // Else transfer grad if branch & Exit
            if (_isbranch)
            {
                base.GradSum(grad);
                return;
            }
            // Validate Shape
            if (grad.Length != _nodes.Length)
                throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}) " +
                            $"it should be ({_nodes.Length} ...).");
            // If Consolidated node: transfer to nodes that were consolidated
            for (int i = 0; i < _nodes.Length; i++)
                base.GradSum(grad[i], _nodes[i]);
        }
        /// <summary>
        /// Sum the gradient for next node or if grad of next node=null: assign grad.
        /// Exits if node or parent node does not need gradient to save calculations.
        /// </summary>
        /// <param name="grad">Grad to be transfered.</param>
        /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
        public override void GradSum(float[][][][] grad, AutoGrad.BackwardNode node = null)
        {
            // Exit: no need to tranfer grad
            if (!_requiresGrad) return;
            // Else transfer grad if branch & Exit
            if (_isbranch)
            {
                base.GradSum(grad);
                return;
            }
            // Validate Shape
            if (grad.Length != _nodes.Length)
                throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}, {grad[0][0][0].Length}) " +
                            $"it should be ({_nodes.Length} ...).");
            // If Consolidated node: transfer to nodes that were consolidated
            for (int i = 0; i < _nodes.Length; i++)
                base.GradSum(grad[i], _nodes[i]);
        }
        /// <summary>
        /// Sum the gradient for next node or if grad of next node=null: assign grad.
        /// Exits if node or parent node does not need gradient to save calculations.
        /// </summary>
        /// <param name="grad">Grad to be transfered.</param>
        /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
        public override void GradSum(float[][][][][] grad, AutoGrad.BackwardNode node = null)
        {
            // Exit: no need to tranfer grad
            if (!_requiresGrad) return;
            // Else transfer grad if branch & Exit
            if (_isbranch)
            {
                base.GradSum(grad);
                return;
            }
            // Validate Shape
            if (grad.Length != _nodes.Length)
                throw new Exception($"Invalid gradient shape: got ({grad.Length}, {grad[0].Length}, {grad[0][0].Length}, {grad[0][0][0].Length}, {grad[0][0][0][0].Length}) " +
                            $"it should be ({_nodes.Length} ...).");
            // If Consolidated node: transfer to nodes that were consolidated
            for (int i = 0; i < _nodes.Length; i++)
                base.GradSum(grad[i], _nodes[i]);
        }

        /// <summary>
        /// Implicit cast to Array of nodes.
        /// </summary>
        /// <param name="array">NodeArray object to be cast</param>
        public static implicit operator Tensor<S>[](NodeArray<T, S> array)
            => array._nodes;

        /// <summary>
        /// Sometimes Nodes need to be consolidated into 1 node or multiple nodes need to be return.
        /// If you are combinging multiple nodes into 1 or branching out from 1 node, this class is used.
        /// </summary>
        /// <param name="nodes">Consolidated nodes or branching nodes</param>
        /// <param name="input">data of consolidated node.</param>
        /// <param name="ctx">Context for backwards pass</param>
        /// <param name="backFunc">Backwards pass for node</param>
        /// <param name="auto">AutoGrad ref</param>
        /// <param name="parent">Parent of node</param>
        /// <param name="requiresGrad">If node needs grad.</param>
        /// <param name="isbranch">True: if branching node. False: if consolidated node.</param>
        public NodeArray(Tensor<S>[] nodes, T input, object ctx, AutoGrad.BckF backFunc, AutoGrad auto, AutoGrad.BackwardNode parent,
                         bool requiresGrad, bool isbranch)
            : base(input, ctx, backFunc, auto, parent, requiresGrad || nodes.Any(n => n._requiresGrad))
        {
            _nodes = nodes;
            _isbranch = isbranch;
        }
    }

    /// <summary>
    /// This is used if you want to go from 1 single node to multiple nodes.
    /// </summary>
    /// <typeparam name="T">Type of data of the node.</typeparam>
    class BranchNode<T> : Tensor<T>
    {
        // Make sure this node is only calc once in the backwards pass.
        readonly int[] _ctr;

        /// <summary>
        /// Recusive Function for backwards pass. Should NOT be called.
        /// </summary>
        public override void BackNode()
        {
            // Validation
            if (--_connects < 0 && !_isStartNode)
                throw new Exception("Broken Gradient Graph.");
            // if more than one connection has to be called yet, or other branching nodes have not been called yet
            if (_connects > 0 || --_ctr[0] > 0)
            {
                // If this node will not be called again
                if (_connects == 0)
                {
                    // If not root node: dec counter for validation
                    if (!_isStartNode)
                        _auto._nodectr--;
                    // if it has a parent, don't call it. The last node in the branch will call it.
                    if (_parent != null)
                    {
                        if (_parent.DecConnects() < 0 && !_parent._isStartNode)
                            throw new Exception("Broken Gradient Graph.");
                        _parent = null;
                    }
                }
                // Exit
                return;
            }
            // Undo _connects-- above
            _connects++;
            // Call recursive call
            base.BackNode();
        }

        /// <summary>
        /// Create Branch Nodes 
        /// This is used if you want to go from 1 single node to multiple nodes.
        /// </summary>
        /// <typeparam name="S">Type of 'inputs'</typeparam>
        /// <param name="auto">AutoGrad ref</param>
        /// <param name="requiresGrad">If node requires grad</param>
        /// <param name="BckFunc">Backward Function</param>
        /// <param name="inputs">Nodes[] data for each.</param>
        /// <returns>Array of Branch Nodes.</returns>
        public static BranchNode<S>[] NewSeries<S>(AutoGrad auto, bool requiresGrad, AutoGrad.BckF BckFunc, params S[] inputs)
        {
            // Make sure this node is only calc once in the backwards pass.
            var ctr = new[] { inputs.Length };
            // Create Branch Nodes
            return inputs.Select(x => new BranchNode<S>(x, auto, requiresGrad, ctr, BckFunc)).ToArray();
        }

        /// <summary>
        /// Do nothing in Backwards Function.
        /// </summary>
        /// <param name="y">Forward pass out node.</param>
        public static void Blank(AutoGrad.BackwardNode y) { }
        /// <summary>
        /// Backwards Function: just tranfer the gradient
        /// </summary>
        /// <param name="y">Forward pass out node.</param>
        public static void Transfer(AutoGrad.BackwardNode y)
        {
            if (y._parent._grad != null) throw new Exception("y._parent._grad != null");
            y._parent._grad = y._grad;
        }

        BranchNode(T input, AutoGrad auto, bool requiresGrad, int[] ctr, AutoGrad.BckF BckFunc)
            : base(input, null, BckFunc, auto, null, requiresGrad)
            => _ctr = ctr;
    }
}
