using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MiniNet
{
    /***
     * AutoGrad Automatically defines the Backwards pass from the given Forward pass.
     * Be sure to set root nodes or inputted nodes in graph using AddRoot() for a single 
       item or AddRootCollection() for multiple items at once.
     ***/
    public class AutoGrad
    {
        // Backwards function type
        public delegate void BckF (BackwardNode y);
        // The roots or inputs that start at the begining
        readonly List<BackwardNode> _roots;
        // Used if '_testing' = true
        public readonly List<BackwardNode> _nodes;
        // Counts nodes in graph
        public int _nodectr;
        // Extra validation checks if equal to true
        public static bool _testing = false;

        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="input">Input to the graph of any type.</param>
        /// <returns>Tensor</returns>
        public Tensor<object> AddRoot(object input)
        {
            var root = new Tensor<object>(input, null, null, this, null, false)
            { _isStartNode = true };
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="input">Input to the graph of float[] type.</param>
        /// <returns>Tensor</returns>
        public Tensor<float[]> AddRoot(float[] input)
        {
            var root = new Tensor<float[]>(input, null, null, this, null, false)
            { _isStartNode = true };
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="input">Input to the graph of float[][] type.</param>
        /// <returns>Tensor</returns>
        public Tensor<float[][]> AddRoot(float[][] input)
        {
            var root = new Tensor<float[][]>(input, null, null, this, null, false)
            { _isStartNode = true };
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="input">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<float[][][]> AddRoot(float[][][] input)
        {
            var root = new Tensor<float[][][]>(input, null, null, this, null, false)
            { _isStartNode = true };
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="input">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<float[][][][]> AddRoot(float[][][][] input)
        {
            var root = new Tensor<float[][][][]>(input, null, null, this, null, false)
            { _isStartNode = true };
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="input">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<int[]> AddRoot(int[] input)
        {
            var root = new Tensor<int[]>(input, null, null, this, null, false)
            { _isStartNode = true };
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="input">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<int[][]> AddRoot(int[][] input)
        {
            var root = new Tensor<int[][]>(input, null, null, this, null, false)
            { _isStartNode = true };
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="root">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<float[]> AddRoot(Tensor<float[]> root)
        {
            root.Clear(this);
            root._isStartNode = true;
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="root">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<float[][]> AddRoot(Tensor<float[][]> root)
        {
            root.Clear(this);
            root._isStartNode = true;
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="root">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<float[][][]> AddRoot(Tensor<float[][][]> root)
        {
            root.Clear(this);
            root._isStartNode = true;
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="root">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<float[][][][]> AddRoot(Tensor<float[][][][]> root)
        {
            root.Clear(this);
            root._isStartNode = true;
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="root">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<int[]> AddRoot(Tensor<int[]> root)
        {
            root.Clear(this);
            root._isStartNode = true;
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="root">Input to the graph.</param>
        /// <returns>Tensor</returns>
        public Tensor<int[][]> AddRoot(Tensor<int[][]> root)
        {
            root.Clear(this);
            root._isStartNode = true;
            _roots.Add(root);
            //_nodectr++;
            return root;
        }
        /// <summary>
        /// For every input tensor this must be called.
        /// </summary>
        /// <param name="input">Input to the graph of any Tensor<T> type.</param>
        /// <param name="requiresGrad">Set to true if you want gradients for this tensor.</param>
        /// <returns></returns>
        public Tensor<object> AddRoot(BackwardNode input, bool requiresGrad = false)
        {
            input.Clear(this, requiresGrad);
            input._isStartNode = true;
            _roots.Add(input);
            var root = new Tensor<object>(input.GetData(), null, BranchNode<object>.Transfer, this, input, input._requiresGrad);
            //_nodectr++;
            return root;
        }

        /// <summary>
        /// For multiple inputs at the same time. For every input tensor this must be called.
        /// </summary>
        /// <param name="inputs">Numerious objects of any type.</param>
        /// <returns>Tensor containing multiple values.</returns>
        public NodeArray<object, object> AddRootCollection(params object[] inputs)
        {
            // Multiple items to be used.
            var branches = BranchNode<object>.NewSeries(this, false, BranchNode<object>.Blank, inputs);
            // Make sure the graph reaches the start
            foreach (var b in branches) b._isStartNode = true;
            // The Tensor that contains multiple values
            var col = new NodeArray<object, object>(branches, null, null, null, this, null, false, true);
            _roots.AddRange(branches);
            return col;
        }
        /// <summary>
        /// For multiple inputs at the same time. For every input tensor this must be called.
        /// </summary>
        /// <param name="inputs">Numerious objects of any Tensor<T> type.</param>
        /// <returns>Tensor containing multiple values.</returns>
        public NodeArray<object, object> AddRootCollection(params BackwardNode[] inputs)
        {
            // Loop each input
            foreach(var n in inputs)
            {
                // If requiresGrad=True make sure it is of supported type
                if (n._requiresGrad)
                {
                    var type = n.GetData().GetType();
                    // supported types of requiresGrad=True
                    if (typeof(float[][]) != type && typeof(float[][][]) != type && typeof(float[][][][]) != type &&
                        typeof(float[]) != type)
                    {
                        // Not supported if requiresGrad=True
                        if (typeof(float) == type)
                            throw new NotSupportedException($"Type: ({type}) is not supported when requiresGrad=true. " +
                                    "Set _requiresGrad=false for this root node.");
                        throw new Exception($"There are no gradient to calculate for type:({type}). " +
                                    "Set _requiresGrad=false for this root node.");
                    }
                }
                n.Clear(this);              // Reset
                n._isStartNode = true;      // Make sure graph reaches the start
            }
            // Transition nodes to be returned to transfer gradients to the inputted nodes
            var branches = inputs.Select(n => BranchNode<object>.NewSeries(this, n._requiresGrad, BranchNode<object>.Transfer, n.GetData())[0]).ToArray();
            for (int i = 0; i < inputs.Length; i++)
                branches[i]._parent = inputs[i];
            // Tensor containing multiple nodes
            var col = new NodeArray<object, object>(branches, null, null, null, this, null, false, true);
            //col._connects += branches.Length;
            //foreach (var b in branches)
            //    b._parent = col;
            //_nodes.Add(col);
            _roots.AddRange(inputs);
            _nodectr += inputs.Length;  // for validation check
            return col;
        }

        /// <summary>
        /// Initialize Graph.
        /// </summary>
        public AutoGrad()
        {
            _roots = new List<BackwardNode>();
            if (_testing)
                _nodes = new List<BackwardNode>();
        }

        /// <summary>
        /// Parent of Tensor(T) for backwards pass.
        /// </summary>
        public abstract class BackwardNode
        {
            // Parent Node for Backwards Pass
            public BackwardNode _parent;
            // If root node
            public bool _isStartNode;
            // Context for backwards pass
            public object _ctx { get; private set; }
            // Gradient for backwards pass
            public object _grad;
            // If needs grad otherwise skips backwards pass calc
            public bool _requiresGrad { get; private set; }
            // Backwards function used in backwards pass
            public BckF _BackwardFunc { get; private set; }
            // AutoGrad ref
            public AutoGrad _auto { get; private set; }
            // Connections to node from forwards pass, modified in backwards pass
            protected int _connects;

            /// <summary>
            /// Execute backwards pass.
            /// </summary>
            /// <param name="outgrad">Not needed. Pass in gradient if you want to.</param>
            public void Backward(object outgrad = null)
            {
                // is null: make sure node is of Singleton dimension
                if (outgrad == null)
                {
                    Item();
                    _grad = 1f;
                }
                // not null: pass in gradient
                else
                    _grad = outgrad;
                // Used in backwards pass
                _connects++;
                // Count of nodes from forward pass
                var nodecnt = _auto._nodectr;
                // Recursive call of backwards pass
                BackNode();
                // Backwards pass validation
                if (_auto._nodectr > 0)
                    throw new Exception($"Broken gradient graph: there was ({_auto._nodectr}) more connection(s) in the " +
                            "forward pass than the backward pass.");
                else if (_auto._nodectr < 0)
                    throw new Exception($"Broken gradient graph: there was ({-_auto._nodectr}) more connection(s) in the " +
                            "backward pass than the forward pass.");
                if (_testing)
                {
                    if (nodecnt != _auto._nodes.Count) throw new Exception("nodecnt != _auto._nodes.Count");
                    var connects = _auto._nodes.Select(n => n._connects).ToList();
                    if (connects.Max() != 0 || connects.Min() != 0) throw new Exception("connects.Max() != 0 || connects.Min() != 0");
                }
                // Validation: Loop through root nodes
                for (int i = 0; i < _auto._roots.Count; i++)
                {
                    var root = _auto._roots[i];     // Current root node
                    var c = root._connects;         // Should have connect of zero
                    if (c > 0)
                        throw new Exception($"Broken gradient graph for root node ({i}): backwards pass needs ({c}) " +
                            $"more connection(s) for the root node.");
                    else if (c < 0)
                        throw new Exception($"There backward pass had ({Math.Abs(c)}) extra connections than the " +
                            $"forward pass on root node ({i}).");
                }
                // More validation if _testing=True. Make sure nodes with requireGrad have a grad.
                if (_testing)
                {
                    var ix = -1;
                    foreach (var n in _auto._roots.Concat(_auto._nodes))
                    {
                        ix++;
                        if (n._requiresGrad && n._grad == null)
                            throw new Exception($"Node at position ({ix}) did not get a gradient.");
                        else if (!n._requiresGrad && n._grad != null)
                            throw new Exception($"Node at position ({ix}) should not have a gradient.");
                    }
                }
            }

            /// <summary>
            /// Recursive Backwards call. Go backwards through the forwards pass. 
            /// Should NOT be called externally.
            /// </summary>
            public virtual void BackNode()
            {
                // Validation
                if (--_connects < 0 && !_isStartNode)
                    throw new Exception("Broken Gradient Graph.");
                // If root node or not ready to transverse up the tree: exit
                if (_isStartNode || _connects > 0)
                    return;
                // Counter for validation
                _auto._nodectr--;
                // Execute backwards Function for node
                var parent = _parent;
                if (_requiresGrad)
                    _BackwardFunc(this);
                // Free Memory
                _parent = null;
                // Exit if no parent
                if (parent == null) return;
                // Else run parent node
                parent.BackNode();
            }

            /// <summary>
            /// Sum the gradient for next node or if grad of next node=null: assign grad.
            /// Exits if node or parent node does not need gradient to save calculations.
            /// </summary>
            /// <param name="grad">Grad to be transfered.</param>
            /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
            public virtual void GradSum(float[] grad, BackwardNode node = null)
            {
                // If null: node is parent
                if (node == null)
                    node = _parent;
                // Exit if node does not need grad
                if (!node._requiresGrad)
                    return;
                // if grad is null assign grad and exit.
                if (node._grad == null)
                {
                    node._grad = grad;
                    return;
                }
                // Else if there is existing gradient sum it
                var pgrad = (float[])node._grad;
                if (pgrad.Length != grad.Length)    // Shape validation.
                    throw new Exception($"pgrad({pgrad.Length}) != grad({grad.Length}).");
                // Sum gradient to pgrad
                Ensum(pgrad, grad);
            }
            /// <summary>
            /// Sum the gradient for next node or if grad of next node=null: assign grad.
            /// Exits if node or parent node does not need gradient to save calculations.
            /// </summary>
            /// <param name="grad">Grad to be transfered.</param>
            /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
            public virtual void GradSum(float[][] grad, BackwardNode node = null)
            {
                // If null: node is parent
                if (node == null)
                    node = _parent;
                // Exit if node does not need grad
                if (!node._requiresGrad)
                    return;
                // if grad is null assign grad and exit.
                if (node._grad == null)
                {
                    node._grad = grad;
                    return;
                }
                // Else if there is existing gradient sum it
                var pgrad = (float[][])node._grad;
                if (pgrad.Length != grad.Length || pgrad[0].Length != grad[0].Length)   // Shape validation.
                    throw new Exception($"pgrad({pgrad.Length}, {pgrad[0].Length}) != grad({grad.Length}, {grad[0].Length}).");
                // Sum gradient to pgrad
                Ensum(pgrad, grad);
            }
            /// <summary>
            /// Sum the gradient for next node or if grad of next node=null: assign grad.
            /// Exits if node or parent node does not need gradient to save calculations.
            /// </summary>
            /// <param name="grad">Grad to be transfered.</param>
            /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
            public virtual void GradSum(float[][][] grad, BackwardNode node = null)
            {
                // If null: node is parent
                if (node == null)
                    node = _parent;
                // Exit if node does not need grad
                if (!node._requiresGrad)
                    return;
                // if grad is null assign grad and exit.
                if (node._grad == null)
                {
                    node._grad = grad;
                    return;
                }
                // Else if there is existing gradient sum it
                var pgrad = (float[][][])node._grad;
                // Shape validation.
                if (pgrad.Length != grad.Length || pgrad[0].Length != grad[0].Length || pgrad[0][0].Length != grad[0][0].Length)
                    throw new Exception($"pgrad({pgrad.Length}, {pgrad[0].Length}, {pgrad[0][0].Length}) != grad({grad.Length}, {grad[0].Length}, {grad[0][0].Length}).");
                // Sum gradient to pgrad
                Ensum(pgrad, grad);
            }
            /// <summary>
            /// Sum the gradient for next node or if grad of next node=null: assign grad.
            /// Exits if node or parent node does not need gradient to save calculations.
            /// </summary>
            /// <param name="grad">Grad to be transfered.</param>
            /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
            public virtual void GradSum(float[][][][] grad, BackwardNode node = null)
            {
                // If null: node is parent
                if (node == null)
                    node = _parent;
                // Exit if node does not need grad
                if (!node._requiresGrad)
                    return;
                // if grad is null assign grad and exit.
                if (node._grad == null)
                {
                    node._grad = grad;
                    return;
                }
                // Else if there is existing gradient sum it
                var pgrad = (float[][][][])node._grad;
                // Shape validation.
                if (pgrad.Length != grad.Length || pgrad[0].Length != grad[0].Length ||
                    pgrad[0][0].Length != grad[0][0].Length || pgrad[0][0][0].Length != grad[0][0][0].Length)
                    throw new Exception($"pgrad({pgrad.Length}, {pgrad[0].Length}, {pgrad[0][0].Length}, {pgrad[0][0][0].Length}) != " +
                            $"grad({grad.Length}, {grad[0].Length}, {grad[0][0].Length}, {grad[0][0][0].Length}).");
                // Sum gradient to pgrad
                Ensum(pgrad, grad);
            }
            /// <summary>
            /// Sum the gradient for next node or if grad of next node=null: assign grad.
            /// Exits if node or parent node does not need gradient to save calculations.
            /// </summary>
            /// <param name="grad">Grad to be transfered.</param>
            /// <param name="node">Usually null. Unless Consolidated node or Branch node</param>
            public virtual void GradSum(float[][][][][] grad, BackwardNode node = null)
            {
                // If null: node is parent
                if (node == null)
                    node = _parent;
                // Exit if node does not need grad
                if (!node._requiresGrad)
                    return;
                // if grad is null assign grad and exit.
                if (node._grad == null)
                {
                    node._grad = grad;
                    return;
                }
                // Else if there is existing gradient sum it
                var pgrad = (float[][][][][])node._grad;
                // Shape validation.
                if (pgrad.Length != grad.Length || pgrad[0].Length != grad[0].Length ||
                    pgrad[0][0].Length != grad[0][0].Length || pgrad[0][0][0].Length != grad[0][0][0].Length)
                    throw new Exception($"pgrad({pgrad.Length}, {pgrad[0].Length}, {pgrad[0][0].Length}, {pgrad[0][0][0].Length}, {pgrad[0][0][0][0].Length}) != " +
                            $"grad({grad.Length}, {grad[0].Length}, {grad[0][0].Length}, {grad[0][0][0].Length}, {grad[0][0][0][0].Length}).");
                // Sum gradient to pgrad
                Ensum(pgrad, grad);
            }

            /// <summary>
            /// sum += add
            /// </summary>
            /// <param name="sum">Inplace sum.</param>
            /// <param name="add">values to be added</param>
            protected static void Ensum(float[] sum, float[] add)
            {
                for (int i = 0; i < sum.Length; i++)
                    sum[i] += add[i];
            }

            /// <summary>
            /// sum += add
            /// </summary>
            /// <param name="sum">Inplace sum.</param>
            /// <param name="add">values to be added</param>
            protected static void Ensum(float[][] sum, float[][] add)
            {
                for (int i = 0; i < sum.Length; i++)
                    Ensum(sum[i], add[i]);
            }
            /// <summary>
            /// sum += add
            /// </summary>
            /// <param name="sum">Inplace sum.</param>
            /// <param name="add">values to be added</param>
            protected static void Ensum(float[][][] sum, float[][][] add)
            {
                for (int i = 0; i < sum.Length; i++)
                    Ensum(sum[i], add[i]);
            }
            /// <summary>
            /// sum += add
            /// </summary>
            /// <param name="sum">Inplace sum.</param>
            /// <param name="add">values to be added</param>
            protected static void Ensum(float[][][][] sum, float[][][][] add)
            {
                for (int i = 0; i < sum.Length; i++)
                    Ensum(sum[i], add[i]);
            }
            /// <summary>
            /// sum += add
            /// </summary>
            /// <param name="sum">Inplace sum.</param>
            /// <param name="add">values to be added</param>
            protected static void Ensum(float[][][][][] sum, float[][][][][] add)
            {
                for (int i = 0; i < sum.Length; i++)
                    Ensum(sum[i], add[i]);
            }

            /// <summary>
            /// --_connects
            /// </summary>
            /// <returns>_connects</returns>
            public int DecConnects() => --_connects;
            /// <summary>
            /// ++_connects
            /// </summary>
            /// <returns>_connects</returns>
            public int IncConnects() => ++_connects;

            /// <summary>
            /// Clear previous values
            /// </summary>
            /// <param name="auto">New AutoGrad ref</param>
            /// <param name="requiresGrad">If needs grad</param>
            public void Clear(AutoGrad auto = null, bool requiresGrad = false)
            {
                _auto = auto;
                _requiresGrad = requiresGrad || _requiresGrad;
                _ctx = null;
                _BackwardFunc = null;
                _parent = null;
                _grad = null;
                _connects = 0;
            }

            /// <summary>
            /// Get Singleton item.
            /// </summary>
            /// <returns>Singleton item</returns>
            public abstract float Item();

            /// <summary>
            /// Get _data from Node as object.
            /// </summary>
            /// <returns>_data from Node as object</returns>
            public abstract object GetData();

            /// <summary>
            /// Checks for same shape as Tensor.
            /// </summary>
            /// <typeparam name="T">Type of values.</typeparam>
            /// <param name="values">Array to compare.</param>
            /// <param name="check">Tensor to compare.</param>
            /// <returns>Array 'values'</returns>
            public static T[] SameShape<T>(T[] values, BackwardNode check)
                => SameShape(values, (T[])check.GetData());
            /// <summary>
            /// Checks for same shape as Tensor.
            /// </summary>
            /// <typeparam name="T">Type of values.</typeparam>
            /// <param name="values">Array to compare.</param>
            /// <param name="check">Tensor to compare.</param>
            /// <returns>Array 'values'</returns>
            public static T[][] SameShape<T>(T[][] values, BackwardNode check)
                => SameShape(values, (T[][])check.GetData());
            /// <summary>
            /// Checks for same shape as Tensor.
            /// </summary>
            /// <typeparam name="T">Type of values.</typeparam>
            /// <param name="values">Array to compare.</param>
            /// <param name="check">Tensor to compare.</param>
            /// <returns>Array 'values'</returns>
            public static T[][][] SameShape<T>(T[][][] values, BackwardNode check)
                => SameShape(values, (T[][][])check.GetData());
            /// <summary>
            /// Checks for same shape as Tensor.
            /// </summary>
            /// <typeparam name="T">Type of values.</typeparam>
            /// <param name="values">Array to compare.</param>
            /// <param name="check">Tensor to compare.</param>
            /// <returns>Array 'values'</returns>
            public static T[][][][] SameShape<T>(T[][][][] values, BackwardNode check)
                => SameShape(values, (T[][][][])check.GetData());

            /// <summary>
            /// Checks for same shape as Array.
            /// </summary>
            /// <typeparam name="T">Type of values & check.</typeparam>
            /// <param name="values">Array to compare.</param>
            /// <param name="check">Array to compare.</param>
            /// <returns>Array 'values'</returns>
            public static T[] SameShape<T>(T[] values, T[] check)
            {
                if (values.Length != check.Length)
                    throw new Exception($"Invalid shape: got ({values.Length}) it should be ({check.Length}).");
                return values;
            }
            /// <summary>
            /// Checks for same shape as Array.
            /// </summary>
            /// <typeparam name="T">Type of values & check.</typeparam>
            /// <param name="values">Array to compare.</param>
            /// <param name="check">Array to compare.</param>
            /// <returns>Array 'values'</returns>
            public static T[][] SameShape<T>(T[][] values, T[][] check)
            {
                if (values.Length != check.Length || values[0].Length != check[0].Length)
                    throw new Exception($"Invalid shape: got ({values.Length}, {values[0].Length}) " +
                            $"it should be ({check.Length}, {check[0].Length}).");
                return values;
            }
            /// <summary>
            /// Checks for same shape as Array.
            /// </summary>
            /// <typeparam name="T">Type of values & check.</typeparam>
            /// <param name="values">Array to compare.</param>
            /// <param name="check">Array to compare.</param>
            /// <returns>Array 'values'</returns>
            public static T[][][] SameShape<T>(T[][][] values, T[][][] check)
            {
                if (values.Length != check.Length || values[0].Length != check[0].Length || values[0][0].Length != check[0][0].Length)
                    throw new Exception($"Invalid shape: got ({values.Length}, {values[0].Length}, {values[0][0].Length}) " +
                            $"it should be ({check.Length}, {check[0].Length}, {check[0][0].Length}).");
                return values;
            }
            /// <summary>
            /// Checks for same shape as Array.
            /// </summary>
            /// <typeparam name="T">Type of values & check.</typeparam>
            /// <param name="values">Array to compare.</param>
            /// <param name="check">Array to compare.</param>
            /// <returns>Array 'values'</returns>
            public static T[][][][] SameShape<T>(T[][][][] values, T[][][][] check)
            {
                if (values.Length != check.Length || values[0].Length != check[0].Length ||
                    values[0][0].Length != check[0][0].Length || values[0][0][0].Length != check[0][0][0].Length)
                    throw new Exception($"Invalid shape: got ({values.Length}, {values[0].Length}, {values[0][0].Length}, {values[0][0][0].Length}) " +
                            $"it should be ({check.Length}, {check[0].Length}, {check[0][0].Length}, {check[0][0][0].Length}).");
                return values;
            }

            /// <summary>
            /// Random Uniform with values ranging from -1 => 1
            /// </summary>
            /// <param name="dim">Dimension of output array</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Array of Random Uniform with values ranging from -1 => 1</returns>
            public static Tensor<float[]> Randn(int dim, bool requiresGrad = false)
            {
                var vals = Rand2.Randn(dim);
                return new Tensor<float[]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }
            /// <summary>
            /// Random Uniform with values ranging from -1 => 1
            /// </summary>
            /// <param name="dim0">Dimension (0) of output array</param>
            /// <param name="dim1">Dimension (1) of output array</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Array of Random Uniform with values ranging from -1 => 1</returns>
            public static Tensor<float[][]> Randn(int dim0, int dim1, bool requiresGrad = false)
            {
                var vals = Rand2.Randn(dim0, dim1);
                return new Tensor<float[][]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }
            /// <summary>
            /// Random Uniform with values ranging from -1 => 1
            /// </summary>
            /// <param name="dim0">Dimension (0) of output array</param>
            /// <param name="dim1">Dimension (1) of output array</param>
            /// <param name="dim2">Dimension (2) of output array</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Array of Random Uniform with values ranging from -1 => 1</returns>
            public static Tensor<float[][][]> Randn(int dim0, int dim1, int dim2, bool requiresGrad = false)
            {
                var vals = Rand2.Randn(dim0, dim1, dim2);
                return new Tensor<float[][][]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }

            /// <summary>
            /// Random Uniform with values ranging from 0 => 1
            /// </summary>
            /// <param name="dim">Dimension of output array</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Array of Random Uniform with values ranging from 0 => 1</returns>
            public static Tensor<float[]> RandUniform(int dim, bool requiresGrad = false)
            {
                var vals = Rand2.Uniform(dim);
                return new Tensor<float[]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }
            /// <summary>
            /// Random Uniform with values ranging from 0 => 1
            /// </summary>
            /// <param name="dim0">Dimension (0) of output array</param>
            /// <param name="dim1">Dimension (1) of output array</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Array of Random Uniform with values ranging from 0 => 1</returns>
            public static Tensor<float[][]> RandUniform(int dim0, int dim1, bool requiresGrad = false)
            {
                var vals = Rand2.Uniform(dim0, dim1);
                return new Tensor<float[][]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }
            /// <summary>
            /// Random Uniform with values ranging from 0 => 1
            /// </summary>
            /// <param name="dim0">Dimension (0) of output array</param>
            /// <param name="dim1">Dimension (1) of output array</param>
            /// <param name="dim2">Dimension (2) of output array</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Array of Random Uniform with values ranging from 0 => 1</returns>
            public static Tensor<float[][][]> RandUniform(int dim0, int dim1, int dim2, bool requiresGrad = false)
            {
                var vals = Rand2.Uniform(dim0, dim1, dim2);
                return new Tensor<float[][][]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }

            /// <summary>
            /// Create a Tensor of Zeros
            /// </summary>
            /// <param name="dim">dimension of Output Tensor</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Tensor of zeros as a value</returns>
            public static Tensor<float[]> Zeros(int dim, bool requiresGrad = false)
            {
                var vals = Utilz.Zeros(dim);
                return new Tensor<float[]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }
            /// <summary>
            /// Create a Tensor of Zeros
            /// </summary>
            /// <param name="dim0">dimension (0) of Output Tensor</param>
            /// <param name="dim1">dimension (1) of Output Tensor</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Tensor of zeros as a value</returns>
            public static Tensor<float[][]> Zeros(int dim0, int dim1, bool requiresGrad = false)
            {
                var vals = Utilz.Zeros(dim0, dim1);
                return new Tensor<float[][]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }
            /// <summary>
            /// Create a Tensor of Zeros
            /// </summary>
            /// <param name="dim0">dimension (0) of Output Tensor</param>
            /// <param name="dim1">dimension (1) of Output Tensor</param>
            /// <param name="dim2">dimension (2) of Output Tensor</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Tensor of zeros as a value</returns>
            public static Tensor<float[][][]> Zeros(int dim0, int dim1, int dim2, bool requiresGrad = false)
            {
                var vals = Utilz.Zeros(dim0, dim1, dim2);
                return new Tensor<float[][][]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }

            /// <summary>
            /// Create a Tensor of Ones
            /// </summary>
            /// <param name="dim">dimension of Output Tensor</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Tensor of ones as a value</returns>
            public static Tensor<float[]> Ones(int dim, bool requiresGrad = false)
            {
                var vals = Utilz.Ones(dim);
                return new Tensor<float[]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }
            /// <summary>
            /// Create a Tensor of Ones
            /// </summary>
            /// <param name="dim0">dimension (0) of Output Tensor</param>
            /// <param name="dim1">dimension (1) of Output Tensor</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Tensor of ones as a value</returns>
            public static Tensor<float[][]> Ones(int dim0, int dim1, bool requiresGrad = false)
            {
                var vals = Utilz.Ones(dim0, dim1);
                return new Tensor<float[][]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }
            /// <summary>
            /// Create a Tensor of Ones
            /// </summary>
            /// <param name="dim0">dimension (0) of Output Tensor</param>
            /// <param name="dim1">dimension (1) of Output Tensor</param>
            /// <param name="dim2">dimension (2) of Output Tensor</param>
            /// <param name="requiresGrad">If Tensor needs a gradient</param>
            /// <returns>Tensor of ones as a value</returns>
            public static Tensor<float[][][]> Ones(int dim0, int dim1, int dim2, bool requiresGrad = false)
            {
                var vals = Utilz.Ones(dim0, dim1, dim2);
                return new Tensor<float[][][]>(vals, null, BranchNode<object>.Blank, null, null, requiresGrad);
            }

            /// <summary>
            /// Parent of Tensor(T) for backwards pass.
            /// </summary>
            /// <param name="ctx">Context for backwards pass.</param>
            /// <param name="backFunc">Backwards Function</param>
            /// <param name="auto">AutoGrad ref</param>
            /// <param name="parent">Parent of node.</param>
            /// <param name="requiresGrad">If gradient is needed</param>
            public BackwardNode(object ctx, BckF backFunc, AutoGrad auto, BackwardNode parent, bool requiresGrad)
            {
                _ctx = ctx;
                _BackwardFunc = backFunc;
                _auto = auto;
                _parent = parent;
                _requiresGrad = requiresGrad;
            }

            // Explicit casts of Tensor ..................................
            public static explicit operator float(BackwardNode node)
                => (float)node.GetData();
            public static explicit operator float[](BackwardNode node)
                => (float[])node.GetData();
            public static explicit operator float[][](BackwardNode node)
                => (float[][])node.GetData();
            public static explicit operator float[][][](BackwardNode node)
                => (float[][][])node.GetData();
            public static explicit operator float[][][][](BackwardNode node)
                => (float[][][][])node.GetData();
            public static explicit operator int[](BackwardNode node)
                => (int[])node.GetData();
            public static explicit operator int[][](BackwardNode node)
                => (int[][])node.GetData();
        }

        

        
    }
}
