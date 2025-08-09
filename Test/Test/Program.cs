using MiniNet;

/******
TEST PROJECT FOR MININET:
=============================================================

My C# Deep Learning AI Library More like PyTorch & TensorFlow

***Mini should really be Mighty!***

This library closer represents the cutting edge library in PyTorch.

This library utilizes what is known as ***Auto Grad***. Just define the forward pass and train the network!

This library also supports ***Multithreading***. It chunks the dataset to the number of cores on your CPU. See <code>NetModule.TrainIt()</code>

The last time I checked (which was awhile ago) C# did not have many good Deep Learning libraries. The one that comes to mind is NeuronDotNet. Not very flexible what so ever! You define some feed forward layers and the end!

This library gives you two different ***transformers***, Conv1d, Linear, Gate ... Also, you have ***functions***: Concatenate, Split, Reshape, Repeat, Take, Transpose ...

I did want to flex my muscles implementing fully commented Deep learning code. Please have a look!

## Functions

This library has various functions:

***T***: Transpose dim0 & dim1.
***Reshape***: Reshape dimensions into a different shape.
***Sqrt***: Bi-Square root activation function. Negative values can be used & will return a negative Square Root value.
***Matmul***: Matrix Multiplication.
***Take***: Takes the first number of values on a dimension.
***Flatten***: Flattens two dimensions into one.
***Cat***: Concatenates values on specific dimension.
***Repeat***: Repeats values on Specific dimensions.
***Split***: Splits values on a specific dimension to multiple values.

## Layers (Modules)

This library has various Layers:

***TransConv***: My Transformer Convolutional 1d Layer. Very close to a Transformer layer, but using Conv1d layers instead of fully connected layers.
***TransConv1d***: My Interesting interpretation of combining a Transformer & Convolutional 1d layer.
***BatchNormBack***: Batch Norm on the last dimension.
***Conv1d***: My version of convolution 1D. Convolution is usually used for images, but I'm using it in place of a fully connected layer.
*Conv1dSparse*: Multiple Conv1d's are stacked ontop of each other over a sequence.
***Embedding***: usually but not always used in Natural Language Processing.
***Gate***: First half of inputs are gated against the second half of inputs.
        Example: value[i]*wt[i] + value[i+length/2]*(1-wt[i])
***Linear***: Linear or Fully Connected Deep Layer.
***LinearSparse***: Stacked Linear Layers by size 'dim.'
 *****/

// Train Network through one iteration .....................
{
    // Initialize Network
    var net = new TestLinear(smoothBatchLen: 1);
    // X values: [[-1, 0.1721811, -0.5875599]]
    // labels values: [[-0.9335992, -1]]
    var x = Rand2.Randn(1, 3);
    var labels = Rand2.Randn(1, 2);
    // Forward Pass Y values: [[-0.02937686, -0.07825255]]
    var y = net.Forward(x);
    // Regular Criterion loss value: 0.912984848
    var loss = Criterion.Regular(y, labels);
    Console.WriteLine(loss.Item());
    // Backward pass
    loss.Backward();
    // Update Parameters of Network
    net.Step(lr: 1e-3f);
}

// Get Gradients through one iteration ......................
{
    // Y values: [0.7952096, 0.09830558, -1]
    // labels values: [-1, 0.6272376, 0.5339667]
    var y = Tensor<object>.Randn(3, requiresGrad: true);
    var labels = Rand2.Randn(3);
    // Loss & Backwards pass
    y = new AutoGrad().AddRoot(y);
    var loss = (y - labels).Abs().Sum();
    loss.Backward();
    // Gradients: [1, -1, -1]
    //y._grad
}

// Test Linear Layer
{
    var test = new MiniNet.TestLinear();
    test.Test();
}
// Test Conv1d Layer
{
    var test = new MiniNet.TestConv();
    test.Test();
}
// Test Embedding Layer
{
    var test = new MiniNet.TestEmbedding();
    test.Test();
}
// Test Gate Layer
{
    var test = new MiniNet.NetGate();
    test.Test();
}
// Test Network
// Example Net using TransConv1d & LinearSparse
{
    var test = new MiniNet.Net();
    test.Test();
}