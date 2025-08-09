# My C# Deep Learning AI Library More like PyTorch & TensorFlow

![Neon Gradient Flow Chart](./imgs/NeonGradientFlowchart.gif)

***Mini should really be Mighty!***

This library closer represents the cutting edge library in PyTorch.

This library utilizes what is known as ***Auto Grad***. Just define the forward pass and train the network!

This library also supports ***Multithreading***. It chunks the dataset to the number of cores on your CPU. See
```csharp 
NetModule.TrainIt()
```

The last time I checked (which was awhile ago) C# did not have many good Deep Learning libraries. The one that comes to mind is NeuronDotNet. Not very flexible what so ever! You define some feed forward layers and the end!

This library gives you two different ***transformers***, Conv1d, Linear, Gate ... Also, you have ***functions***: Concatenate, Split, Reshape, Repeat, Take, Transpose ...

I did want to flex my muscles implementing fully commented Deep learning code. Please have a look!

## Simple Example 1 & 2

### Simple Example 1

```csharp
// Get Gradients through one iteration ......................
// Y values: [0.7952096, 0.09830558, -1]
// labels values: [-1, 0.6272376, 0.5339667]
var y = Tensor<object>.Randn(3, requiresGrad: true);
var labels = Rand2.Randn(3);
// Loss & Backwards pass
y = new AutoGrad().AddRoot(y);
var loss = (y - labels).Abs().Sum();
loss.Backward();
// Gradients: [1, -1, -1]
y._grad
```

### Simple Example 2

```csharp
// Train Network through one iteration .....................
// Initialize Network
var net = new TestLinear(smoothBatchLen:1);
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
```

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

***Conv1dSparse***: Multiple Conv1d's are stacked ontop of each other over a sequence.

***Embedding***: usually but not always used in Natural Language Processing.

***Gate***: First half of inputs are gated against the second half of inputs.
Example: 
```csharp
value[i]*wt[i] + value[i+length/2]*(1-wt[i])
```

***Linear***: Linear or Fully Connected Deep Layer.

***LinearSparse***: Stacked Linear Layers by size 'dim.'

## Module & NetModule\<T\>

***Module***: like PyTorch is the class you inherit from to create a layer with parameters or nest modules and so on.

***NetModule\<T\>*** inherits from Module. You inherit this module when you have the final network to train and do inference. It has extra functions for training and inference. You must override the method
```csharp
Tensor<T> _Fwd(Tensor<object>)
```
for you forward pass.

## Tensor\<T\>

Used like python frameworks. One can use operators + - * /. Also, have functions of Mean(), Sum(), and Abs(). These functions and operator *ALL* use Auto Grad.

<code>newTensor = tensorA + tensorB</code>
Example of Addition. Tensors must be of the same float[1d-4d] arrays and shape. 
Supported operators: + - * /

<code>newTensor = TensorA - float[1d-4d]</code>
Example of Subtraction of a float array. Tensor must be float[1d-4d] and float[1d-4d] of the same shape.
Supported operators: + - * /

<code>newTensor = TensorA * singleNumber</code>
Example of Multiplication of a single number. 'singlenumber': can be of type double, float, or int.
Supported operators: + - * /

## Mac & Linux OS

This Library is written in .NET 4.7. However I ported it with Visual Studio Community 2022 to .NET 7. It should work with Mac, Linux, and Windows.

1) Install if you haven't already, .NET SDK. Open with Visual Studio Code and install the C# Dev Kit extension if you haven't already.
2) Open & restore
Open the repo’s folder in VS Code (not just a file). VS Code will detect the solution/project; when prompted, Restore to pull NuGet packages. You can also run:
```csharp
dotnet restore
dotnet build
```
3) Press Run and Debug (or F5) and pick C#: Launch when asked—VS Code will generate the needed launch config. 

![Neural Network Flowchart Animation](imgs/NeuralNetworkFlowchartAnimation.gif)

## Example 2

```csharp
public class Net : NetModule<float[][]>
{
    // Norms
    public readonly NormInf _normx, _normy;
    // Layers
    readonly TransConv1d _trans1, _trans2, _ixtrans;
    readonly LinearSparse _fcs;
    readonly BatchNormBack _normfc;

    // Must override _Fwd() Function for Forward pass
    public override Tensor<float[][]> _Fwd(Tensor<object> x)
    {
        return Foward(Tensor<object>.CastFloat2d(x));
    }

    // All you have to do is this (Define the Forward pass) !! I use it!
    // You don't have to define the backwards pass.
    // Again, You do not need to define the forward pass & backwards pass explicitly
    public Tensor<float[][]> Foward(Tensor<float[][]> x)
    {
        // Split into 2 one dim1
        var split = Function.Split.Fwd(x, 1, 12, 12);
        x = split[0];
        var xix = split[1];
        // Transformer 1
        var y = _trans1.Forward(x);
        // Take 4 on dim1
        y = Function.Take.Fwd(y, _fcs._dim, 1);
        // Flatten dim1 & dim2
        var y2 = Function.Flatten.Fwd(y, 1);
        // Transformer 2
        y = _trans2.Forward(y2);
        // Take 4 on dim1
        y = Function.Take.Fwd(y, _fcs._dim, 1);
        // Transformer 3
        var yix = _ixtrans.Forward(xix);
        // Take 4 on dim1
        yix = Function.Take.Fwd(yix, _fcs._dim, 1);
        // Concatenate y & yix on dim2
        y = Function.Cat.Fwd(2, y, yix);
        // BatchNorm on dim -1
        y = _normfc.Forward(y);
        // Reshape
        var x3 = Function.T.Fwd(Function.Reshape.Fwd(x, -1, 4, 3));     // Transpose dim1 & dim0
        y = Function.T.Fwd(y);                                          // Transpose dim1 & dim0
        // Concatenate x3 & y on dim2
        y = Function.Cat.Fwd(2, x3, y);
        // Linear Sparse
        y = Function.T.Fwd(_fcs.Forward(y));                            // Transpose dim1 & dim0
        // Flatten dim1 & dim2
        return Function.Flatten.Fwd(y, 1);
    }

    // Example Net using TransConv1d & LinearSparse
    public Net(int nInput = 24, int nOut = 8, int smoothBatchLen = 30, string name0 = null)
        : base(smoothBatchLen, name0, "Net")
    {
        _normx = new NormInf(nInput);
        _normy = new NormInf(nOut);
        // TransConv1d
        _trans1 = new TransConv1d(nInput / 2, 7, 6, 3, batchNorm: false, name0:ToString());
        _trans2 = new TransConv1d(nInput / 2, 7, 6, 3, name0: ToString());
        _ixtrans = new TransConv1d(nInput / 2, 7, 6, 3, name0: ToString());
        // Linear Sparse
        _fcs = new LinearSparse(9, 4, 2, name0: ToString());
        // Batch Norm
        _normfc = new BatchNormBack(6, 4, name0: ToString());
        _modules.AddRange(new[] { _normx, _normy });
        _modules.Add(_trans1);
        _modules.Add(_trans2);
        _modules.Add(_ixtrans);
        _modules.Add(_fcs);
        _modules.Add(_normfc);
        // Be sure to call this at the end of the constuctor
        GetAllModules(_modules);
    }
}
```