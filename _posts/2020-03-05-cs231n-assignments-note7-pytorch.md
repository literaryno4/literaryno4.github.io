---
layout: post
title:  "CS231n Assignments Note7: PyTorch"

categories: jekyll update
---

Table of Contents:

- [What is PyTorch?](#pytorch)

- [How to Use?](#how)

  - [Barebones PyTorch](#bb)

  - [PyTorch Module API](#pm)

  - [PyTorch Sequential API](#ps)(Recommended)

- [Reference](#rf)

<a name='torch'></a>

## What is PyTorch?

After take a deep insight of neural networks, it is pleasant to know that we, in practice, need not write the implement of backward pass and there are automatic differentiation engines coming with deep learning frameworks like PyTorch.

In other words, it is a **replacement for NumPy** to use the power of GPUs, and a **deep learning research platform** that provides maximum flexibility and speed ([source](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)).

<a name='how'></a>

## How to Use?

Of course, the best way to use a readymade API is to refer to [The Official Docs](https://pytorch.org/docs/stable/index.html) and [The Official Tutorials](https://pytorch.org/tutorials/).

There is another [tutorial](https://github.com/jcjohnson/pytorch-examples) recommended as well.

A general workflow look like this:

[CONSTRUCT A NETWORK CLASS] -> [INITIALIZE PARAMETERS] -> [TRAIN MODELS USING LOOPS]

Next, we look through three different method to to that.

<a name='bb'></a>

### Barebones PyTorch

Take three layer ConvNet as an example:

``` python

def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?
    
    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None

    conv1 = F.conv2d(x, weight=conv_w1, bias=conv_b1, padding=2)
    relu1 = F.relu(conv1)
    conv2 = F.conv2d(relu1, weight=conv_w2, bias=conv_b2, padding=1)
    relu2 = F.relu(conv2)
    relu2_flat = flatten(relu2)
    scores = relu2_flat.mm(fc_w) + fc_b
    
    return scores

```

``` python

def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

```
def train_part2(model_fn, params, learning_rate):
    """
    Train a model on CIFAR-10.
    
    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD
    
    Returns: Nothing
    """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()

```

After implement these, we train a model:

``` python

learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

conv_w1 = None
conv_b1 = None
conv_w2 = None
conv_b2 = None
fc_w = None
fc_b = None

conv_w1 = random_weight((channel_1, 3, 5, 5))
conv_b1 = zero_weight((channel_1,))
conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = zero_weight((channel_2,))
fc_w = random_weight((channel_2 * 32 * 32, 10))
fc_b = zero_weight(10)

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
train_part2(three_layer_convnet, params, learning_rate)

```

<a name='pm'></a>

### PyTorch Module API

We can also use module API as well:

``` python

class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, channel_1, (5, 5), padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        #self.relu = nn.Relu(conv1)
        self.conv2 = nn.Conv2d(channel_1, channel_2, (3, 3), padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        #self.relu = nn.Relu(conv2)
        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        scores = None
        
        conv1 = self.conv1(x)
        relu1 = F.relu(conv1)
        conv2 = self.conv2(relu1)
        relu2 = F.relu(conv2)
        relu2_flat = flatten(relu2)
        scores = self.fc(relu2_flat)
 
        return scores

```

``` python

def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()

```

``` python

learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

model = None
optimizer = None

model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_part34(model, optimizer)

```

<a name='ps'></a>

### PyTorch Sequential API

The **most recommended** method is to use Sequential API like this:

``` python

channel_1 = 32
channel_2 = 16
learning_rate = 1e-2

model = None
optimizer = None

model = nn.Sequential(
    nn.Conv2d(3, channel_1, (5, 5), padding=2),
    nn.ReLU(),
    nn.Conv2d(channel_1, channel_2, (3, 3), padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2 * 32 * 32, 10)
)

optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                     momentum=0.9, nesterov=True)

train_part34(model, optimizer) # implemented above. 

```

<a name='rf'></a>

## Reference

[CS231n](http://cs231n.stanford.edu/)

[literaryno4/cs231n](https://github.com/literaryno4/cs231n)

[PyTorch Tutorials](https://pytorch.org/tutorials/)

[PyTorch Docs](https://pytorch.org/docs/stable/index.html)

[PyTorch Examples](https://github.com/jcjohnson/pytorch-examples)

[Assignment #2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets]( http://cs231n.github.io/assignments2019/assignment2/)
