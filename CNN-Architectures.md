**LeNet**

![Image for post](media/image1.png){width="6.268055555555556in"
height="1.734430227471566in"}

LeNet-5 was used on large scale to automatically classify hand-written
digits on bank cheques in the USA.

This network is a First convolutional neural network (CNN).

These networks are built upon 3 main ideas:

1.  local receptive fields,

2.  shared weights and

3.  spacial subsampling.

Local receptive fields with shared weights are the essence of the
convolutional layer and most architectures described below use
convolutional layers in one form or another.

Another reason why LeNet is an important architecture is that before it
was invented, character recognition had been done mostly by using
feature engineering by hand, followed by a machine learning model to
learn to classify hand engineered features.

LeNet made hand engineering features redundant, because the network
learns the best internal representation from raw images automatically.

Description

By modern standards, LeNet-5 is a very simple network.

It only has 7 layers, among which there are;

-   3 convolutional layers (C1, C3 and C5),

-   2 sub-sampling (pooling) layers (S2 and S4), and

-   1 fully connected layer (F6), that are followed by the

-   output layer. Convolutional layers use 5 by 5 convolutions with
    stride 1.

Sub-sampling layers are 2 x 2 average pooling layers.

Tanh sigmoid activations are used throughout the network.

There are several interesting architectural choices that were made in
LeNet-5 that are not very common in the modern era of deep learning.

-   First, individual convolutional kernels in the layer C3 do not use
    all of the features produced by the layer S2, which is very unusual
    by today's standard.

    -   One reason for that is to made the network less computationally
        demanding.

    -   The other reason was to make convolutional kernels learn
        different patterns.

This makes perfect sense: if different kernels receive different inputs,
they will learn different patterns.

-   Second, the output layer uses 10 Euclidean Radial Basis Function
    neurons that compute L2 distance between the input vector of
    dimension 84 and **manually predefined weights vectors** of the same
    dimension.

The number 84 comes from the fact that essentially the weights represent
a 7x12 binary mask, one for each digit.

This forces network to transform input image into an internal
representation that will make outputs of layer F6 as close as possible
to hand-coded weights of the 10 neurons of the output layer.

LeNet-5 was able to achieve error rate below 1% on the MNIST data set,
which was very close to the state of the art at the time (produced by a
boosted ensemble of three LeNet-4 networks).

**LeNet-1**

![Image for post](media/image2.png){width="6.268055555555556in"
height="2.077514216972878in"}In Lenet-1,\
**28×28 input image \>**\
**Four 24×24 feature maps convolutional layer (5×5 size) \>\
Average Pooling layers (2×2 size) \>\
Eight 12×12 feature maps convolutional layer (5×5 size) \>\
Average Pooling layers (2×2 size) \>\
Directly fully connected to the output**

With convolutional and subsampling/pooling layers introduced, LeNet-1
got the **error rate **of** 1.7%** on test data

It is noted that, at the moment authors invented the LeNet, they used
average pooling layer, output the average values of 2×2 feature maps.
Right now, many LeNet implementation use max pooling that only the
maximum value from 2×2** **feature maps is output, and it turns out that
it can help for speeding up the training. As the strongest feature is
chosen, larger gradient can be obtained during back-propagation.

LeNet-4![Image for post](media/image3.png){width="6.268055555555556in"
height="2.015298556430446in"}**32×32 input image \>**\
**Four 28×28 feature maps convolutional layer (5×5 size) \>\
Average Pooling layers (2×2 size) \>\
Sixteen 10×10 feature maps convolutional layer (5×5 size) \>\
Average Pooling layers (2×2 size) \>\
Fully connected to 120 neurons \>\
Fully connected to 10 outputs**

With more feature maps, and one more fully connected layers, **error
rate** is **1.1%** on test data.

**LeNet-5**![Image for
post](media/image4.png){width="6.268055555555556in"
height="1.7971347331583551in"}

LeNet-5, the most popular LeNet people talked about, only has slight
differences compared with LeNet-4.

**32×32 input image \>**\
**Six 28×28 feature maps convolutional layer (5×5 size) \>\
Average Pooling layers (2×2 size) \>\
Sixteen 10×10 feature maps convolutional layer (5×5 size) \>\
Average Pooling layers (2×2 size) \>\
Fully connected to 120 neurons \>\
Fully connected to 84 neurons \>\
Fully connected to 10 outputs**

With more feature maps, and one more fully connected layers, **error
rate** is **0.95%** on test data.

Boosted LeNet-4

![Image for post](media/image5.png){width="6.268055555555556in"
height="4.770378390201225in"}

Boosting is a technique to combine the results from several/many weak
classifiers to get a more accurate results. In LeNet-4, the outputs of
three LeNet-4 are simply added together, the one with maximum value
would be the predicted classification class. And there is an enhancement
that when the first net has a high confidence answer, the other nets
would not be called.

With boosting, the **error rate** on test data is **0.7%** which is even
smaller than that of LeNet-5.

This boosting technique has been used for years, until now.

Summary on LeNet

1.  **LeNet-1: 1.7%**

2.  **LeNet-4: 1.1%**

3.  **LeNet-5: 0.95%**

4.  **Boosted LeNet-4: 0.7%**

We can see that, the error rate is reducing while adding more deep
learning components or some machine learning techniques.

LeNet-5 layers:
---------------

1.  Convolution \#1. Input = 32x32x1. Output = 28x28x6 conv2d

2.  SubSampling \#1. Input = 28x28x6. Output = 14x14x6. SubSampling is
    > simply Average Pooling so we use avg_pool

3.  Convolution \#2. Input = 14x14x6. Output = 10x10x16 conv2d

4.  SubSampling \#2. Input = 10x10x16. Output = 5x5x16 avg_pool

5.  Fully Connected \#1. Input = 5x5x16. Output = 120

6.  Fully Connected \#2. Input = 120. Output = 84

7.  Output 10

Having seen the architecture schema and the formula above, we can go
over each layer of LeNet-5.

1.  **Layer 1 (C1)**: The first convolutional layer with 6 kernels of
    > size 5×5 and the stride of 1. Given the input size (32×32×1), the
    > output of this layer is of size 28×28×6.

2.  **Layer 2 (S2)**: A subsampling/pooling layer with 6 kernels of size
    > 2×2 and the stride of 2. The subsampling layer in the original
    > architecture was a bit more complex than the traditionally used
    > max/average pooling layers. I will quote \[1\]: " The four inputs
    > to a unit in S2 are added, then multiplied by a trainable
    > coefficient, and added to a trainable bias. The result is passed
    > through a sigmoidal function.". As a result of non-overlapping
    > receptive fields, the input to this layer is halved in size
    > (14×14×6).

3.  **Layer 3 (C3):** The second convolutional layer with the same
    > configuration as the first one, however, this time with 16
    > filters. The output of this layer is 10×10×16.

4.  **Layer 4 (S4):** The second pooling layer. The logic is identical
    > to the previous one, but this time the layer has 16 filters. The
    > output of this layer is of size 5×5×16.

5.  **Layer 5 (C5):** The last convolutional layer with 120 5×5 kernels.
    > Given that the input to this layer is of size 5×5×16 and the
    > kernels are of size 5×5, the output is 1×1×120. As a result,
    > layers S4 and C5 are fully-connected. That is also why in some
    > implementations of LeNet-5 actually use a fully-connected layer
    > instead of the convolutional one as the 5th layer. The reason for
    > keeping this layer as a convolutional one is the fact that if the
    > input to the network is larger than the one used in \[1\] (the
    > initial input, so 32×32 in this case), this layer will not be a
    > fully-connected one, as the output of each kernel will not be 1×1.

6.  **Layer 6 (F6):** The first fully-connected layer, which takes the
    > input of 120 units and returns 84 units. In the original paper,
    > the authors used a custom activation function --- a variant of
    > the *tanh* activation function. For a thorough explanation, please
    > refer to Appendix A in \[1\].

7.  **Layer 7 (F7):** The last dense layer, which outputs 10 units. In
    > \[1\], the authors used Euclidean Radial Basis Function neurons as
    > activation functions for this layer.

**\
**

**AlexNet**

AlexNet is the name given to a **Convolutional Neural Network
Architecture** that won the LSVRC competition in **2012**.

LSVRC** (**Large Scale Visual Recognition Challenge) is a competition
where research teams evaluate their algorithms on a huge dataset of
labeled images (**ImageNet**) and compete to achieve higher accuracy on
several visual recognition tasks. This made a huge impact on how teams
approach the completion afterward.

The Architecture of AlexNet

![Image for post](media/image6.png){width="6.268055555555556in"
height="2.162236439195101in"}

The **AlexNet contains 8 layers** with weights;

**5 convolutional layers**

**3 fully connected layers**.

At the end of each layer, ReLu activation is performed except for the
last one, which outputs with a softmax with a distribution over the 1000
class labels. Dropout is applied in the first two fully connected
layers. As the figure from the above shows also applies Max-pooling
after the first, second, and fifth convolutional layers. The kernels of
the second, fourth, and fifth convolutional layers are connected only to
those kernel maps in the previous layer, which reside on the same GPU.
The kernels of the third convolutional layer are connected to all kernel
maps in the second layer. The neurons in the fully-connected layers are
connected to all neurons in the previous layer.

Main ideas
==========

ReLU nonlinearity, training on multiple GPUs, local response
normalization, overlapping pooling, data augmentation, dropout

Why it is important
===================

AlexNet won the ImageNet competition in 2012 by a large margin. It was
the biggest network at the time. The network demonstrated the potential
of training large neural networks quickly on massive datasets using
widely available gaming GPUs; before that neural networks had been
trained mainly on CPUs. AlexNet also used novel ReLU activation, data
augmentation, dropout and local response normalization. All of these
allowed to achieve state-of-the art performance in object recognition in
2012.

Brief description
=================

**ReLU nonlinearity**
---------------------

Experimenting with layers helped them a lot. The next thing they
targetted was the sort of non-linear transformation that data would have
to go through when entering a neuron.

AlexNet team chose a non linear activation function with the
non-linearity being a **Rectified Linear Unit (ReLU)**. They claimed
that it ran much faster than **TanH** the more popular choice for
linearity at the time.

AlexNet team chose a non linear activation function with the
non-linearity being a **Rectified Linear Unit (ReLU)**. They claimed
that it ran much faster than **TanH** the more popular choice for
linearity at the time.

![Image for post](media/image7.png){width="3.5780905511811025in"
height="5.597222222222222in"}

The questions is why would we transform our data and that too in a non-linear fashion!
--------------------------------------------------------------------------------------

AlexNet is a Convolutional Neural Network. Hence it is bound to be made
up of neurons. These b*iologically inspired neural networks* possess an
activation function which decides whether the input stimulus is enough
for a neuron to fire --- i.e. get activated.

Without the non-linearity introduced by the activation function,
multiple layers of a neural network are equivalent to a single layer
neural network --- the 8 layer depth would be useless without this.

The keyword being faster. Above all AlexNet needed a faster training
time and ReLU helped them. But they needed something more. Something
that could transform the speed with which CNNs were computed. This is
where the GPUs figured.

ReLU is a so-called *non-saturating activation*. This means that
gradient will never be close to zero for a positive activation and as
result, the training will be faster.

By contrast, sigmoid activations are *saturating*, which makes gradient
close to zero for large absolute values of activations. Very small
gradient will make the network train slower or even stop, because the
step size during gradient descent's weight update will be small or zero
(so-called **vanishing gradient problem**).

By employing ReLU, training speed of the network was **six times
faster** as compared to classical sigmoid activations that had been
popular before ReLU. Today, ReLU is the default choice of activation
function.

The questions is why would we transform our data and that too in a non-linear fashion!
--------------------------------------------------------------------------------------

AlexNet is a Convolutional Neural Network. Hence it is bound to be made
up of neurons. These b*iologically inspired neural networks* possess an
activation function which decides whether the input stimulus is enough
for a neuron to fire --- i.e. get activated.

Without the non-linearity introduced by the activation function,
multiple layers of a neural network are equivalent to a single layer
neural network --- the 8 layer depth would be useless without this.

The keyword being faster. Above all AlexNet needed a faster training
time and ReLU helped them. But they needed something more. Something
that could transform the speed with which CNNs were computed. This is
where the GPUs figured.

GPUs and Training Time
----------------------

GPUs are devices that can perform parallel computations. Remember how an
average laptop is either a *Quadcore(4 cores) *or an *Octacore(8
cores). *This refers to the number of parallel computations that can
happen in a processor. A GPU can have 1000s of cores leading to a lot of
parallelization. AlexNet made use of a GPU that NVIDIA launched a year
before AlexNet came out.

The noticeable thing was that AlexNet made use of 2 GPUs in parallel
which made their design extremely fast.

Local response normalization
----------------------------

Not only did they need to speed up the processing they also needed the
data to be *balanced. *So they used **local response normalization
(LRN)**. It basically helps you normalize your data. AlexNet employed
LRN to aid generalization. Response normalization reduced their top-1
and top-5 error rates by 1.4% and 1.2%, respectively.

The biological equivalent of LRN is called "lateral inhibition". This
refers to the capacity of an excited neuron to subdue its neighbors. The
neuron does that to increase the contrast in its surroundings, thereby
increasing the sensory perception for that particular are.

![Image for post](media/image8.png){width="5.840277777777778in" height="1.1078248031496063in"}
----------------------------------------------------------------------------------------------

Local response normalization formula from the paper

After layers C1 and C2, activities of neurons were normalized according
to the formula above. What this did is scaled the activities down by
taking into account 5 neuron activities at preceding and following
feature channels at the same spatial position.

![Image for post](media/image9.png){width="2.748571741032371in"
height="2.0555555555555554in"}

An example of local response normalization

These activities were squared and used together with
parameters *n*, *k*, *alpha* and *beta* to scale down each neuron's
activity. Authors argue that this created "competition for big
activities amongst neuron outputs computed using different kernels".
This approach reduced top-1 error by 1%. In the table above you can see
an example of neuron activations scaled down by using this approach.
Also note that the values of *n*, *k*, *alpha* and *beta* were selected
using cross-validation.

Overlapping pooling
-------------------

Every CNN has pooling as an essential step. Up until 2012 most pooling
schemes involved non-overlapping pools of pixels. AlexNet was ready to
experiment with this part of the process.

Pooling is the process of picking a patch of s x s pixels and finding
its max or mean.

Traditionally, these patches were non-overlapping i.e. once
an ***s** x **s ***patch is used you don't touch these pixels again and
move on to the next ***s** x **s ***patch. They realized that
overlapping pooling reduced the top-1 and top-5 error rates by 0.4% and
0.3%, respectively, as compared with the non-overlapping scheme.

![Image for post](media/image10.gif){width="2.5416666666666665in"
height="2.6944444444444446in"}

Overlapped Pooling

AlexNet used max pooling of size 3 and stride 2. This means that the
largest values were pooled from 3x3 regions, centers of these regions
being 2 pixels apart from each other vertically and horizontally.
Overlapping pooling reduced tendency to overfit and also reduced test
error rates by 0.4% and 0.3% (for top-1 and top-5 error
correspondingly).

Overfitting Prevention

Having tackled normalization and pooling AlexNet was faced with a huge
overfitting challenge. Their 60-million parameter model was bound to
overfit. They needed to come up with an overfitting prevention strategy
that could work at this scale.

Whenever a system has huge number of parameters, it becomes prone to
overfitting.

Overfitting --- Given a question that you've already seen you can answer
perfectly but you'll perform poorly on unseen questions.

They employed two methods to battle overfitting

-   Data Augmentation

-   Dropout

Data augmentation
-----------------

Data augmentation is increasing the size of your dataset by creating
transforms of each image in your dataset. These transforms can be simple
scaling of size or reflection or rotation.

![Image for post](media/image11.png){width="6.268055555555556in"
height="1.5748490813648295in"}

See how no. 6 is rotated in various directions

![Image for post](media/image12.jpeg){width="6.268055555555556in"
height="2.532937445319335in"}

These schemes led to an error reduction of 1% in their top-1 error
metric. By augmenting the data you not only increase the dataset but the
model tries to become rotation invariant, color invariant etc. and
prevents overfitting

Data augmentation is a regularization strategy (a way to prevent
overfitting). AlexNet uses two data augmentation approaches.

The first takes random crops of input images, as well as rotations and
flips and uses them as inputs to the network during training. This
allows to vastly increase the size of the data; the authors mention the
increase by the factor of 2048. Another benefit is the fact that
augmentation is performed on the fly on CPU while the GPUs train
previous batch of data. In other words, this type of augmentation is
essentially computationally free, and also does not require to store
augmented images on disk.

The second data augmentation strategy is so-called **PCA color
augmentation**. First, PCA on all pixels of ImageNet training data set
is performed (a pixel is treated as a 3-dimensional vector for this
purpose). As result, we get a 3x3 covariance matrix, as well as 3
eigenvectors and 3 eigenvalues. During training, a random intensity
factor based on PCA components is added to each color channel of an
image, which is equivalent to changing intensity and color of
illumination. This scheme reduces top-1 error rate by over 1% which is a
significant reduction.

Test time data augmentation
---------------------------

The authors do not explicitly mention this as contribution of their
paper, but they still employed this strategy. During test time, 5 crops
of original test image (4 corners and center) are taken as well as their
horizontal flips. Then predictions are made on these 10 images.
Predictions are averaged to make the final prediction. This approach is
called **test time augmentation **(TTA). Generally, it does not need to
be only corners, center and flips, any suitable augmentation will work.
This improves testing performance and is a very useful tool for deep
learning practitioners.

Dropout
-------

The second technique that AlexNet used to avoid overfitting was dropout.
It consists of setting to zero the output of each hidden neuron with
probability 0.5. The neurons which are "dropped out" in this way do not
contribute to the forward pass and do not participate in [back-
propagation](https://medium.com/x8-the-ai-community/how-to-train-your-d%CC%B6r%CC%B6a%CC%B6g%CC%B6o%CC%B6neural-net-backpropagation-intiution-3fc575ec7f3d).
So every time an input is presented, the neural network samples a
different architecture.

This *new-architecture-everytime* is akin to using multiple
architectures without expending additional resources. The model,
therefore, forced to learn more robust features.

![Image for post](media/image13.gif){width="3.6805555555555554in"
height="1.8803805774278215in"}

Dropout in action.

AlexNet used 0.5 dropout during training. This means that during forward
pass, 50% of all activations of the network were set to zero and also
did not participate in backpropagation. During testing, all neurons were
active and were not dropped. Dropout reduces "complex co-adaptations" of
neurons, preventing them to depend heavily on other neurons being
present. Dropout is a very efficient regularization technique that makes
the network learn more robust internal representations, significantly
reducing overfitting.

Architecture
------------

![Image for post](media/image14.png){width="6.268055555555556in"
height="2.195858486439195in"}

AlexNet architecture from paper.

Architecture itself is relatively simple. There are 8 trainable layers:
5 convolutional and 3 fully connected. ReLU activations are used for all
layers, except for the output layer, where softmax activation is used.
Local response normalization is used only after layers C1 and C2 (before
activation). Overlapping max pooling is used after layers C1, C2 and C5.
Dropout was only used after layers F1 and F2.

Due to the fact that the network resided on 2 GPUs, it had to be split
in 2 parts that communicated only partially. Note that layers C2, C4 and
C5 only received as inputs outputs of preceding layers that resided on
the same GPU. Communication between GPUs only happened at layer C3 as
well as F1, F2 and the output layer.

The network was trained using stochastic gradient descent with momentum
and learning rate decay. In addition, during training, learning rate was
decreased manually by the factor of 10 whenever validation error rate
stopped improving.

Pros of AlexNet

1.  AlexNet is considered as the milestone of CNN for image
    > classification.

2.  Many methods, such as the conv + pooling design, dropout, GPU,
    > parallel computing, ReLU, are still the industrial standard for
    > computer vision.

3.  The unique advantage of AlexNet is the direct image input to the
    > classification model.

4.  The convolution layers can automatically extract the edges of the
    > images and fully connected layers learning these features

5.  Theoreticallythecomplexityofvisualpatternscanbeeffectiveextractedbyaddingmoreconvlayer

Cons of AlexNet

1.  AlexNet is NOT deep enough compared to the later model, such as
    > VGGNet, GoogLENet, and ResNet.

2.  The use of large convolution filters (5\*5) is not encouraged
    > shortly after that.

3.  Use normal distribution to initiate the weights in the neural
    > networks, can not effectively solve the problem of gradient
    > vanishing, replaced by the Xavier method later.

4.  The performance is surpassed by more complex models such as
    > GoogLENet (6.7%), and ResNet (3.6%)

ZFNet(2013)

Not surprisingly, the ILSVRC 2013 winner was also a CNN which became
known as ZFNet. It achieved a top-5 error rate of 14.8% which is now
already half of the prior mentioned non-neural error rate. It was mostly
an achievement by tweaking the hyper-parameters of AlexNet while
maintaining the same structure with additional Deep Learning elements as
discussed earlier in this essay.

![Image for post](media/image15.png){width="6.268055555555556in"
height="1.543561898512686in"}

VGGNet

VGG is an acronym for the Visual Geometric Group from Oxford University
and **VGG-16** is a network with 16 layers proposed by the Visual
Geometric Group. These 16 layers contain the trainable parameters and
there are other layers also like the Max pool layer but those do not
contain any trainable parameters. This architecture was the 1st runner
up of the Visual Recognition Challenge of 2014
i.e. ***ILSVRC-2014 ***and was developed
by* **Simonyan ***and ***Zisserman***.

The VGG research group released a series of the convolution network
model starting from VGG11 to VGG19. The main intention of the VGG group
on depth was to understand how the depth of convolutional networks
affects the accuracy of the models of large-scale image classification
and recognition. The minimum VGG11 has 8 convolutional layers and 3
fully connected layers as compared to the maximum VGG19 which has 16
convolutional layers and the 3 fully connected layers. The different
variations of VGGs are exactly the same in the last three fully
connected layers. The overall structure includes 5 sets of convolutional
layers, followed by a MaxPool. But the difference is that as the depth
increases that is as we move from VGG11 to VGG19 more and more cascaded
convolutional layers are added in the five sets of convolutional layers.

The below-shown figure is the overall network configuration of different
models created by VGG that uses the same principle but only varies in
depth.

**The Kernel size is 3x3 and the pool size is 2x2 for all the layers.\
**The input to the Vgg 16 model is 224x224x3 pixels images. then we have
two convolution layers with each 224x224x64 size, then we have a pooling
layer which reduces the height and width of the image to 112x112x64.

Then we have two conv128 layers with each 112x112x128 size after that we
have a pooling layer which again reduces the height and width of the
image to 56x56x128.

Then we have three conv256 layers with each 56x56x256 size, after that
again a pooling layer reduces the image size to 28x28x256.

Then we have three conv512 layers with each 28x28x512 size, after that
again a pooling layer reduces the image size to 14x14x512 =.

Then again we have three conv512 layers with each 14x14x521 layers,
after that, we have a pooling layer with 7x7x521 and then we have two
dense or fully-connected layers with each of 4090 nodes. and at last, we
have a final dense or output layer with 1000 nodes of the size which
classify between 1000 classes of image net.

![Image for post](media/image16.png){width="6.268055555555556in"
height="5.615461504811899in"}

**Image from Original Paper- Reference \[1\]**

From the above comparison table that represents a different network, we
can see that as the model moves from simpler to complex the depth of the
network is getting increased. This is the best way to solve any problem,
means to say, solve the problem using a simpler model and then gradually
optimize it by making it complex.

The number of trainable parameters in different models can be seen from
the following figure:

![Image for post](media/image17.png){width="5.277777777777778in"
height="0.9166666666666666in"}

**Image from Original Paper- Reference \[1\]**

Here we'll explore the architecture of VGG-16 deeply.

![Image for post](media/image18.png){width="6.268055555555556in"
height="3.98205927384077in"}

**The architecture of VGG-16 --- Image from Researchgate.net**

In the figure above, all the blue rectangles represent the convolution
layers along with the non-linear activation function which is a
rectified linear unit (or ReLU). As can be seen from the figure that
there are 13 blue and 5 red rectangles i.e there are 13 convolution
layers and 5 max-pooling layers. Along with these, there are 3 green
rectangles representing 3 fully connected layers. So, the total number
of layers having tunable parameters is 16 of which 13 is for convolution
layers and 3 for fully connected layers, thus the name is given as
VGG-16. At the output, we have a softmax layer having 1000 outputs per
image category in the imagenet dataset.

In this architecture, we have started with a very low channel size of 64
and then gradually increased by a factor of 2 after each max-pooling
layers, until it reaches 512.

The flattened architecture of VGG-16 is as shown below:

![Image for post](media/image19.png){width="6.268055555555556in"
height="2.4245756780402448in"}

**Image by Author**

The architecture is very simple. It has got 2 contiguous blocks of 2
convolution layers followed by a max-pooling, then it has 3 contiguous
blocks of 3 convolution layers followed by max-pooling, and at last, we
have 3 dense layers. The last 3 convolution layers have different depths
in different architectures.

The important thing to analyze here is that after every max-pooling the
size is getting half.

Features of VGG-16 network
==========================

1.  **Input Layer: **It accepts color images as an input with the size
    > 224 x 224 and 3 channels i.e. Red, Green, and Blue.

2.  **Convolution Layer: **The images pass through a stack of
    > convolution layers where every convolution filter has a very small
    > receptive field of 3 x 3 and stride of 1. Every convolution kernel
    > uses row and column padding so that the size of input as well as
    > the output feature maps remains the same or in other words, the
    > resolution after the convolution is performed remains the same.

3.  **Max pooling: **It is performed over a max-pool window of size 2 x
    > 2 with stride equals to 2, which means here max pool windows are
    > non-overlapping windows.

4.  Not every convolution layer is followed by a max pool layer as at
    > some places a convolution layer is following another convolution
    > layer without the max-pool layer in between.

5.  The first two fully connected layers have 4096 channels each and the
    > third fully connected layer which is also the output layer have
    > 1000 channels, one for each category of images in the imagenet
    > database.

6.  The hidden layers have ReLU as their activation function.

***NOTE:** An important thing to observe here is we have use kernel size
of 3 x 3 which is the least possible size for capturing the notion of
left/right, up/down, and center. Also, a stack of two 3 × 3 convolution
layers (without spatial pooling or max-pooling in between) has an
effective receptive field of 5×5 and the same three 3 × 3 convolution
layers has an effective receptive field of 7 × 7.*

**Let's understand this mathematically:**
-----------------------------------------

The formula involved in calculating the output size from each
convolution layer is given as- \[(N-f)/S\] + 1

Suppose we have an input of a shape of 224 x 224 with K channels i.e.
224 x 224 x K. We'll apply convolution with different sizes kernels with
stride =1.

Case-1: When we have a kernel size of 3 x 3
-------------------------------------------

-   **After the First convolution**

N = 224, f = 3, S = 1

Output shape** =** \[(N-f)/S\] + 1** =** \[(224--3)/1\] + 1 **=** 222

-   **After the Second convolution**

N = 222, f = 3, S = 1

Output shape** =** \[(N-f)/S\] + 1** =** \[(222--3)/1\] + 1 **=** 220

-   **After the Third convolution**

N = 220, f = 3, S = 1

Output shape** =** \[(N-f)/S\] + 1** =** \[(220--3)/1\] + 1 **=** 218

So, after three simultaneous convolutions, we got an output of
size **218 x 218 x K**

Case-2: When we have a kernel size of 7 x 7
-------------------------------------------

N = 224, f = 7, S = 1

Output shape** =** \[(N-f)/S\] + 1** =** \[(224--7)/1\] + 1 **=** 218

So, after one convolution only we got an output of size **218 x 218 x
K**

Hence, looking at the above two cases we say that the three 3 × 3
convolution layers have an effective receptive field of 7 × 7.

Advantages of having 3 x 3 kernel size
--------------------------------------

1.  As we know more the layers of convolution more sharply the features
    > will be extracted from our input as compared to when we have fewer
    > layers. So having 3 x 3 kernel size would lead to much better
    > feature extraction than 7 x 7 kernel size.

2.  When we take 3 x 3 kernel size the number of trainable parameters
    > will be 27K² as compared to 7 x 7 kernel size when taken gives
    > 49K² trainable parameters which is 81% more.

Calculations involved in getting output size from each layer
------------------------------------------------------------

The complete architecture of the VGG-16 has been summed up in the table
shown below:

![Image for post](media/image20.png){width="6.268055555555556in"
height="2.5877898075240595in"}***Input Layer*:**

-   The size of the input image is **224 x 224**.

***Convolution Layer - 1*:**

-   **Input size = N =** 224

-   **Filter size = f = **3 x 3

-   **No. of filters =** 64

-   **Strides = S = **1

-   **Padding = P = **1

-   **Output feature map size =** \[(224--3+2)/1\] + 1 **= **224

-   **Output with channels = **224 x 224 x 64

***Convolution Layer - 2*:**

-   **Input size = N =** 224

-   **Filter size = f = **3 x 3

-   **No. of filters =** 64

-   **Strides = S = **1

-   **Padding = P = **1

-   **Output feature map size =** \[(224--3+2)/1\] + 1 **= **224

-   **Output with channels = **224 x 224 x 64

***Max-Pooling Layer - 1*:**

-   **Input size = N =** 224

-   **Filter size = f = **2 x 2

-   **Strides = S = **2

-   **Padding = P =** 0

-   **Output feature map size =** \[(224--2+0)/2\] + 1 **= **112

-   **Output with channels = **112 x 112 x 64

***Convolution Layer - 3*:**

-   **Input size = N =** 112

-   **Filter size = f = **3 x 3

-   **No. of filters =** 128

-   **Strides = S = **1

-   **Padding = P = **1

-   **Output feature map size =** \[(112--3+2)/1\] + 1 **= 112**

-   **Output with channels = **112 x 112 x 128

***Convolution Layer - 4*:**

-   **Input size = N =** 112

-   **Filter size = f = **3 x 3

-   **No. of filters =** 128

-   **Strides = S = **1

-   **Padding = P = **1

-   **Output feature map size =** \[(112--3+2)/1\] + 1 **= **112

-   **Output with channels = **112 x 112 x 128

***Max-Pooling Layer - 2*:**

-   **Input size = N =** 112

-   **Filter size = f = **2 x 2

-   **Strides = S = **2

-   **Padding = P =** 0

-   **Output feature map size =** \[(112--2+0)/2\] + 1 **= **56

-   **Output with channels = **56 x 56 x 128

***Convolution Layer - 5*:**

-   **Input size = N =** 56

-   **Filter size = f = **3 x 3

-   **No. of filters =** 256

-   **Strides = S = **1

-   **Padding = P = **1

-   **Output feature map size =** \[(56--3+2)/1\] + 1 **= **56

-   **Output with channels = **56 x 56 x 256

***Convolution Layer - 6*:**

-   **Input size = N =** 56

-   **Filter size = f = **3 x 3

-   **No. of filters =** 256

-   **Strides = S = **1

-   **Padding = P = **1

-   **Output feature map size =** \[(56--3+2)/1\] + 1 **= **56

-   **Output with channels = **56 x 56 x 256

***Convolution Layer - 7*:**

-   **Input size = N =** 56

-   **Filter size = f = **3 x 3

-   **No. of filters =** 256

-   **Strides = S = **1

-   **Padding = P = **1

-   **Output feature map size =** \[(56--3+2)/1\] + 1 **= **56

-   **Output with channels = **56 x 56 x 256

***Max-Pooling Layer - 3*:**

-   **Input size = N =** 56

-   **Filter size = f = **2 x 2

-   **Strides = S = **2

-   **Padding = P =** 0

-   **Output feature map size =** \[(56--2+0)/2\] + 1 **= **28

-   **Output with channels = **28 x 28 x 256

**Similar calculations will be performed for the rest of the network.**

Difference between VGG-16 and AlexNet
=====================================

1.  As compared to VGG-16 where all the convolution kernels are of the
    > uniform size 3 x 3 with stride as 1, the AlexNet has convolution
    > kernels of variable size like 5 x 5 and 3 x 3.

Though AlexNet uses multiple kernels of different sizes, the realization
of every convolution kernel of different sizes can be done using
multiple 3 x 3 size kernels.

**VGG-16** and **VGG-19** ![Image for
post](media/image21.png){width="6.268055555555556in"
height="4.793218503937008in"}

**Different VGG Layer Structures Using Single Scale (256) Evaluation**

To obtain the optimum deep learning layer structure, ablation study has
been done as shown in the above figure.

1.  First of all, **VGG-11 already obtains 10.4% error rate**, which is
    > similar to that of ZFNet in ILSVRC 2013. VGG-11 is set as
    > benchmark.

2.  **VGG-11 (LRN) obtains 10.5% error rate, is the one with additional
    > local response normalization (LRN) **operation suggested by
    > AlexNet. **By comparing VGG-11 and VGG-11 (LRN), the error rate
    > doesn't improve which means LRN is not useful. **In fact, LRN is
    > not used any more in later on deep learning network, instead,
    > batch normalization (BN) is used.

3.  **VGG-13** **obtains 9.9% error rate, which means the additional
    > conv helps the classification accuracy**.

4.  **VGG-16 (Conv1) obtains 9.4% error rate, which means the additional
    > three 1×1 conv layers help the classification accuracy**. **1×1
    > conv actually helps to increase non-linearlity of the decision
    > function. **Without changing the dimensions of input and
    > output, **1×1 conv is doing the projection mapping in the same
    > high dimensionality. **This technique is essential in a paper
    > called "Network in Network" \[7\] and also in the GoogLeNet \[8\]
    > (the winner of ILSVRC 2014) and ResNet \[9\] (the winner of ILSVRC
    > 2015). I will talk more about GoogLeNet and ResNet review stories
    > in the coming future.

5.  **VGG-16 obtains 8.8% error rate which means the deep learning
    > network is still improving by adding number of layers.**

6.  **VGG-19 obtains 9.0% error rate which means the deep learning
    > network is NOT improving by adding number of layers. **Thus,
    > authors stop adding layers.

By observing the addition of layers one by one, we can observe
that **VGG-16 and VGG-19 start converging** and the accuracy improvement
is slowing down. When people are talking about VGGNet, they usually
mention VGG-16 and VGG-19.

**GoogLeNet**

In the convolutional neural networks prior to the InceptionNet primarily
focused on increasing the depth of the network to extract feature of
features, for improving the learning capabilities of the model. The
developers of the InceptionNet were the first to focus on increasing the
width and the depth of the model simultaneously to attain better
accuracy while keeping the computing resources constant.

GoogleLeNet is a 22 layer deep network that was first iteration of
InceptionNet version series to be built using the Inception module. The
ideology behind the Inception module was that neurons that extract
features together should learn together.

Most of the earlier iterations of the convolutional architectures
focused on varying the kernel size to extract best features. On the
other hand, the InceptionNet architecture focusses on parallel
processing and extraction of various feature maps concurrently. This is
the primarily attribute of the InceptionNet that differentiates it from
all the other image classification models.

If ResNet was all about going deeper, the Inception Family™ is all about
going wider. In particular, the authors of Inception were interested in
the computational efficiency of training larger nets. In other
words: *how can we scale up neural nets without increasing computational
cost?*

The original paper focused on a new building block for deep nets, a
block now known as the "Inception module." At its core, this module is
the product of two key insights.

The first insight relates to layer operations. In a traditional conv
net, each layer extracts information from the previous layer in order to
transform the input data into a more useful representation. However,
each layer type extracts a different kind of information. The output of
a 5x5 convolutional kernel tells us something different from the output
of a 3x3 convolutional kernel, which tells us something different from
the output of a max-pooling kernel, and so on and so on. At any given
layer, how do we know what transformation provides the most "useful"
information?

Insight \#1: why not let the model choose?

An Inception module computes *multiple different transformations* over
the same input map* *in parallel, concatenating their results into a
single output. In other words, for each layer, Inception does a 5x5
convolutional transformation, *and* a 3x3, *and* a max-pool. And the
next layer of the model gets to decide if (and how) to use each piece of
information.

![Image for post](media/image22.jpeg){width="6.268055555555556in"
height="4.054259623797026in"}

The increased information density of this model architecture comes with
one glaring problem: we've drastically increased computational costs.
Not only are large (e.g. 5x5) convolutional filters inherently expensive
to compute, stacking multiple different filters side by side greatly
increases the number of feature maps per layer. And this increase
becomes a deadly bottleneck in our model.

Think about it this way. For each additional filter added, we have to
convolve over *all* the input maps to calculate a single output. See the
image below: creating one output map from a single filter involves
computing over *every single map* from the previous layer.

![Image for post](media/image23.png){width="4.625in"
height="3.173611111111111in"}

Let's say there are *M* input maps. One additional filter means
convolving over *M* more maps; *N* additional filters means convolving
over *N\*M* more maps. In other words, as the authors note, "any uniform
increase in the number of \[filters\] results in a quadratic increase of
computation." Our naive Inception module just tripled or quadrupled the
number of filters. Computationally speaking, this is a Big Bad Thing.

This leads to insight \#2: using 1x1 convolutions to perform
dimensionality reduction. In order to solve the computational
bottleneck, the authors of Inception used 1x1 convolutions to "filter"
the depth of the outputs. A 1x1 convolution only looks at one value at a
time, but across multiple channels, it can extract spatial information
and compress it down to a lower dimension. For example, using 20 1x1
filters, an input of size 64x64x100 (with 100 feature maps) can be
compressed down to 64x64x20. By reducing the number of input maps, the
authors of Inception were able to stack different layer transformations
in parallel, resulting in nets that were simultaneously deep (many
layers) and "wide" (many parallel operations).

Inception module with naive version

The above depicted Inception module simultaneously performs 1 \* 1
convolutions, 3 \* 3 convolutions, 5 \* 5 convolutions, and 3 \* 3 max
pooling operations. Thereafter, it sums up the outputs from all the
operations in a single place and builds the next feature. The
architecture does not follow Sequential model approach where every
operation such as pooling or convolution is performed one after the
other.

As the inception module extracts a different kind of data or information
from every convolution or pooling operation different features are
extracted from each operation. For instance, 1 \* 1 convolutions and 3
\* 3 convolutions will generate different information. After the
individual operations have been performed simultaneously all the
extracted data will be combined into a single feature map with all the
properties. This will in turn increase the accuracy of the model as it
will focus on multiple features simultaneously. The output dimension of
all the extracted feature maps will be different as the kernel size for
every operation will not be the same. These different feature maps
generated through different operations are concatenated together using
padding operation, which will make he output dimension of every
operation the same.

Architectural Details
=====================

One big problem with this stacked inception module is that even a modest
number of 5×5 convolutions would be prohibitively expensive on top of a
convolutional layer with numerous filters. This problem becomes even
more pronounced once pooling units are added. Even while the
architecture might cover the optimal sparse structure, it would do that
very inefficiently; the merging of the output of the pooling layer with
the outputs of convolutional layers would definitely lead to a
computational blow up within a few stages.

Thus, authors borrowed Network-in-Network architecture which was
proposed by [Lin et al.](https://arxiv.org/pdf/1312.4400.pdf) to
increase the representational power of neural networks. It can be viewed
as an additional 1 × 1 convolutional layer followed typically by the
ReLU activation. Authors applied it in forms of

-   **dimension reductions** --- *1×1 convolutions used for computing
    > reductions before the expensive 3×3 and 5×5 convolutions*

-   **projections** --- *1×1 convolutions used for shielding a large
    > number of input filters of the last stage to the next after
    > max-pooling*

wherever the computational requirements would increase too much
(computational bottlenecks). This allows for not just increasing the
depth, but also the width of our networks without a significant
performance penalty.

Inception module with dimension reduction

The Inception module with dimension reduction works in a similar manner
as the naïve one with only one difference. Here features are extracted
on a pixel level using 1 \* 1 convolutions before the 3 \* 3
convolutions and 5 \* 5 convolutions. When the 1 \* 1 convolution
operation is performed the dimension of the image is not changed.
However, the output achieved offers better accuracy.

The below images show the math intuition of the inception module.

Formula for output image

**Output Image Shape = ((Input Image Shape -- Kernel Size)/Stride)+1**

Math intution for 1 \* 1 convolution operation

**1 x 1 convolution for inception module with 200 x 200 as input image
shape**

**=((200 -- 1)/1)+1 = 200**

Math intution for 3 \* 3 convolution operation

**3 x 3 convolution for inception module with 200 x 200 as input image
shape**

**=((200 -- 3)/1)+1 = 198**

Math intution for 5 \* 5 convolution operation

**5 x 5 convolution for inception module with 200 x 200 as input image
shape**

**=((200 -- 5)/1)+1 = 196**

Maths intuition for dimension reduction for the inception module

![Image for post](media/image24.jpeg){width="4.208333333333333in"
height="4.598980752405949in"}

Four Parallel Channel Processing

**1 \* 1 Convolution Operation:**

The input feature map can be reduced in dimension and upgraded without
too much loss of input separation information. This operation has no
receptive field as it gathers data on a pixel level.

**3 \* 3 Convolution Operation:**

The operation increases the receptive field of the feature map. This
allows the kernel to gather information regarding various shapes and
sizes.

**5 \* 5 Convolution Operation:**

The operation further increases the receptive field of the feature map.

**3 \* 3 Max Pooling:**

The pooling layer will lose space information. However, it will be
effectively applied on various space fields, increasing the
effectiveness of the four-channel parallel processing.

While implementing various operations simultaneously we might lose
certain information or dimensions. But, it is completely fine as if one
convolution operation does give a certain feature than the other
operation will.

**Disadvantage:**

Larger model models using InceptionNet are prone to overfit especially
with limited number of label samples. The model can be biased towards
certain classes that have labels present in high volume than the other.

**Versions of InceptionNet**

The Inception network on the other hand, was complex (heavily
engineered). It used a lot of tricks to push performance; both in terms
of speed and accuracy. Its constant evolution lead to the creation of
several versions of the network. The popular versions are as follows:

-   Inception v1.

-   Inception v2

-   Inception v3.

-   Inception v4

Each version is an iterative improvement over the previous one.
Understanding the upgrades can help us to build custom classifiers that
are optimized both in speed and accuracy.

Inception v1
============

This is where it all started. Let us analyze what problem it was
purported to solve, and how it solved it.

The Premise:
------------

-   **Salient parts** in the image can have extremely **large
    > variation** in size. For instance, an image with a dog can be
    > either of the following, as shown below. The area occupied by the
    > dog is different in each image.

```{=html}
<!-- -->
```
-   Because of this huge variation in the location of the information,
    > choosing the **right kernel size** for the convolution operation
    > becomes tough. A **larger kernel** is preferred for information
    > that is distributed more **globally**, and a **smaller kernel** is
    > preferred for information that is distributed more** locally.**

-   **Very deep networks** are prone to **overfitting**. It also hard to
    > pass gradient updates through the entire network.

-   Naively stacking large convolution operations is **computationally
    > expensive**.

The Solution:
-------------

Why not have filters with **multiple sizes** operate on the **same
level**? The network essentially would get a bit "**wider**" rather than
"deeper". The authors designed the inception module to reflect the same.

The below image is the "naive" inception module. It
performs **convolution** on an input, with **3 different sizes of
filters **(1x1, 3x3, 5x5). Additionally, **max pooling** is also
performed. The outputs are **concatenated** and sent to the next
inception module.

![Image for post](media/image25.png){width="6.268055555555556in"
height="3.4805905511811024in"}

As stated before, deep neural networks are **computationally
expensive**. To make it cheaper, the authors **limit** the number
of **input channels** by adding an **extra 1x1 convolution** before the
3x3 and 5x5 convolutions. Though adding an extra operation may seem
counterintuitive, 1x1 convolutions are far more cheaper than 5x5
convolutions, and the reduced number of input channels also help. Do
note that however, the 1x1 convolution is introduced after the max
pooling layer, rather than before.

![Image for post](media/image26.png){width="6.268055555555556in"
height="3.9435454943132107in"}

Using the dimension reduced inception module, a neural network
architecture was built. This was popularly known as GoogLeNet (Inception
v1). The architecture is shown below:

![Image for post](media/image27.png){width="6.268055555555556in"
height="2.276081583552056in"}

GoogLeNet. The orange box is the stem, which has some preliminary
convolutions. The purple boxes are auxiliary classifiers. The wide parts
are the inception modules.

GoogLeNet has 9 such inception modules stacked linearly. It is 22 layers
deep (27, including the pooling layers). It uses global average pooling
at the end of the last inception module.

Needless to say, it is a pretty **deep classifier**. As with any very
deep network, it is subject to the **vanishing gradient problem**.

To prevent the **middle part** of the network from "**dying out**", the
authors introduced **two auxiliary classifiers** (The purple boxes in
the image). They essentially applied softmax to the outputs of two of
the inception modules, and computed an **auxiliary loss **over the same
labels. The **total loss function** is a **weighted sum** of
the **auxiliary loss** and the **real loss**. Weight value used in the
paper was 0.3 for each auxiliary loss.

\# The total loss used by the inception net during training.\
**total_loss = real_loss + 0.3 \* aux_loss_1 + 0.3 \* aux_loss_2**

Needless to say, auxiliary loss is purely used for training purposes,
and is ignored during inference.

Inception v2
============

**Inception v2** and **Inception v3** were presented in
the **same paper**. The authors proposed a number of upgrades which
increased the accuracy and reduced the computational complexity.
Inception v2 explores the following:

The Premise:
------------

-   Reduce representational bottleneck. The intuition was that, neural
    > networks perform better when convolutions didn't alter the
    > dimensions of the input drastically. Reducing the dimensions too
    > much may cause loss of information, known as a "representational
    > bottleneck"

-   Using smart factorization methods, convolutions can be made more
    > efficient in terms of computational complexity.

The Solution:
-------------

-   **Factorize 5x5** convolution **to two 3x3 **convolution operations
    > to improve computational speed. Although this may seem
    > counterintuitive, a 5x5 convolution is **2.78 times more
    > expensive** than a 3x3 convolution. So stacking two 3x3
    > convolutions infact leads to a boost in performance. This is
    > illustrated in the below image.

-   ![Image for post](media/image28.png){width="2.7847222222222223in"
    > height="2.544601924759405in"}

The left-most 5x5 convolution of the old inception module, is now
represented as two 3x3 convolutions.

-   Moreover, they **factorize** convolutions of filter size **nxn **to
    > a **combination **of** 1xn and nx1** convolutions. For example, a
    > 3x3 convolution is equivalent to first performing a 1x3
    > convolution, and then performing a 3x1 convolution on its output.
    > They found this method to be **33% more cheaper** than the single
    > 3x3 convolution. This is illustrated in the below image.

-   ![Image for post](media/image29.png){width="2.064724409448819in"
    > height="2.4583333333333335in"}

Here, put n=3 to obtain the equivalent of the previous image. The
left-most 5x5 convolution can be represented as two 3x3 convolutions,
which inturn are represented as 1x3 and 3x1 in series.

-   The **filter banks** in the module were **expanded** (made wider
    > instead of deeper) to remove the representational bottleneck. If
    > the module was made deeper instead, there would be excessive
    > reduction in dimensions, and hence loss of information. This is
    > illustrated in the below image.

-   ![Image for post](media/image30.png){width="2.58752624671916in"
    > height="2.050832239720035in"}

Making the inception module wider. This type is equivalent to the module
shown above.

-   The above three principles were used to build three different types
    > of inception modules (Let's call them modules** A,B** and **C** in
    > the order they were introduced. These names are introduced for
    > clarity, and not the official names). The architecture is as
    > follows:

-   ![Image for post](media/image31.png){width="4.2125in"
    > height="3.6666546369203847in"}

Here, "figure 5" is module A, "figure 6" is module B and "figure 7" is
module C.

Inception v3
============

The Premise
-----------

-   The authors noted that the **auxiliary classifiers** didn't
    > contribute much until near the end of the training process, when
    > accuracies were nearing saturation. They argued that they function
    > as **regularizes**, especially if they have BatchNorm or Dropout
    > operations.

-   Possibilities to improve on the Inception v2 without drastically
    > changing the modules were to be investigated.

The Solution
------------

-   **Inception Net v3** incorporated all of the above upgrades stated
    > for Inception v2, and in addition used the following:

1.  RMSProp Optimizer.

2.  Factorized 7x7 convolutions.

3.  BatchNorm in the Auxillary Classifiers.

4.  Label Smoothing (A type of regularizing component added to the loss
    > formula that prevents the network from becoming too confident
    > about a class. Prevents over fitting).

Inception v4
============

**Inception v4** and **Inception-ResNet** were introduced in
the **same paper**. For clarity, let us discuss them in separate
sections.

The Premise
-----------

-   Make the modules more **uniform**. The authors also noticed that
    > some of the modules were **more complicated than necessary**. This
    > can enable us to boost performance by adding more of these uniform
    > modules.

The Solution
------------

-   The "**stem**" of Inception v4 was **modified**. The stem here,
    > refers to the initial set of operations performed before
    > introducing the Inception blocks.

-   ![Image for post](media/image32.jpeg){width="3.8027777777777776in"
    > height="2.9315529308836394in"}

The top image is the stem of Inception-ResNet v1. The bottom image is
the stem of Inception v4 and Inception-ResNet v2.

-   They had three main inception modules, named A,B and C (Unlike
    > Inception v2, these modules are infact named A,B and C). They look
    > very similar to their Inception v2 (or v3) counterparts.

-   ![Image for post](media/image33.jpeg){width="5.226388888888889in"
    > height="1.5221358267716536in"}

(From left) Inception modules A,B,C used in Inception v4. Note how
similar they are to the Inception v2 (or v3) modules.

-   Inception v4 introduced specialized "**Reduction Blocks**" which are
    > used to change the width and height of the grid. The earlier
    > versions didn't explicitly have reduction blocks, but the
    > functionality was implemented.

-   ![Image for post](media/image34.jpeg){width="4.501594488188976in"
    > height="2.1604155730533683in"}

(From Left) Reduction Block A (35x35 to 17x17 size reduction) and
Reduction Block B (17x17 to 8x8 size reduction). Refer to the paper for
the exact hyper-parameter setting (V,l,k).
