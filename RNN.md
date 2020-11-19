**RNN**

The main difference between normal neural nets and RNNs is that RNNs
have two 'dimensions' --- time t (i.e. along the sequence length) and
the depth l(the usual layers). In fact, in RNNs it is somewhat
incomplete to say '*the output at layer *l'; we rather say '*the output
at layer *l* and time *t*' .(time* here is not necessarily timestamp but
the position ).

The most crucial idea of RNNs which makes them suitable for sequence
problems is that:

The state of the network updates itself as it sees new elements in the
sequence.This is the core idea of an RNN --- it updates what it has
learnt as it sees new inputs in the sequence. The 'next state' is
influenced by the previous one, and so it can learn the dependence
between the subsequent states which is a characteristic of sequence
problems.

Equation for RNN
================

Let's quickly recall the feedforward equations of a normal neural
network:

![Image for post](media/image1.png){width="1.5486111111111112in"
height="0.6875in"}

where,

Wl : weight matrix at layer l

bl : bias at layer l , zl : input into layer l, fl : activation function
at layer l and al : output or activations from layer l.

![Image for post](media/image2.png){width="6.268055555555556in"
height="2.213882327209099in"}

We say that there is a **recurrent **relationship between al(t+1) and
its previous state al(t), and hence the name Recurrent Neural Networks.

![Image for post](media/image3.png){width="6.268055555555556in"
height="1.627652012248469in"}

![Image for post](media/image4.gif){width="6.268055555555556in"
height="4.448297244094488in"}

We will provide input to the hidden layer at each step. A recurrent
neuron now stores all the previous step input and merges that
information with the current step input. Thus it also captures some
information regarding the correlation between current data step and the
previous steps. The decision at a time step t-1 affects the decision
taken at time t.

![Image for post](media/image5.png){width="2.9930555555555554in"
height="3.125in"}

Here, the weights and bias of these hidden layers are different. And
hence each of these layers behave independently and cannot be combined
together. To combine these hidden layers together, we shall have the
same weights and bias for these hidden layers.

![Image for post](media/image6.png){width="1.4375in" height="3.125in"}

We can now combines these layers together, that the weights and bias of
all the hidden layers is the same. All these hidden layers can be rolled
in together in a single recurrent layer.

So they start looking somewhat like this

how various types of sequences are fed to RNNs 🎮-
-------------------------------------------------

Consider the following text string: "A girl walked into a bar, and she
said 'Can I have a drink please?'.

The bartender said 'Certainly {}". There are many options for what could
fill in the {} symbol in the above string, for instance, "miss", "ma'am"
and so on. However, other words could also fit, such as "sir", "Mister"
etc. In order to get the correct gender of the noun, the neural network
needs to "recall" that two previous words designating the likely gender
(i.e. "girl" and "she") were used.

![Image for post](media/image7.png){width="5.125in"
height="2.6944444444444446in"}

For instance, first, we supply the word vector for "A" to the
network *F* --- the output of the nodes in *F *are fed into the "next"
network and also act as a stand-alone output (h0). The next network
(though it is really the same network) *F* at time *t=1* takes the next
word vector for "girl" and the previous output h0 into its hidden nodes,
producing the next output h1 and so on.

![Image for post](media/image8.png){width="6.268055555555556in"
height="2.549260717410324in"}

**We can say instead of giving all input togather like in ANN , we give
the input one by one after another separated by 't' time frame in
sequence.**

RNNs: Simplified Notations🤓
===========================

The RNN feedforward equations are:

![Image for post](media/image9.png){width="4.430555555555555in"
height="2.2708333333333335in"}

![Image for post](media/image10.png){width="6.0in"
height="1.0763888888888888in"}

This form is not only more concise but also more computationally
efficient. Rather than doing two matrix multiplications and adding them,
the network can do one large matrix multiplication.

RNN Architectures Types
=======================

There are four types of Recurrent Neural Networks:

1.  **One to One **: This type of neural network is known as the Vanilla
    > Neural Network. It's used for general machine Example -learning
    > problems, which has a single input and a single output.prediction
    > task like image classification.

2.  **One to Many** : This type of neural network has a single input and
    > multiple outputs. An example of this is the image caption.

This type of architecture is generally used as a **generative model**.
Among popular use of this architecture are applications such as
generating music (given a genre, for example), generating landscape
images given a keyword, generating text given an instruction/topic, etc.

**3. Many to One : **This RNN takes a sequence of inputs and generates a
single output. Sentiment analysis is a good example of this kind of
network where a given sentence can be classified as expressing positive
or negative sentiments.

![Image for post](media/image11.jpeg){width="6.268055555555556in"
height="1.9615179352580927in"}

**4. Many to Many : **This RNN takes a sequence of inputs and generates
a sequence of outputs. It further can be of two types :

-   **Many to Many : Equal input and output length **-In this type of
    > RNN, the input (X) and output (Y) both are a sequence of multiple
    > entities spread over timesteps. In this architecture, the network
    > spits out an output at each timestep. There is a one-to-one
    > correspondence between the input and output at each timestep. You
    > can use this architecture for various tasks.ex-build
    > a** part-of-speech tagger** where each word in the input sequence
    > is tagged with its part-of-speech at every timestep.

-   **Many-to-many RNN: Unequal input and output lengths --- **In the
    > previous many-to-many artitecture, we had assumed that the lengths
    > of the input and output sequences are equal. However, this is not
    > always the case. There are many problems where the **lengths of
    > the input and output sequences are different**. For example,
    > consider the task of machine translation --- the length of a Hindi
    > sentence can be different from the corresponding English
    > sentence.the encoder-decoder architecture is used in tasks where
    > the input and output sequences are of different lengths.

-   ![Image for post](media/image12.jpeg){width="3.5347222222222223in"
    > height="1.655232939632546in"}

The above architecture comprises of two components --- an encoder and a
decoder both of which are RNNs themselves. The output of the encoder,
called the **encoded vector** (and sometimes also the '**context
vector**'), captures a representation of the input sequence. The encoded
vector is then fed to the decoder RNN which produces the output
sequence.

The input and output can now be of different lengths since there is no
one-to-one correspondence between them anymore. This architecture gives
the RNNs much-needed flexibility for real-world applications such as
language translation.

Backpropagation Through Time (BPTT)⏳
====================================

RNNs use a slightly modified version of backpropagation to update the
weights. In a standard neural network, the errors are propagated from
the output layer to the input layer. However, in RNNs, errors are
propagated not only from right to left but also through the time axis.

Backpropagation breaks down in a recurrent neural network, because of
the recurrent or loop connections.This was addressed with a modification
of the Backpropagation technique called Backpropagation Through Time or
BPTT.

![Image for post](media/image13.png){width="2.8819444444444446in"
height="1.9627865266841644in"}

In general we can say -

A recurrent neural network is shown one input each timestep and predicts
one output.Conceptually, **BPTT works by unrolling all input timesteps.
Each timestep has one input timestep, one copy of the network, and one
output. **Errors are then calculated and accumulated for each timestep.
The network is rolled back up and the weights are updated.

**But loss calculation depends on the type of task and the
architecture**.

-   In a **many-to-one** architecture (such as classifying a sentence as
    > correct/incorrect), the loss is simply the difference between the
    > predicted and the actual label. **The loss is computed and
    > backpropagated after the entire sequence has been digested by the
    > network.**

-   ![Image for post](media/image14.png){width="6.268055555555556in"
    > height="1.4553127734033247in"}

```{=html}
<!-- -->
```
-   On the other hand, in a** many-to-many** architecture, the network
    > emits an output at multiple time steps, and the loss is calculated
    > at each time step. The **total loss** (= the sum of the losses at
    > each time step) is propagated back into the network after the
    > entire sequence has been ingested.

-   ![Image for post](media/image15.png){width="6.268055555555556in"
    > height="1.2651126421697287in"}

We can now add the losses for all the sequences (i.e. for a batch of
input sequences) and backpropagate the total loss into the network.

BPTT can be computationally expensive as the number of timesteps
increases.If input sequences are comprised of thousands of timesteps,
then this will be the number of derivatives required for a single update
weight update.

Limitation of RNN
=================

*Training RNNs is considered to be difficult, in order to preserve
long-range dependencies it often meets one of the problems
called **Exploding Gradients **( weights become too large that over-fits
the model ) or** Vanishing Gradients** ( weights become too small that
under-fits the model ). The occurrence of these two problems depends on
the activation functions used in the hidden layer, with the sigmoid
activation function vanishing gradient problem sounds reasonable while
with rectified linear unit exploding gradient make sense.*

![Image for post](media/image16.png){width="6.268055555555556in"
height="3.485805993000875in"}

RNNs are designed to learn patterns in sequential data, i.e. patterns
across 'time'. RNNs are also capable of learning what are
called **long-term dependencies. **For example, in a machine translation
task, we expect the network to learn the interdependencies between the
first and the eighth word, learn the grammar of the languages, etc. This
is accomplished through the **recurrent layers **of the net --- each
state learns the cumulative knowledge of the sequence seen so far by the
network.

Although this feature is what makes RNNs so powerful, it introduces a
severe problem --- as the sequences become longer, it becomes much
harder to backpropagate the errors back into the network. The gradients
'die out' by the time they reach the initial time steps during
backpropagation.

You could still use some workarounds to solve the problem of exploding
gradients. You can impose an upper limit to the gradient while training,
commonly known as **gradient clipping**. By controlling the maximum
value of a gradient, you could do away with the problem of exploding
gradients.

But the problem of vanishing gradients is a more serious one. The
vanishing gradient problem is so rampant and serious in the case of RNNs
that it renders RNNs useless in practical applications.

To solve the vanishing gradients problem, many attempts have been made
to tweak the vanilla RNNs such that the gradients don't die when
sequences get long. The most popular and successful of these attempts
has been the **long, short-term memory network**, or the **LSTM **.
LSTMs have proven to be so effective that they have almost replaced
vanilla RNNs.

Recurrent neural networks (RNNs) are a class of artificial neural
networks which are often used with sequential data. The 3 most common
types of recurrent neural networks are

1.  vanilla RNN,

2.  long short-term memory (LSTM), proposed by Hochreiter and
    > Schmidhuber in 1997, and

3.  gated recurrent units (GRU), proposed by Cho *et. al* in 2014.

What is an RNN?

A recurrent neural network is a neural network that is specialized for
processing a sequence of data **x(t)= x(1), . . . , x(τ)** with the time
step index ***t*** ranging from **1 to τ**. For tasks that involve
sequential inputs, such as speech and language, it is often better to
use RNNs. In a NLP problem, if you want to predict the next word in a
sentence it is important to know the words before it. RNNs are
called *recurrent* because they perform the same task for every element
of a sequence, with the output being depended on the previous
computations. Another way to think about RNNs is that they have a
"memory" which captures information about what has been calculated so
far.

**Architecture : **Let us briefly go through a basic RNN network.

![Image for post](media/image17.png){width="4.194444444444445in"
height="1.7834645669291338in"}

The left side of the above diagram shows a notation of an RNN and on the
right side an RNN being *unrolled* (or unfolded) into a full network. By
unrolling we mean that we write out the network for the complete
sequence. For example, if the sequence we care about is a sentence of 3
words, the network would be unrolled into a 3-layer neural network, one
layer for each word.

**Input:** *x(t)*​ is taken as the input to the network at time
step *t.* For example, *x1,*could be a one-hot vector corresponding to a
word of a sentence.

**Hidden state***: h(t)*​ represents a hidden state at time t and acts
as "memory" of the network. *h(t)*​ is calculated based on the current
input and the previous time step's hidden state: **h(t)**​
= *f*(*U ***x(t)**​ + *W ***h(t**−**1)**​). The function *f* is taken to
be a non-linear transformation such as *tanh*, *ReLU.*

**Weights**: The RNN has input to hidden connections parameterized by a
weight matrix U, hidden-to-hidden recurrent connections parameterized by
a weight matrix W, and hidden-to-output connections parameterized by a
weight matrix V and all these weights (*U*,*V*,*W)* are shared across
time.

**Output**: *o(t)*​ illustrates the output of the network. In the figure
I just put an arrow after *o(t) *which is also often subjected to
non-linearity, especially when the network contains further layers
downstream.

**Forward Pass**

The ﬁgure does not specify the choice of activation function for the
hidden units. Before we proceed we make few assumptions: 1) we assume
the hyperbolic tangent activation function for hidden layer. 2) We
assume that the output is discrete, as if the RNN is used to predict
words or characters. A natural way to represent discrete variables is to
regard the output **o** as giving the un-normalized log probabilities of
each possible value of the discrete variable. We can then apply the
softmax operation as a post-processing step to obtain a vector ***ŷ***of
normalized probabilities over the output.

The RNN forward pass can thus be represented by below set of equations.

![Image for post](media/image18.png){width="2.2777777777777777in"
height="1.0165737095363079in"}

This is an example of a recurrent network that maps an input sequence to
an output sequence of the same length. The total loss for a given
sequence of **x** values paired with a sequence of **y** values would
then be just the sum of the losses over all the time steps. We assume
that the outputs ***o(t)***are used as the argument to the softmax
function to obtain the vector ***ŷ*** of probabilities over the output.
We also assume that the loss **L** is the negative log-likelihood of the
true target ***y(t)***given the input so far.

Backward Pass

The gradient computation involves performing a forward propagation pass
moving left to right through the graph shown above followed by a
backward propagation pass moving right to left through the graph. The
runtime is O(τ) and cannot be reduced by parallelization because the
forward propagation graph is inherently sequential; each time step may
be computed only after the previous one. States computed in the forward
pass must be stored until they are reused during the backward pass, so
the memory cost is also O(τ). The back-propagation algorithm applied to
the unrolled graph with O(τ) cost is called back-propagation through
time (BPTT). Because the parameters are shared by all time steps in the
network, the gradient at each output depends not only on the
calculations of the current time step, but also the previous time steps.

Computing Gradients

Given our loss function *L*, we need to calculate the gradients for our
three weight matrices *U, V, W, and *bias terms* b, c *and update
them* *with a learning rate ***α***. Similar to normal back-propagation,
the gradient gives us a sense of how the loss is changing with respect
to each weight parameter. We update the weights W to minimize loss with
the following equation:

![Image for post](media/image19.png){width="1.4930555555555556in"
height="0.44112970253718287in"}

The same is to be done for the other weights U, V, b, c as well.

Let us now compute the gradients by BPTT for the RNN equations above.
The nodes of our computational graph include the parameters U, V, W, b
and c as well as the sequence of nodes indexed by t for x (t), h(t),
o(t) and L(t). For each node **n** we need to compute the
gradient **∇nL** recursively, based on the gradient computed at nodes
that follow it in the graph.

**Gradient with respect to output o(t) **is calculated assuming the o(t)
are used as the argument to the softmax function to obtain the
vector ***ŷ*** of probabilities over the output. We also assume that the
loss is the negative log-likelihood of the true target y(t).

![Image for post](media/image20.png){width="4.451388888888889in"
height="0.7087270341207349in"}

Let us now understand how the gradient flows through hidden state h(t).
This we can clearly see from the below diagram that at time t, hidden
state h(t) has gradient flowing from both current output and the next
hidden state.

![Image for post](media/image21.png){width="1.875in"
height="1.2651695100612423in"}

Red arrow shows gradient flow

We work our way backward, starting from the end of the sequence. At the
ﬁnal time step τ, h(τ) only has o(τ) as a descendant, so its gradient is
simple:

![Image for post](media/image22.png){width="2.2916666666666665in"
height="0.4680161854768154in"}

We can then iterate backward in time to back-propagate gradients through
time, from t=τ −1 down to t = 1, noting that h(t) (for t \< τ ) has as
descendants both o(t) and h(t+1). Its gradient is thus given by:

![Image for post](media/image23.png){width="4.486111111111111in"
height="0.799588801399825in"}

Once the gradients on the internal nodes of the computational graph are
obtained, we can obtain the gradients on the parameter nodes. The
gradient calculations using the chain rule for all parameters is:

![Image for post](media/image24.png){width="3.4375in"
height="1.743046806649169in"}

Implementation

We will implement a full Recurrent Neural Network from scratch using
Python. We will try to build a text generation model using an RNN. We
train our model to predict the probability of a character given the
preceding characters. It's a *generative model*. Given an existing
sequence of characters we sample a next character from the predicted
probabilities, and repeat the process until we have a full sentence.

General steps to follow:

1.  Initialize weight matrices *U, V, W *from random distribution* *and
    > bias b, c with zeros

2.  Forward propagation to compute predictions

3.  Compute the loss

4.  Back-propagation to compute gradients

5.  Update weights based on gradients

6.  Repeat steps 2--5

Advantages of Recurrent Neural Network
======================================

1.  **RNN **can model sequence of data so that each sample can be
    > assumed to be dependent on previous ones

2.  Recurrent neural network are even used with convolutional layers to
    > extend the effective pixel neighbourhood.

**Disadvantages of Recurrent Neural Network**
=============================================

1.  Gradient vanishing and exploding problems.

2.  Training an RNN is a very difficult task.

3.  It cannot process very long sequences if using *tanh* or *relu* as
    > an activation function.

Applications of RNN
===================

RNNs can be used in a lot of different places. Following are a few
examples where a lot of RNNs are used.

1. Language Modelling and Generating Text
-----------------------------------------

Given a sequence of word, here we try to predict the likelihood of the
next word. This is useful for translation since the most likely sentence
would be the one that is correct.

2. Machine Translation
----------------------

Translating text from one language to other uses one or the other form
of RNN. All practical day systems use some advanced version of a RNN.

3. Speech Recognition
---------------------

Predicting phonetic segments based on input sound waves, thus
formulating a word.

4. Generating Image Descriptions
--------------------------------

A very big use case is to understand what is happening inside an image,
thus we have a good description. This works in a combination of CNN and
RNN. CNN does the segmentation and RNN then used the segmented data to
recreate the description. It's rudimentary but the possibilities are
limitless.

5. Video Tagging
----------------

This can be used for video search where we do image description of a
video frame by frame.

**Long Short Term Memory (LSTM)?**
==================================

Long Short-Term Memory (LSTM) networks are a modified version of
recurrent neural networks, which makes it easier to remember past data
in memory. The vanishing gradient problem of RNN is resolved here. LSTM
is well-suited to classify, process and predict time series given time
lags of unknown duration. It trains the model by using back-propagation.
In an LSTM network, three gates are present:

![Image for post](media/image25.jpeg){width="3.5in"
height="2.2327580927384076in"}

1.  **Input gate** --- discover which value from input should be used to
    > modify the memory. **Sigmoid** function decides which values to
    > let through **0,1. **and **tanh **function gives weightage to the
    > values which are passed deciding their level of importance ranging
    > from**-1** to **1.**

> ![Image for post](media/image26.png){width="2.789831583552056in"
> height="0.7408147419072616in"}

**2. Forget gate **--- discover what details to be discarded from the
block. It is decided by the **sigmoid function. **it looks at the
previous state(**ht-1**) and the content input(**Xt**) and outputs a
number between **0(***omit this*)and **1(***keep this***)**for each
number in the cell state **Ct−1**.

![Image for post](media/image27.png){width="2.513888888888889in"
height="0.38524496937882763in"}

**3. Output gate** --- the input and the memory of the block is used to
decide the output. **Sigmoid** function decides which values to let
through **0,1. **and **tanh **function gives weightage to the values
which are passed deciding their level of importance ranging
from**-1** to **1 **and multiplied with output of **Sigmoid.**

![Image for post](media/image28.png){width="2.866101268591426in"
height="0.7569444444444444in"}

**Bidirectional RNN**

> ![Image for post](media/image29.png){width="5.698199912510936in"
> height="2.0137620297462817in"}

Fig 1: General Structure of Bidirectional Recurrent Neural Networks.

Bidirectional recurrent neural networks(RNN) are really just putting two
independent RNNs together. The input sequence is fed in normal time
order for one network, and in reverse time order for another. The
outputs of the two networks are usually concatenated at each time step,
though there are other options, e.g. summation.

This structure allows the networks to have both backward and forward
information about the sequence at every time step. The concept seems
easy enough. But when it comes to actually implementing a neural network
which utilizes bidirectional structure, confusion arises...

The Confusion

The first confusion is about **the way to forward the outputs of a
bidirectional RNN to a dense neural network**. For normal RNNs we could
just forward the outputs at the last time step, and the following
picture I found via Google shows similar technique on a bidirectional
RNN.

![Image for post](media/image30.jpeg){width="2.138888888888889in"
height="1.9729713473315835in"}

Fig 2: A confusing formulation.

But wait... if we pick the output at the last time step, the reverse RNN
will have only seen the last input (x_3 in the picture). It'll hardly
provide any predictive power.

The second confusion is about the **returned hidden states**. In seq2seq
models, we'll want hidden states from the encoder to initialize the
hidden states of the decoder. Intuitively, if we can only choose hidden
states at one time step(as in PyTorch), we'd want the one at which the
RNN just consumed the last input in the sequence. But **if **the hidden
states of time step *n* (the last one) are returned, as before, we'll
have the hidden states of the reversed RNN with only one step of inputs
seen.
