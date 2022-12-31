
*title:* Dive into Deep Learning
*authors:* Aston Zhang, Zachary C Lipton, Mu Li, Alexander J Smola
*year:* 2022
*tags:* #deep-learning #gradient_boosting #semi-supervised #transformer
*status:* #📥
*code:* [2.3. Linear Algebra — Dive into Deep Learning 0.17.0 documentation (d2l.ai)](https://d2l.ai/chapter_preliminaries/linear-algebra.html?highlight=frobenius)

## Norms

Analogous to $L_{2}$ norms of vectors, the Frobenius norm of a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ is the square root of the sum of the squares of the matrix elements:

$$
\|\mathbf{X}\|_{F}=\sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} x_{i j}^{2}}
$$

The Frobenius norm satisfies all the properties of vector norms. It behaves as if it were an $L_{2}$ norm of a matrix-shaped vector. Invoking the following function will calculate the Frobenius norm of a matrix.


## Attention

- framework for designing attention mechanisms consists of:
    - volitional (~free) cues = queries
    - sensory inputs  = keys
    - nonvolitional cue of sensory input = keys
- attention pooling mechanism  enables a given query (volitional cue) to interact with keys (nonvolitional cues) which guides a biased selection over values (sensory inputs)

- self attention enjoys both parallel computation and the shortest maximum path length. Which makes it appealing to design deep architectures by using self-attention. Do not require a convolutional layer or recurrent layer.

- It's an instance of an encoder-decoder architecture. Input and output sequence embeddings are added with positional encoding before being fed into the encoder and the decoder that stack modules based on self-attention.


## Multilayer perceptron

As before, by the matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, we denote a minibatch of $n$ examples where each example has $d$ inputs (features). For a one-hidden-layer MLP whose hidden layer has $h$ hidden units, denote by $\mathbf{H} \in \mathbb{R}^{n \times h}$ the outputs of the hidden layer, which are hidden representations. In mathematics or code, $\mathbf{H}$ is also known as a hidden-layer variable or a hidden variable. Since the hidden and output layers are both fully connected, we have hidden-layer weights $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ and biases $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$ and output-layer weights $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ and biases $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$. Formally, we calculate the outputs $\mathbf{O} \in \mathbb{R}^{n \times q}$ of the one-hidden-layer MLP as follows:
$$
\begin{aligned}
&\mathbf{H}=\mathbf{X W}^{(1)}+\mathbf{b}^{(1)} \\
&\mathbf{O}=\mathbf{H W}^{(2)}+\mathbf{b}^{(2)}
\end{aligned}
$$

(p. 131)

We can view the equivalence formally by proving that for any values of the weights, we can just collapse out the hidden layer, yielding an equivalent single-layer model with parameters $\mathbf{W}=$ $\mathbf{W}^{(1)} \mathbf{W}^{(2)}$ and $\mathbf{b}=\mathbf{b}^{(1)} \mathbf{W}^{(2)}+\mathbf{b}^{(2)}$ :
$$
\mathbf{O}=\left(\mathbf{X} \mathbf{W}^{(1)}+\mathbf{b}^{(1)}\right) \mathbf{W}^{(2)}+\mathbf{b}^{(2)}=\mathbf{X} \mathbf{W}^{(1)} \mathbf{W}^{(2)}+\mathbf{b}^{(1)} \mathbf{W}^{(2)}+\mathbf{b}^{(2)}=\mathbf{X} \mathbf{W}+\mathbf{b} .
$$

In order to realize the potential of multilayer architectures, we need one more key ingredient: a nonlinear activation function $\sigma$ to be applied to each hidden unit following the affine transformation. The outputs of activation functions (e.g., $\sigma(\cdot)$ ) are called activations. In general, with activation functions in place, it is no longer possible to collapse our MLP into a linear model:
$$
\begin{aligned}
&\mathbf{H}=\sigma\left(\mathbf{X} \mathbf{W}^{(1)}+\mathbf{b}^{(1)}\right) \\
&\mathbf{O}=\mathbf{H} \mathbf{W}^{(2)}+\mathbf{b}^{(2)}
\end{aligned}
$$
Since each row in $\mathbf{X}$ corresponds to an example in the minibatch, with some abuse of notation, we define the nonlinearity $\sigma$ to apply to its inputs in a rowwise fashion, i.e., one example at a time. Note that we used the notation for softmax in the same way to denote a rowwise operation in Section 3.4.5. Often, as in this section, the activation functions that we apply to hidden layers are not merely rowwise, but elementwise. That means that after computing the linear portion of the layer, we can calculate each activation without looking at the values taken by the other hidden units. This is true for most activation functions.

To build more general MLPs, we can continue stacking such hidden layers, e.g., $\mathbf{H}^{(1)}=\sigma_{1}\left(\mathbf{X W}^{(1)}+\right.$ $\left.\mathbf{b}^{(1)}\right)$ and $\mathbf{H}^{(2)}=\sigma_{2}\left(\mathbf{H}^{(1)} \mathbf{W}^{(2)}+\mathbf{b}^{(2)}\right)$, one atop another, yielding ever more expressive models.

## Universal approximators

- Even witha single-hidden-layer network, given enough nodes (possibly absurdly many), and the right set of weights, we can model any function, though actually learning that function is the hard part.
- Moreover, just because a single-hidden-layer network can learn any function does not mean that you should try to solve all of your problems with single-hidden-layer networks. In fact, we can approximate many functions much more compactly by using deeper (vs. wider) networks.

## Activation functions

Activation functions decide whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it. They are differentiable operators to transform input signals to outputs, while most of them add non-linearity.

## ReLU
The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks, is the rectified linear unit (ReLU). ReLU provides a very simple nonlinear transformation. Given an element $x$, the function is defined as the maximum of that element and 0 :
$$
\operatorname{ReLU}(x)=\max (x, 0).
$$