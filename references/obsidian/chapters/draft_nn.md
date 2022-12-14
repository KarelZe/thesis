
A FFN tries to approximate an arbitrary function $f^{*}$. To do so, it defines a mapping $\boldsymbol{y}=f(\boldsymbol{x} ; \boldsymbol{\theta})$ from some input $\boldsymbol{x}$ to some output $\boldsymbol{y}$ and learns the parameters $\boldsymbol{\theta}$, that approximate the true output best.

Structurally, a FFN consists of an input layer, one or more hidden layer and output layer. Thereby, each layer is made up of neurons and relies on input from the previous layer. In the most trivial case, the network consists of only a single hidden layer and the output layer. Formally, the output is calculated as shown in equation (...). $\mathbf{X} \in \mathbb{R}^{n \times d}$ denotes the input consisting of $d$ features and $n$ samples, $\mathbf{H} \in \mathbb{R}^{n \times h}$  the output of the hidden layer with $h$ hidden units and $\mathbf{O} \in \mathbb{R}^{n \times q}$ the final output. The  weights and bias for the hidden layer and output layer are denoted by $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ and biases $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$ and output-layer weights $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ and biases $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q} .$

$$
\begin{aligned} \mathbf{H} &=\sigma\left(\mathbf{X} \mathbf{W}^{(1)}+\mathbf{b}^{(1)}\right) \\ \mathbf{O} &=\mathbf{H} \mathbf{W}^{(2)}+\mathbf{b}^{(2)} \end{aligned}
$$
As seen above, an affine transformation is applied to the input, followed activation function $\sigma(\cdot)$, that decides whether a neuron in the hidden layer is activated. The final prediction is then obtained  after another affine transformation the output layer. Here, the parameter set consists of $\boldsymbol{\theta} = \left \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)},\mathbf{W}^{(2)}, \mathbf{b}^{(2)} \right\}$.

To learn the function approximation, FFNs are trained using backpropagation by adjusting the parameters $\boldsymbol{\theta}$ of each layer to minimize a loss function $\mathcal{L}(\cdot)$. As backpropagation requires the calculation of the gradient, both the activation and loss functions have to be differentiable.

ReLU is a common choice. It's non-linear and defined as the element-wise maximum between the input $\boldsymbol{x}$ and $0$:

$$
\operatorname{ReLU}(\boldsymbol{x})=\max (\boldsymbol{x}, 0).
$$

The usage of ReLU as activation function is desirable for a number of reasons. First, it can be computated efficiently as no exponential function is required. Secondly, it solves the vanishing gradient problem present in other activation functions [[@glorotDeepSparseRectifier2011]].

Networks with a single hidden layer can approximate any arbitrary function given enough data and network capacity [[@hornikMultilayerFeedforwardNetworks1989]].  
In practice, similiar effects can be achieved by stacking several hidden layers and thereby deepening the network, while being more compact [[@zhangDiveDeepLearning2021]].

Deep neural nets combine several hidden layers by feeding the previous hidden layer's output into the subsequent hidden layer. Assuming a $\operatorname{ReLU}(\cdot)$ activation function, the stacking for a network with two hidden layers can be formalized as: $\boldsymbol{H}^{(1)}=\operatorname{ReLU}_{1}\left(\boldsymbol{X W}^{(1)}+\boldsymbol{b}^{(1)}\right)$ and $\boldsymbol{H}^{(2)}=\operatorname{ReLU}_{2}\left(\mathbf{H}^{(1)} \mathbf{W}^{(2)}+\mathbf{b}^{(2)}\right)$.

Feed forward networks are restricted to information flowing through the network in a forward manner. To also incorporate feedback from the output, we introduce Recursive Neural Nets as part of section (...).