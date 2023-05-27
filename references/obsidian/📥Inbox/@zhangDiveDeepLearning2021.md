
*title:* Dive into Deep Learning
*authors:* Aston Zhang, Zachary C Lipton, Mu Li, Alexander J Smola
*year:* 2022
*tags:* #deep-learning #gbm #semi-supervised #transformer
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

In order to realise the potential of multilayer architectures, we need one more key ingredient: a nonlinear activation function $\sigma$ to be applied to each hidden unit following the affine transformation. The outputs of activation functions (e.g., $\sigma(\cdot)$ ) are called activations. In general, with activation functions in place, it is no longer possible to collapse our MLP into a linear model:
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

Activation functions decide whether a neurone should be activated or not by calculating the weighted sum and further adding bias with it. They are differentiable operators to transform input signals to outputs, while most of them add non-linearity.

## ReLU
The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks, is the rectified linear unit (ReLU). ReLU provides a very simple nonlinear transformation. Given an element $x$, the function is defined as the maximum of that element and 0 :
$$
\operatorname{ReLU}(x)=\max (x, 0).
$$


## Annotations
“he core idea behind the Transformer model is the attention mechanism, an innovation that was originally envisioned as an enhancement for encoder-decoder RNNs applied to sequenceto-sequence applications, like machine translations (Bahdanau et al., 2014)” ([Zhang et al., 2021, p. 384](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=384&annotation=SDXNAXVK))

“The intuition behind attention is that rather than compressing the input, it might be better for the decoder to revisit the input sequence at every step. Moreover, rather than always seeing the same representation of the input, one might imagine that the decoder should selectively focus on particular parts of the input sequence at particular decoding steps.” ([Zhang et al., 2021, p. 384](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=384&annotation=9L3CHEDW))

“Bahdanau’s attention mechanism provided a simple means by which the decoder could dynamically attend to different parts of the input at each decoding step.” ([Zhang et al., 2021, p. 384](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=384&annotation=RRNSTMAM))

“the attention mechanism (Bahdanau et al., 2014). We will cover the specifics of its application to machine translation later. For now, simply consider the following: denote by D def = f(k1; v1); : : : (km; vm)g a database of m tuples of keys and values. Moreover, denote by q a query. Then we can define the attention over D as Attention(q; D) def = m ∑ i=1 (q; ki)vi; (11.1.1) where (q; ki) 2 R (i = 1; : : : ; m) are scalar attention weights.” ([Zhang et al., 2021, p. 386](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=386&annotation=VPE6I8TX))

“The name attention derives from the fact that the operation pays particular attention to the terms for which the weight is significant (i.e., large)” ([Zhang et al., 2021, p. 386](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=386&annotation=X97MZPNZ))

“A common strategy to ensure that the weights sum up to 1 is to normalise them via (q; ki) = (q; ki) ∑ j(q; kj) : (11.1.2) In particular, to ensure that the weights are also nonnegative, one can resort to exponentiation. This means that we can now pick any function a(q; k) and then apply the softmax operation used for multinomial models to it via (q; ki) = exp(a(q; ki)) ∑ j exp(a(q; kj)) : (11.1.3)” ([Zhang et al., 2021, p. 386](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=386&annotation=ITG3VTFM))

“One of the benefits of the attention mechanism is that it can be quite intuitive, particularly when the weights are nonnegative and sum to 1. In this case we might interpret large weights as a way for the model to select components of relevance. While this is a good intuition, it is important to remember that it is just that, an intuition. Regardless, we may want to visualise its effect on the given set of keys, when applying a variety of different queries.” ([Zhang et al., 2021, p. 387](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=387&annotation=X9F3LDKS))

“By design, the attention mechanism provides a differentiable means of control by which a neural network can select elements from a set and to construct an associated weighted sum over representations” ([Zhang et al., 2021, p. 388](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=388&annotation=EWSZ97YR))

“In Section 11.2, we used a number of different distance-based kernels, including a Gaussian kernel to model interactions between queries and keys. As it turns out, distance functions are slightly more expensive to compute than inner products. As such, with the softmax operation to ensure nonnegative attention weights, much of the work has gone into attention scoring functions a in (11.1.3) and Fig.11.3.1 that are simpler to compute.” ([Zhang et al., 2021, p. 393](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=393&annotation=4V5NTPVC))

“Last, we need to keep the order of magnitude of the arguments in the exponential function under control. Assume that all the elements of the query q 2 Rd and the key ki 2 Rd are independent and identically drawn random variables with zero mean and unit variance. The dot product between both vectors has zero mean and a variance of d.” ([Zhang et al., 2021, p. 394](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=394&annotation=7YCZPHP5))

“To ensure that the variance of the dot product still remains one regardless of vector length, we use the scaled dot-product attention scoring function. That is, we rescale the dot-product by 1/pd.” ([Zhang et al., 2021, p. 394](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=394&annotation=42ATF3MN))

“Let’s return to the dot-product attention introduced in (11.3.2). In general, it requires that both the query and the key have the same vector length, say d, even though this can be addressed easily by replacing q⊤k with q⊤Mk where M is a suitably chosen matrix to translate between both spaces. For now assume that the dimensions match.” ([Zhang et al., 2021, p. 396](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=396&annotation=XJ3S8XGE))

“In practise, we often think in minibatches for efficiency, such as computing attention for n queries and m key-value pairs, where queries and keys are of length d and values are of length v. The scaled dot-product attention of queries Q 2 Rnd, keys K 2 Rmd, and values V 2 Rmv thus can be written as softmax ( QK⊤ pd ) V 2 Rnv:” ([Zhang et al., 2021, p. 396](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=396&annotation=TKW54GYM))

“In practise, given the same set of queries, keys, and values we may want our model to combine knowledge from different behaviours of the same attention mechanism, such as capturing dependencies of various ranges (e.g., shorter-range vs. longer-range) within a sequence. Thus, it may be beneficial to allow our attention mechanism to jointly use different representation subspaces of queries, keys, and values.” ([Zhang et al., 2021, p. 404](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=404&annotation=XTRRNAIU))

“To this end, instead of performing a single attention pooling, queries, keys, and values can be transformed with h independently learnt linear projections. Then these h projected queries, keys, and values are fed into attention pooling in parallel. In the end, h attention pooling outputs are concatenated and transformed with another learnt linear projection to produce the final output. This design is called multi-head attention, where each of the h attention pooling outputs is a head (Vaswani et al., 2017). Using fully connected layers to perform learnable linear transformations, Fig.11.5.1 describes multi-head attention.” ([Zhang et al., 2021, p. 404](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=404&annotation=28WTDU7B))

“Suppose that the input representation X 2 Rnd contains the d-dimensional embeddings for n tokens of a sequence. The positional encoding outputs X+P using a positional embedding matrix P 2 Rnd of the same shape, whose element on the ith row and the (2j)th or the (2j + 1)th column is pi;2j = sin (i 100002j/d ) ; pi;2j+1 = cos (i 100002j/d ) : (11.6.2) At first glance, this trigonometric-function design looks weird. Before explanations of this design, let’s first implement it in the following PositionalEncoding class.” ([Zhang et al., 2021, p. 409](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=409&annotation=TYRB7TI2))

“In the positional embedding matrix P, rows correspond to positions within a sequence and columns represent different positional encoding dimensions. In the example below, we can see that the 6th and the 7th columns of the positional embedding matrix have a higher frequency than the 8th and the 9th columns. The offset between the 6th and the 7th (same for the 8th and the 9th) columns is due to the alternation of sine and cosine functions.” ([Zhang et al., 2021, p. 409](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=409&annotation=WCDIVUPI))

“To see how the monotonically decreased frequency along the encoding dimension relates to absolute positional information, let’s print out the binary representations of 0; 1; : : : ; 7. As we can see, the lowest bit, the second-lowest bit, and the third-lowest bit alternate on every number, every two numbers, and every four numbers, respectively.” ([Zhang et al., 2021, p. 410](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=410&annotation=UCSLHMJT))

“In binary representations, a higher bit has a lower frequency than a lower bit. Similarly, as demonstrated in the heat map below, the positional encoding decreases frequencies along the encoding dimension by using trigonometric functions. Since the outputs are float numbers, such continuous representations are more space-efficient than binary representations.” ([Zhang et al., 2021, p. 410](zotero://select/library/items/Z9Q65UFB)) ([pdf](zotero://open-pdf/library/items/GHS48FST?page=410&annotation=88Z3VKY2))