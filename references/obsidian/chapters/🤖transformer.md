

Transformers have been proposed by [[🧠Deep Learning Methods/Transformer/@vaswaniAttentionAllYou2017]] vor sequence-to-sequence modelling as radical new approach to Recurrent Neural Networks. Among others, the inherent sequential processing bounds the capabilities for learning long sequences and efficient parallel implementations in RNNs. Transformers adress these issues by utilizing a so-called attention mechanism to model dependencies between the input and output sequences of arbitrary length. *Attention* is a pooling mechanism, that uses query vector $\boldsymbol{q}$ to  perform a biased selection over similar keys $\boldsymbol{v}$ to obtain their corresponding values $\boldsymbol{k}$  ([[@zhangDiveDeepLearning2021]]).

On an abstract level, the Transformer consists of an encoder and a decoder.  The encoder maps the input sequence $\left(x_{1}, \ldots, x_{n}\right)$ with $x_{i} \in \mathbb{R}^{d}$, to a sequence of continous representations $\boldsymbol{z}=\left(z_{1}, \ldots, z_{n}\right)$, from which the decoder generates an output sequence of symbols $\boldsymbol{y}=\left(y_{1}, \ldots, y_{m}\right)$. Previously generated parts of the sequence are considered as an additional input, making the model autoregressive.

The encoder creates an attention-based representation,  empowering the search for similar observations within a large context. The encoder consists of a layer with two sublayers, which are stacked $N=6$ times. The layer is composed multi-head self-attention mechanism and point-wise, fully-connected feed forward networks. More over, residual connections are added around each of the sublayers, feeding into a normalizaton layer, that sits on top of each sub-layer. Residual connections add a copy of the input back to the output of a calculation to assure that the input is retained. Layer normalization helps to promote convergence and can cut training times (Ba et al).

The decoder obtains the output from the encoded representation. Similarily, it stacks $N=6$ identical layers. Besides the two sublayers from the encoder, a third sublayer performing multi-headed attention. The additional self-attention sub-layer is masked to prevent to positions to attending to future positions.  As before, residual connections and layer normalization are used for each of the sublayers.

Inputs and output tokens for the encoder respective decoder are not  processed as-is, but converted to learned embeddings of vectors with dimension $d_{\text {model }}$ first and enhanced with a positional encoding. We describe positional encoding later in detail later.

The specific attention mechanism used by [[🧠Deep Learning Methods/Transformer/@vaswaniAttentionAllYou2017]] is the *scaled dot product attention*, which is a faster and more space-efficient variant of the *dot-product attention* due to efficient matrix implementation. For this both queries, key and values are grouped into matrices.

$\operatorname{Attention}(\boldsymbol{Q}, \boldsymbol{K}, V)=\sigma\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{T}}{\sqrt{d_{k}}}\right) \boldsymbol{V}$

As shown in equation (...) the dot-product attention is defined as the dot products of the query with all keys devided by a scaling factor.  $\boldsymbol{Q}\boldsymbol{K}^{T}$ gives the probability distribution over all keys. The soft-max is used to obtain the weights to the values.

To further increase performance, the attention function is not performed on a single set of queries, keys and values, but rather on different linear projections of theirs. With *multi-headed attention*, the attention function is then performed on $h=8$ of the projections in parallel. The intermediate results from the different subspaces, get concatenated and the final value is obtained after a subsequent linear projection.

RNNs naturally maintain the order of tokens in a sequence through reccurence. In absence of such a mechanism, *positional encoding* at the bottom of the Transformer's encoder and decoder is added to the input embeddings to induce positional information from external.
The author suggest some *sinusodial positional encoding*, similar to the one used in chapter (...).

(formula?)

All in all transformers pose an hardware-efficient alternative for modelling sequences, including the prediction of time series.



The positional encoding uses $\sin(\cdot)$ and $\cos(\cdot)$ at different frequencies, similar to chapter (...).


Because self-attention operation is permutation invariant, it is important to use proper positional encodingto provide order information to the model. The positional encoding $\mathbf{P} \in \mathbb{R}^{L \times d}$ has the same dimension as the input embedding, so it can be added on the input directly. The vanilla Transformer considered two types of encodings:
(1) Sinusoidal positional encoding is defined as follows, given the token position $i=1, \ldots, L$ and the dimension $\delta=1, \ldots, d$ :
$$
\operatorname{PE}(i, \delta)= \begin{cases}\sin \left(\frac{i}{10000^{2 \delta^{\prime} / d}}\right) & \text { if } \delta=2 \delta^{\prime} \\ \cos \left(\frac{i}{10000^{25^{\prime} / d}}\right) & \text { if } \delta=2 \delta^{\prime}+1\end{cases}
$$
In this way each dimension of the positional encoding corresponds to a sinusoid of different wavelengths in different dimensions, from $2 \pi$ to $10000 \cdot 2 \pi$.




Recurrent models are inherently sequential, as their hidden state depends on the previous hidden state. This precludes them from parallelization, which is key for longer sequence lengths, as computations are memory bound. (Attention is all you need)

The transformer architecture uses a so-called attention mechanism in the encoder and decoder instead of recurrence, which allows them to be parallelized. (Attention is all you need)

The attention mechanism allows to model dependencies in sequences independent of their distance in the input and output sequences. (Attention is all you need)

In the transformer the number of operations required to relate signals from one two arbitrary input and output positions grows with the distance between distant position.  With the Transformer this is reduced to a constant number of operations. This comes at the cost of reduced effective resolution due to avreging attention-weighted positions, which is coutnerfeit bei Multi-Head attention. (Attention is all you need)

Self attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.  (Attention is all you need)


## Architecture
Here, the encoder maps an input sequence of symbol representations $\left(x_{1}, \ldots, x_{n}\right)$ to a sequence of continuous representations $\mathrm{z}=\left(z_{1}, \ldots, z_{n}\right)$. Given $\mathrm{z}$, the decoder then generates an output sequence $\left(y_{1}, \ldots, y_{m}\right)$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1 respectively.


Encoder: The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm $(x+$ Sublayer $(x))$, where Sublayer $(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text {model }}=512$.

Decoder: The decoder is also composed of a stack of $N=6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

## Attention
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

- nice explanation of transformers, such as dot-product attention https://t.co/WOlBY3suy4

Very high level overview: https://www.youtube.com/watch?app=desktop&v=SZorAJ4I-sA



## Point-wise Feed-Forward Networks



## Optimizer
- Adam



- General Introduction: [[🧠Deep Learning Methods/Transformer/@vaswaniAttentionAllYou2017]]
- What is Attentition?
- What is the difference between LSTMs and Transformers? Why are transformers preferable?



#### Attention is all you need; Attentional Neural Network Models | Łukasz Kaiser | Masterclass

- https://www.youtube.com/watch?v=rBCqOTEfxvg

- RNNs suffer from vanishing gradients
- Some people used CNNs, but path length is still logarithmic (going down a tree). Is limited to position.
- Attention: make a query with your vector and look at similar things in the past. Looks at everything, but choose things, that are similar.
- Encoder attention allows to go from one word to another. (Encoder Self-Attention)
- MaskedDecoder Self-Attention (is a single matrix multiply with a mask) to mask out all prev. elements not relevant
- Attention A(Q, K, V) (q = query vector) (K, V matrices= memory) (K = current word working on, V = all words generated before). You want to use q to find most similar k and get values that correspond to keys. (QK^T) gives a probability distribution over keys, which is then multiplied with values
- n^2 * d complexity
- to preserve the order of words they use multi-head attention
- attention heads can be interpreted (see winograd schemas)

#### Attention in Dive into Deep Learning (Zhang et al)
- framework for designing attention mechanisms consists of:
    - volitional (~free) cues = queries
    - sensory inputs  = keys
    - nonvolitional cue of sensory input = keys
- attention pooling mechanism  enables a given query (volitional cue) to interact with keys (nonvolitional cues) which guides a biased selection over values (sensory inputs)

- self attention enjoys both parallel computation and the shortest maximum path length. Which makes it appealing to design deep architectures by using self-attention. Do not require a convolutional layer or recurrent layer.

- It's an instance of an encoder-decoder architecture. Input and output sequence embeddings are added with positional encoding before being fed into the encoder and the decoder that stack modules based on self-attention.

## Advantages over LSTM
- More hardware friendly LSTMs require 4 linear layers for the gates and the cell state, for every timestep.


## Points to consider
- consistst of encoder and decoder
- uses an attention mechanism
- No need for recusive loops as with RNN and LSTM
- faster processing due to parallelization
- handle long-term depndencies
- decoder takes a sequence as an input, parallel processing inside encoder, input for the decoder, output of a sequence
- the sequence length of encoder and decoder is flexible (compare translation)
- the decoder processes autoregressive, meaning it considers previous outputs $y_i$, output before $y_i < y_j$.
- components.
	- encoder:
		- consists of 6 encoder blocks. Every encoder block consits of multi-head atttention and fully connected layers, and a normalization layer
		- residual connection?
	- self-attention
		- key mechanism to transfer sequences
		- from a sequence with variable size $x$ onto a sequence with the same size $I$ with the property ...