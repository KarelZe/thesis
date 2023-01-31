Before studying Transformers for tabular data, we revisit *residual connections* and *layer norm* in the Transformer block and discuss alternatives for their arrangement. What seems like a pedantic detail, is vital for the training process and convergence of the overall network, as we show.

Recall from our overview of Transformers (see Chapter [[🤖Transformer]]), that both the encoder and decoder stack multiple Transformer blocks, each of which consists of several sub-layers, resulting in deep networks. As neural networks are commonly trained using back-propagation, which relies on the gradient of the error with respect to the parameters to be propagated through the network starting at the last layer, vanishing or exploding gradients pose a major difficulty in training deep neural networks (see e. g. [[@heDeepResidualLearning2015]]). Likewise, for Transformers, stacking multiple layers in the encoder and decoder, not just prevents the gradient information to flow efficiently through the network, but also hampers the overall training performance ([[@wangLearningDeepTransformer2019]];  p. 1,811).  

As a remedy, [[@vaswaniAttentionAllYou2017]] (p. 3) add residual connections around each sub-layer. As shown in Equation $(3)$, the encoded token sequence $X \in \mathbb{R}^{d_e \times \ell_x}$ consists of the sub-layer's output added element-wisely to its input:
$$
X = X + \texttt{sub\_layer}\left(X\right)\tag{3}.
$$
<<<<<<< HEAD
Intuitively, the residual connection provides an alternative path for information to flow through the network, since some information can bypass the sub-layer and hence reach deeper layers within the stack. Also, exploding or vanishing gradients are mitigated, as gradients can bypass the sub-layer, eventually contributing towards an easier optimization ([[@liuRethinkingSkipConnection2020]]).  Residual connections moreover help to preserve the positional embeddings (see chapter [[🧵Positional Embedding]]), as the layer's inputs are maintained in the identity mapping.
=======
where $x_l$ and $x_{l+1}$ are the input and output of the $l$-th sub-layer, and $y_l$ is the intermediate output followed by the post-processing function $f(\cdot)$. In this way, $x_l$ is explicitly exposed to $y_l$ (see Eq. (2)).</mark>

![[residual-connection.png]]
(from [[@heDeepResidualLearning2015]])

Intuitively, the residual connection provides an alternative path for information to flow through the network, since some information can bypass the sub-layer and is added to its output. Also, exploding or vanishing gradients are mitigated, as gradients can bypass the sub-layer, ultimately resulting in an easier optimization ([[@liuRethinkingSkipConnection2020]]).  Residual connections also help to preserve the positional embeddings ([[🧵Positional Embedding]]) as, the layer's input are maintained in the identity mapping.<mark style="background: #FFB8EBA6;"> (may come back and read this https://transformer-circuits.pub/2021/framework/index.html)</mark>

<mark style="background: #FFF3A3A6;">(Introduce the word residual  stream, residual learning.)</mark>

<mark style="background: #D2B3FFA6;">(shortest description A transformer starts with a token embedding, followed by a series of “residual blocks”, and finally a token unembedding. Each residual block consists of an attention layer, followed by an MLP layer. Both the attention and MLP layers each “read” their input from the residual stream (by performing a linear projection), and then “write” their result to the residual stream by adding a linear projection back in. Each attention layer consists of multiple heads, which operate in parallel. from https://transformer-circuits.pub/2021/framework/index.html Think about it!)</mark>

>>>>>>> origin/main

**Notes:**
[[🔗residual connections notes]]