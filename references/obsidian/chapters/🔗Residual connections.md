Before studying Transformers for tabular data, we revisit *residual connections* and *layer norm* in the Transformer block and discuss alternatives for their arrangement. What seems like a pedantic detail, is vital for the training process and convergence of the overall network, as we show.

Recall from our overview of Transformers (see Chapter [[🤖Transformer]]), that both the encoder and decoder stack multiple Transformer blocks, each of which consists of several sub-layers, resulting in deep networks. As neural networks are commonly trained using back-propagation, which relies on the gradient of the error with respect to the parameters to be propagated through the network starting at the last layer, vanishing or exploding gradients pose a major difficulty in training deep neural networks (see e. g. [[@heDeepResidualLearning2015]]). Likewise, for Transformers, stacking multiple layers in the encoder and decoder, not just prevents the gradient information to flow efficiently through the network, but also hampers the overall training performance ([[@wangLearningDeepTransformer2019]];  p. 1,811).  

As a remedy, [[@vaswaniAttentionAllYou2017]] (p. 3) add residual connections around each sub-layer. As shown in Equation $(3)$, the encoded token sequence $X \in \mathbb{R}^{d_e \times \ell_x}$ consists of the sub-layer's output added element-wisely to its input:
$$
X = X + \texttt{sub\_layer}\left(X\right)\tag{3}.
$$
Intuitively, the residual connection provides an alternative path for information to flow through the network, since some information can bypass the sub-layer and hence reach deeper layers within the stack. Also, exploding or vanishing gradients are mitigated, as gradients can bypass the sub-layer, eventually contributing towards an easier optimization ([[@liuRethinkingSkipConnection2020]]).  Residual connections moreover help to preserve the positional embeddings (see chapter [[🧵Positional Embedding]]), as the layer's inputs are maintained in the identity mapping.

**Notes:**
[[🔗residual connections notes]]