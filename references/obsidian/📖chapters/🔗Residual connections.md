Summarize deep is necessary

Recall from cref, that both the encoder and decoder stack multiple Transformer blocks, each of which consists of several sub-layers, resulting in deep networks. As neural networks are commonly trained using back-propagation, which relies on the gradient of the error to be propagated through the network starting at the last layer, vanishing or exploding gradients pose a major difficulty in training deep neural networks ([[@heDeepResidualLearning2015]]1). Likewise, for Transformers, stacking multiple layers in the encoder and decoder, not just prevents the gradient information to flow efficiently through the network, but also hampers the overall training performance ([[@wangLearningDeepTransformer2019]];  p. 1,811).  

As a remedy, ([[@vaswaniAttentionAllYou2017]]3) add residual connections ([[@heDeepResidualLearning2015]]1--2) around each sub-layer. As shown in Equation cref, the encoded token sequence $X \in \mathbb{R}^{d_e \times \ell_x}$ consists of the sub-layer's output added element-wisely to its input:
$$
\boldsymbol{X} = \boldsymbol{X} + \operatorname{sub\_layer}\left(\boldsymbol{X}\right).
$$
Intuitively, the residual connection provides an alternative path for information to flow through the network, since some information can bypass the sub-layer and hence reach deeper layers within the stack. Also, exploding or vanishing gradients are mitigated, as gradients can bypass the sub-layer, eventually contributing towards an easier optimization ([[@liuRethinkingSkipConnection2020]]3591).  Residual connections moreover help to preserve the positional embeddings (see chapter [[🧵Positional Embedding]]), as the layer's inputs are maintained in the identity mapping.

**Notes:**
[[🔗residual connections notes]]