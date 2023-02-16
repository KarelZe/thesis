


The [[🅰️Attention]]  mechanism allows to attend to other tokens in the immediate context. To retain general information on the task, outside and independent of the immediate context, each transformer block adds a point-wise feed-forward network, which acts as a persistent memory to the model ([[@sukhbaatarAugmentingSelfattentionPersistent2019]]3).  

The network consists of a linear transformation, followed by a non-linear activation function, and a second linear layer. For the $l$-th layer, the network is given by:
$$
X = X+W_{\mathrm{mlp} 2}^l \operatorname{ReLU}\left(W_{\mathrm{mlp} 1}^l X+b_{\mathrm{mlp} 1}^l 1^{\top}\right)+b_{\mathrm{mlp} 2}^l 1^{\top},
$$
with $W_{\text {mlp } 1}^l \in \mathbb{R}^{d_{\mathrm{mlp}} \times d_{\mathrm{e}}}, b_{\mathrm{mlp} 1}^l \in \mathbb{R}^{d_{\mathrm{mlp}}}, W_{\mathrm{mlp} 2}^l \in \mathbb{R}^{d_{\mathrm{e}} \times d_{\mathrm{mlp}}}$ and $b_{\mathrm{mlp} 2}^l \in \mathbb{R}^{d_{\mathrm{e}}}$ being learnable parameters identical for all embeddings in the layer. The network is applied to each embedding separately and identically.
 
([[@vaswaniAttentionAllYou2017]] 9) set the hidden dimension to be two to eight magnitudes of the embedding dimension. The large capacity strengthens the model's ability to retain information but also contributes significantly to the high computational requirements and memory footprint of Transformers ([[@tayEfficientTransformersSurvey2022]]5) and ([[@kitaevReformerEfficientTransformer2020]]1). Both linear transformations are separated by a *Rectified Linear Units* $\operatorname{ReLU}$ ([[@glorotDeepSparseRectifier2011]]318) activation function to add non-linearities to the network.

Like the attention layer, the position-wise FFN is surrounded by residual connections (see ), and followed by layer-normalization (see ). Optionally, dropout ([[@srivastavaDropoutSimpleWay]] 1930) is added to prevent the model from overfitting. 

**Notes:**
[[🎱Position-wise FFN notes]]