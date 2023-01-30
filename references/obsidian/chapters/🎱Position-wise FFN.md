
Besides the attention sub-layer ([[🅰️Attention]]), the transformer block contains a two-layered feed-forward network, which is applied to each embedding separately and identically ([[@vaswaniAttentionAllYou2017]]) (p. 5). The network consists of a linear transformation, followed by a non-linear activation function, and a second linear layer. For the $l$-th layer, the network is given by:
$$
\tag{1}
X = X+W_{\mathrm{mlp} 2}^l \operatorname{ReLU}\left(W_{\mathrm{mlp} 1}^l X+b_{\mathrm{mlp} 1}^l 1^{\top}\right)+b_{\mathrm{mlp} 2}^l 1^{\top},
$$
with $W_{\text{mlp1}}^{l}, b_{\text{mlp1}}^{l}, W_{\text{mlp2}}^{l}, b_{\text{mlp2}}^{l} \in \mathbb{R}^{\cdots}$ being learnable parameters identical for all embeddings in the layer. 

In the transformer, the point-wise feed-forward network acts as a persistent memory, capable of storing general information on the task and independent from the immediate context ([[@sukhbaatarAugmentingSelfattentionPersistent2019]]) (p. 3). (see also [[🤖Transformer#^54aa8a]] and [[@gevaTransformerFeedForwardLayers2021]]). [[@vaswaniAttentionAllYou2017]] (p. 9) set the hidden dimension to be two to eight magnitudes of the embedding dimension. The large capacity strengthens the model's ability to retain information but also contributes significantly to the high computational requirements and memory footprint of transformers (See [[@tayEfficientTransformersSurvey2022]] (p. 5); [[@kitaevReformerEfficientTransformer2020]] (p. 1)).  

The classical transformer of [[@vaswaniAttentionAllYou2017]] (p. 5) uses the *Rectified Linear Units* $\operatorname{ReLU}$ ([[@glorotDeepSparseRectifier2011]]; p. 318) activation (see Equation $(1)$), given by $\max(0, X)$. Later variants (see e. g., [[@devlinBERTPretrainingDeep2019]] or [[@radfordImprovingLanguageUnderstanding]]) commonly replace the $\operatorname{ReLU}$ with the *Gaussian Error Linear Units* $\operatorname{GELU}$ ([[@hendrycksGaussianErrorLinear2020]], p. 2) activation, which has empirically proven to improve the performance and convergence behavior of transformers ([[@narangTransformerModificationsTransfer2021]], p. 16; and [[@shazeerGLUVariantsImprove2020]] p. 4).

Like the [[🅰️Attention]] sub-layer, the feed-forward sub-layer is surrounded by residual connections ([[@heDeepResidualLearning2015]]) and followed by a layer-normalization ([[@baLayerNormalization2016]] (p. 4)) layer. Optionally, dropout ([[@srivastavaDropoutSimpleWay]] p. 1,930) is applied to prevent the model from overfitting. 

**Notes:**
[[🎱Position-wise FFN notes]]