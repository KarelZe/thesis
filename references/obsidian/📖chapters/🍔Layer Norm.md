 ([[@vaswaniAttentionAllYou2017]]3) extensively draw on *layer normalization* after the multi-headed attention and feed-forward sub-layers. Layer normalization is used for normalizing the activations of the sub-layer and to stabilize and accelerate the training of the network ([[@baLayerNormalization2016]]2). For the transformer, the normalization statistics are calculated separately for every instance, which guarantees scalability across different batch sizes. For a vector $\boldsymbol{e}\in \mathbb{R}^{d_e}$ the normalized output is given by: 
$$
\tag{4}
\widehat{\boldsymbol{e}}=\frac{e-m}{\sqrt{v}} \odot \boldsymbol{\gamma}+\boldsymbol{\beta},
$$
calculated with the statistics $m = \sum_{i=1}^{d_{e}} \boldsymbol{e}[i] / d_{e}$ and $v = \sum_{i=1}^{d_{e}}(\boldsymbol{e}[i]-m)^2 / d_{e}$. Typically, the scale $\gamma$ and bias $\beta$ are set to preserve a zero mean and unit variance.

![[layer-norm.png]]
(Own work inspired by [[@wangLearningDeepTransformer2019]])

Until now it remains unclear, how the layer normalization intertwines with the sub-layers and the residual connections. Transformers are distinguished by the order in which layer normalization is added into the pre-norm and post-norm Transformer. Post-norm Transformers add layer normalization to the sub-layer *after* adding the input from the residual connections. The arrangement is depicted in (Cref). In contrast for the pre-norm Transformer, the normalization is applied *before* the self-attention and feed-forward sub-layers and inside the residual connections. Pre-norm requires one additional normalization layer to pass only well-conditioned outputs from the Transformer block to the successive layers ([[@xiongLayerNormalizationTransformer2020]]5). The setup is depicted in (Cref).

([[@vaswaniAttentionAllYou2017]]3) employ post-layer normalization. ThisMany recent transformers feature a pre-normalization setup, as observed by [[@narangTransformerModificationsTransfer2021]]. Parts of its success, lie in faster training, omitting the need for costly learning rate warm-up stages, whereby the learning rate is initially decreased to keep the gradients balanced ([[@xiongLayerNormalizationTransformer2020]] (p. 2); [[@liuVarianceAdaptiveLearning2021]]; [[@liuUnderstandingDifficultyTraining2020]]). 

poses additional requirements for the training such as the need for warm-up stages to 

Also, *post-norm* transformers are particularly brittle to train with several documented convergence failures with its root cause in vanishing gradients, exploding gradients, and a higher dependency on the residual stream ([[@liuUnderstandingDifficultyTraining2020]] (p. 8), [[@shazeerAdafactorAdaptiveLearning2018]],  [[@wangLearningDeepTransformer2019]]). Their pre-norm counterpart increases the robustness in training, sometimes at the cost performance, as documented in [[@liuUnderstandingDifficultyTraining2020]].  

We come back to this observation in Cref ([[🤖TabTransformer]]) and [[💡Training of models (supervised)]]. 
