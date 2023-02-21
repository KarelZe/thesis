Relates to #attention #shap #random-feature-permutation  

In addition to [[🧭Kernel SHAP]], transformer-based models offer some interpretability through their attention mechanism. Feature attributions can be derived from attention by visualizing features that the model attends to in an *attention map*. While attention maps are specific to Transformers or other attention-based architectures, rendering them useless for cross-model comparisons, they give additional insights from different attention layers and attention heads of the model on a per-trade and global basis. An example is shown in Figure [[#^401670]].

![[attention-map-for-heads.png]]
(Copied from [[@zhangDiveDeepLearning2021]]; Code available or http://nlp.seas.harvard.edu/annotated-transformer/)

![[attention-map-saint.png]] ^401670
(Copied from [[@somepalliSAINTImprovedNeural2021]])

In the *tabular domain* various approaches for obtaining attention from multiple attention heads and transformer blocks have been explored in the literature [[@somepalliSAINTImprovedNeural2021]] and [[@borisovDeepNeuralNetworks2022]] gather attention maps from the first attention layer only, and [[@borisovDeepNeuralNetworks2022]] obtain feature attributions by taking the diagonal of the attention matrix $\mathbf{A}$ or through column-wise summation. In contrast, [[@gorishniyRevisitingDeepLearning2021]] leverage all attention matrices by averaging over multiple transformer blocks, attention heads, and samples to obtain global feature attributions. Both approaches may be myopic, as attention heads may contribute unequally to the result or as some attention layers are neglected entirely.

While not explored systematically for the tabular domain, the roles of different attention heads have been studied intensively in transformer-based machine translation (see e. g., [[@voitaAnalyzingMultiHeadSelfAttention2019]], [[@tangAnalysisAttentionMechanisms2018]]).  [[@voitaAnalyzingMultiHeadSelfAttention2019]] observe that attention heads have varying importance and serve distinct purposes like learning positional or syntactic relations between tokens. Also, all attention layers contribute to the model's prediction. Transferring their result back to the tabular domain, averaging over multiple heads or considering selected attention layers only may lead to undesired obfuscation effects. 

%%
<mark style="background: #FFB8EBA6;">TODO: Argue, why it makes sense to look at attention maps? Attention is a filter [[🅰️Attention]]</mark>

TODO: write about how higher layers learn details, whereas lower layers focus on more general/basic information. Try to find the paper again, that had some good references... 
%%


%%
interesting blog post on importance of heads and pruning them:
https://lena-voita.github.io/posts/acl19_heads.html

Heads have a different importance and many can even be pruned: [[@michelAreSixteenHeads2019]]

The study of the transformer architecture has focused on the role and function of self-attention layers (Voita et al., 2019; Clark et al., 2019; Vig and Belinkov, 2019) and on inter-layer differences (i.e. lower vs. upper layers) (Tenney et al., 2019; Jawahar et al., 2019). (look up citations in [[@gevaTransformerFeedForwardLayers2021]])

Also related are interpretability methods that explain predictions (Han et al., 2020; Wiegreffe and Pinter, 2019) (look up citations in [[@gevaTransformerFeedForwardLayers2021]])


<mark style="background: #FFB8EBA6;">(TODO: Use different notations for dot-product. In previous chapters it's no dot at all)</mark>

%%
Provide some short discussion. Address problems with interpretation of attention proabilites?
Possible papers: [[@bastingsElephantInterpretabilityRoom2020]] and  [[@jainAttentionNotExplanation2019]] and [[@wiegreffeAttentionNotNot2019]] and https://medium.com/@byron.wallace/thoughts-on-attention-is-not-not-explanation-b7799c4c3b24 and https://medium.com/@yuvalpinter/attention-is-not-not-explanation-dbc25b534017
A research on what transformers actually learn for simple language models: https://transformer-circuits.pub/2021/framework/index.html

A library that can also investigate models across multiple nodes:
https://transformer-circuits.pub/2021/garcon/index.html

For visualization see also:
http://nlp.seas.harvard.edu/annotated-transformer/
%%

%%

As part of their *rollout attention* method ([[@abnarQuantifyingAttentionFlow2020]]3) combine raw attention from multiple layers through recursive matrix multiplication with the weight matrices from attention layers below, as shown in this Equation: [^1]
$$
\begin{aligned}
\hat{\mathbf{A}}^{(b)} & =I+\mathbb{E}_h \mathbf{A}^{(b)} \\
\text { rollout } & =\hat{\mathbf{A}}^{(1)} \cdot \hat{\mathbf{A}}^{(2)} \ldots \ldots \cdot \hat{\mathbf{A}}^{(B)}
\end{aligned}
$$
In each layer the raw attention scores $\mathbf{A}^{(b)}$ are averaged over $h$ heads, denoted by $\mathbb{E}_h$. The identity matrix $I$ is added to account for the residual connections (see cref [[🔗Residual connections]]).  While *rollout attention* considers all attention layers in the calculation of feature attributions, it does not consider a signal and attributes equal weights to all attention heads ([[@cheferTransformerInterpretabilityAttention2021]]786). 

In an effort to explain the decision-making process for multi-modal Transformer, including self-attention-based Transformer, ([[@cheferGenericAttentionmodelExplainability2021]]3) incorporate gradients to when averaging across the heads of a layer, as shown in Equation (...):
%%%
![[agg-attention-heads.png]]
(image from their talk https://www.youtube.com/watch?v=bQTL34Dln-M&t=7s)
%%%
$$
\begin{aligned}
\bar{\mathbf{A}}^{(b)}&=I+ \mathbb{E}_h\left(\left(\nabla \mathbf{A}^{(b)} \odot \mathbf{A}^{(b)}\right)^{+}\right) \\
\text { ??? } & =\bar{\mathbf{A}}^{(1)} \cdot \bar{\mathbf{A}}^{(2)} \ldots \ldots \cdot \bar{\mathbf{A}}^{(B)}
\end{aligned}
$$
%%
(Note the formula in paper is a little different. But considering the commentary to Eq. 5 and Eq. 6, the explanation in their talk and the right distributivity law https://en.wikipedia.org/wiki/Matrix_multiplication of matrices it should be the same. Not sure though why they used a different notation.)
%%
In this equation, $\odot$ represents the element-wise product between the gradient of the attention map $\nabla \mathbf{A}^{(b)}:=\frac{\partial y_t}{\partial \mathbf{A}}$ for the model's target class $t$ and the attention map $\mathbf{A}^{(b)}$. As previously suggested in ([[@cheferTransformerInterpretabilityAttention2021]]786), negative contributions are eliminated in order to focus on positive relevance, and the results are averaged over the heads dimension. Like all previous approaches cref-eq can be computed with a single forward pass and is thus computationally efficient.


In absence of ground truth for the true feature attribution, we also calculate attention maps using cref. Inline with previous research, feature attributions are also summed over the first attention layer or all transformer blocks. All of these approaches can be computed with a single forward pass and are computationally efficient. The level of agreement between attributions from attention maps and [[🧭Kernel SHAP]] is quantified by calculating Spearman's rank correlation between them.

Due to the limitation that TabTransformer ([[@huangTabTransformerTabularData2020]]2--3) only performs self-attention on categorical features, feature attributions for numerical features are omitted.

[^1:] Notation from [[@cheferTransformerInterpretabilityAttention2021]] (p. 786).