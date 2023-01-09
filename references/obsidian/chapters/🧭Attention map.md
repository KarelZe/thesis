Relates to #attention #shap #random-feature-permutation 
 
### Attention visualization

In addition to [[🥌Kernel SHAP]], transformer-based models offer some interpretability through their attention mechanism. Feature attributions can be derived from attention by visualizing features that the model attends to in a *attention map*. While attention maps are specific to transformers or other attention-based architectures,  rendering them useless for cross-model comparisons, they give additional insights from different attention layers and attention heads of the model on a per-trade and global basis. An example is shown in Figure [[#^401670]].

![[attention-map-saint.png]] ^401670

In the *tabular domain* various approaches for obtaining attention from multiple attention heads and transformer blocks have been explored in the literature [[@somepalliSAINTImprovedNeural2021]] and [[@borisovDeepNeuralNetworks2022]] gather attention maps from the first attention layer only, and [[@borisovDeepNeuralNetworks2022]] obtain feature attributions by taking the diagonal of the attention matrix $\mathbf{A}$ or through column-wise summation. In contrast, [[@gorishniyRevisitingDeepLearning2021]] leverage all attention matrices by averaging over multiple transformer blocks, attention heads, and samples to obtain global feature attributions. Both approaches may be myopic, as attention heads may contribute unequally to the result or as some attention layers are neglected entirely.

While not explored systematically for the tabular domain, the roles of different attention heads have been studied intensively in transformer-based machine translation (see e. g., [[@voitaAnalyzingMultiHeadSelfAttention2019]], [[@tangAnalysisAttentionMechanisms2018]]).  [[@voitaAnalyzingMultiHeadSelfAttention2019]] observe that attention heads have a varying importance and serve distinct purposes like learning positional or syntactic relations between tokens. Also, all attention layers contribute to the model's prediction. Transferring their result back to the tabular domain, averaging over multiple heads or considering selected attention layers only and may lead to undesired obfuscation effects. 

As part of their *rollout attention* method [[@abnarQuantifyingAttentionFlow2020]] (p. 3) combine raw attention from multiple layers through recursive matrix multiplication with the weight matrices from attention layers below, as shown in this Equation: [^1]
$$
\begin{aligned}
\hat{\mathbf{A}}^{(b)} & =I+\mathbb{E}_h \mathbf{A}^{(b)} \\
\text { rollout } & =\hat{\mathbf{A}}^{(1)} \cdot \hat{\mathbf{A}}^{(2)} \ldots \ldots \cdot \hat{\mathbf{A}}^{(B)}
\end{aligned}
$$
In each layer the raw attention scores $\mathbf{A}^{(b)}$ are averaged over $h$ heads, denoted by $\mathbb{E}_h$. The identity matrix $I$ is added to account for the residual connections present in the  [[network_architecture]]. While *rollout attention* considers all attention layers in the calculation of feature attributions, it does not consider a signal and attributes equal weights to all attention heads [[@cheferTransformerInterpretabilityAttention2021]] (p. 786). 

In an effort to explain the decision-making process of any Transformer-based architecture, and in particular architectures based on self-attention like [[🤖TabTransformer]] or [[🤖FTTransformer]], [[@cheferGenericAttentionmodelExplainability2021]] (p. 3) propose to incorporate gradients when averaging across the heads of a layer, as shown in Equation (...):
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

In this equation, $\odot$ represents the element-wise product between the gradient of the attention map $\nabla \mathbf{A}^{(b)}:=\frac{\partial y_t}{\partial \mathbf{A}}$ for the model's target class $t$ and the attention map $\mathbf{A}^{(b)}$. As previously suggested in [[@cheferTransformerInterpretabilityAttention2021]] (p. 786), negative contributions are eliminated in order to focus on positive relevance, and the results are averaged over the heads dimension.

In absence of a ground truth for the true feature attribution, we also calculate attention maps using Eq. (...) and Eq. (...). Inline with previous research, feature attributions are also summed over the first attention layer or over all transformer blocks. All of these approaches, can be computed with a single forward pass and are computationally efficient. The level of agreement between attributions from attention maps and [[🥌Kernel SHAP]] is quantified by calculating Spearman's rank correlation between them.

Due to the limitation that TabTransformer ([[@huangTabTransformerTabularData2020]]) only feds categorical features through the transformer, only feature-attributions for non-continuous features can be estimated.

[^1:] Notation from [[@cheferTransformerInterpretabilityAttention2021]] (p. 786).