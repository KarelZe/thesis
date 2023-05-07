Relates to #attention #shap #random-feature-permutation #explainability #interpreability 

In addition to [[📖chapters/🧭Random Feature Permutation]], transformer-based models offer *some* interpretability through their attention mechanism [^1] . Recall from our discussion on attention (Cref [[🅰️Attention]]) that the attention matrix stores how much attention a token pays to each of the keys. Thus, feature attributions can be derived from attention by visualizing features that the model attends to in an attention map. While attention maps are specific to Transformers or other attention-based architectures, rendering them useless for cross-model comparisons, they give additional insights from different attention layers and attention heads of the model on a per-trade and global basis. An example is shown in Figure [[#^401670]].

(figure)

In the tabular domain, various approaches for obtaining attention from multiple attention heads and transformer blocks have been explored in the literature. ([[@somepalliSaintImprovedNeural2021]]18) and ([[@borisovDeepNeuralNetworks2022]]11) gather attention maps from the first attention layer only, and ([[@borisovDeepNeuralNetworks2022]]11) obtain feature attributions by taking the diagonal of the attention matrix $\boldsymbol{A}$ or through column-wise summation. In contrast, ([[@gorishniyRevisitingDeepLearning2021]]10) leverage all attention matrices by averaging over multiple transformer blocks, attention heads, and samples to obtain global feature attributions. In the light of (Cref [[🗼Overview Transformer]] and [[🅰️Attention]]), where we emphasized the unique roles of attention heads and lower sublayers, both approaches may be myopic, as attention heads may contribute unequally to the result or as later attention layers are neglected entirely.

While not explored systematically in the tabular domain yet, the *rollout attention* method of ([[@abnarQuantifyingAttentionFlow2020]]3) combines raw attention from multiple layers through recursive matrix multiplication with the weight matrices from attention layers below, as shown in this Equation: [^1]
$$
\begin{aligned}
\hat{\boldsymbol{A}}^{(l)} & =\boldsymbol{I}+\mathbb{E}_h \boldsymbol{A}^{(l)} \\
\operatorname { rollout } & =\hat{\boldsymbol{A}}^{(1)} \cdot \hat{\boldsymbol{A}}^{(2)} \ldots\cdot\hat{\boldsymbol{A}}^{(L)}
\end{aligned}
$$
In each layer the raw attention scores $\boldsymbol{A}^{(b)}$ are averaged over $h$ heads, denoted by $\mathbb{E}_h$. The identity matrix $\boldsymbol{I}$ is added to account for the residual connections (see cref [[🔗Residual connections]]).  While *rollout attention* considers all attention layers in the calculation of feature attributions, it does not consider a signal and attributes equal weights to all attention heads ([[@cheferTransformerInterpretabilityAttention2021]]786). This contradicts our observations in [[🅰️Attention]].

In an effort to explain the decision-making process of multi-modal Transformer, including self-attention-based Transformer, ([[@cheferGenericAttentionmodelExplainability2021]]3) incorporate gradients to when averaging across the heads of a layer, as shown in Equation (...).
$$
\begin{aligned}
\bar{\boldsymbol{A}}^{(l)}&=I+ \mathbb{E}_h\left(\left(\nabla \boldsymbol{A}^{(l)} \odot \boldsymbol{A}^{(l)}\right)^{+}\right) \\
\operatorname {w\_rollout} & =\bar{\boldsymbol{A}}^{(1)} \cdot \bar{\boldsymbol{A}}^{(2)} \ldots \cdot \bar{\boldsymbol{A}}^{(L)}
\end{aligned}
$$
In this approach, the element-wise product between the gradient of the attention map $\nabla \boldsymbol{A}^{(l)}=\frac{\partial y_t}{\partial \boldsymbol{A}}$ for the model's target class $t$ and the attention map $\boldsymbol{A}^{(l)}$ is calculated to weight the attention head's importance. As previously suggested in ([[@cheferTransformerInterpretabilityAttention2021]]786), negative contributions are eliminated to focus on the positive relevance, and the results are averaged over the heads dimension. Like all other presented approaches (cref-eq and cref-eq) can be computed with a single forward pass and is thus computationally efficient.

In absence of ground truth for the true feature attribution, we also calculate attention maps using cref. Inline with previous research, feature attributions are also summed over the first attention layer or all transformer blocks. Due to the limitation that TabTransformer ([[@huangTabTransformerTabularData2020]]2--3) only performs self-attention on categorical features, feature attributions for numerical features are omitted. The level of agreement between attributions from attention maps and [[📖chapters/🧭Random Feature Permutation]] is quantified by calculating Spearman's rank correlation between them.

The next chapter discusses different metrics to assess the prediction quality of our models.

[^1:] Notation from [[@cheferTransformerInterpretabilityAttention2021]] (p. 786).
[^2:] One has to distinguish interpretability through *explainability* from *transparency* ([[@liptonMythosModelInterpretability2017]] 4--5). In recent research a major controversy embarked around the question, whether attention offers explanations to model predictions (cp. [[@bastingsElephantInterpretabilityRoom2020]]150) (cp. [[@jainAttentionNotExplanation2019]] 5--7) and (cp. [[@wiegreffeAttentionNotNot2019]]9). The debate sparked around opposing definitions of explainability and the consistency of attention scores with other, established feature-importance measures. Our focus is less on post-hoc explainability of the model, but rather on transparency. Consistent with ([[@wiegreffeAttentionNotNot2019]]8) we view attention scores as a vehicle to model transparency. 

**Notes:**
[[🧭Attention map notes]]