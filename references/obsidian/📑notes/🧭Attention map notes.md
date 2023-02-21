## Attention is (not) explainability

“While many papers published on the topic of explainable AI have been criticised for not defining explanations (Lipton, 2018; Miller, 2019), the first key studies which spawned interest in attention as explanation (Jain and Wallace, 2019; Serrano and Smith, 2019; Wiegreffe and Pinter, 2019) do say that they are interested in whether attention weights faithfully represent the responsibility each input token has on a model prediction. That is, the narrow definition of explanation implied there is that it points at the most important input tokens for a prediction (arg max), accurately summarizing the reasoning process of the model (Jacovi and Goldberg, 2020b).” ([[@bastingsElephantInterpretabilityRoom2020]], 2020, p. 149)

“We have provided evidence that correlation between intuitive feature importance measures (including gradient and feature erasure approaches) and learned attention weights is weak for recurrent encoders (Section 4.1). We also established that counterfactual attention distributions — which would tell a different story about why a model made the prediction that it did — often have modest effects on model output (Section 4.2).” (Jain and Wallace, 2019, p. 10)

“A recent paper (Jain and Wallace, 2019) points to possible pitfalls that may cause researchers to misapply attention scores as explanations of model behavior, based on a premise that explainable attention distributions should be consistent with other feature-importance measures as well as exclusive given a prediction.1” (Wiegreffe and Pinter, 2019, p. 1)

““Under this definition, it should appear sensible of the NLP community to treat attention scores as a vehicle of (partial) transparency.” (Wiegreffe and Pinter, 2019, p. 8)Attention mechanisms do provide a look into the inner workings of a model, as they produce an easily-understandable weighting of hidden states.” (Wiegreffe and Pinter, 2019, p. 8)

TODO: write about how higher layers learn details, whereas lower layers focus on more general/basic information. Try to find the paper again, that had
interesting blog post on importance of heads and pruning them:
https://lena-voita.github.io/posts/acl19_heads.html




Possible papers: [[@bastingsElephantInterpretabilityRoom2020]] and  [[@jainAttentionNotExplanation2019]] and [[@wiegreffeAttentionNotNot2019]] and https://medium.com/@byron.wallace/thoughts-on-attention-is-not-not-explanation-b7799c4c3b24 and https://medium.com/@yuvalpinter/attention-is-not-not-explanation-dbc25b534017
A research on what transformers actually learn for simple language models: https://transformer-circuits.pub/2021/framework/index.html

A library that can also investigate models across multiple nodes:
https://transformer-circuits.pub/2021/garcon/index.html

Argue, why it makes sense to look at attention maps? Attention is a filter [[🅰️Attention]]

## Rollout 
$$
\begin{aligned}
\hat{\mathbf{A}}^{(b)} & =I+\mathbb{E}_h \mathbf{A}^{(b)} \\
\operatorname { rollout } & =\hat{\mathbf{A}}^{(1)} \cdot \hat{\mathbf{A}}^{(2)} \ldots \ldots \cdot \hat{\mathbf{A}}^{(B)}
\end{aligned}
$$
Notation from [[@cheferTransformerInterpretabilityAttention2021]] (p. 786).

While not explored systematically for the tabular domain, the roles of different attention heads have been studied intensively in transformer-based machine translation (cp. [[@voitaAnalyzingMultiHeadSelfAttention2019]]) and  (cp. [[@tangAnalysisAttentionMechanisms2018]]).  [[@voitaAnalyzingMultiHeadSelfAttention2019]] observe that attention heads have varying importance and serve distinct purposes like learning positional or syntactic relations between tokens. Also, all attention layers contribute to the model's prediction. 


## Cheffer

![[agg-attention-heads.png]]
(image from their talk https://www.youtube.com/watch?v=bQTL34Dln-M&t=7s)

$$
\begin{aligned}
\bar{\mathbf{A}}^{(b)}&=I+ \mathbb{E}_h\left(\left(\nabla \mathbf{A}^{(b)} \odot \mathbf{A}^{(b)}\right)^{+}\right) \\
\operatorname {weightedrollout} & =\bar{\mathbf{A}}^{(1)} \cdot \bar{\mathbf{A}}^{(2)} \ldots \ldots \cdot \bar{\mathbf{A}}^{(B)}
\end{aligned}
$$
(Note the formula in paper is a little different. But considering the commentary to Eq. 5 and Eq. 6, the explanation in their talk and the right distributivity law https://en.wikipedia.org/wiki/Matrix_multiplication of matrices it should be the same. Not sure though why they used a different notation.)

## Visualization

For visualization see also:
http://nlp.seas.harvard.edu/annotated-transformer/
![[attention-map-saint.png]] ^401670
(Copied from [[@somepalliSAINTImprovedNeural2021]])