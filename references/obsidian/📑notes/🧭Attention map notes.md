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

## Visualization

For visualization see also:
http://nlp.seas.harvard.edu/annotated-transformer/
![[attention-map-saint.png]] ^401670
(Copied from [[@somepalliSAINTImprovedNeural2021]])