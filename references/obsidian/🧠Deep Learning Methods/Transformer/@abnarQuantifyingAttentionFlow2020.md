*title:* Quantifying Attention Flow in Transformers
*authors:* Samira Abnar, Willem Zuidema
*year:* 2020
*tags:* #attention #multiheaded-attention #attentention-visualization #transformer
*status:* #📦 
*related:*
- [[@cheferTransformerInterpretabilityAttention2021]] (use rollout as a benchmark)
- [[@cheferGenericAttentionmodelExplainability2021]] (their idea is closely related)
*code:*
*review:*

## Notes 📍
- They propose two methods. Attention flow and Rollout attention. The later is intuitive. The first is very computationally demanding.

From [[@cheferTransformerInterpretabilityAttention2021]]:

For comparison, using the same notation, the rollout method [[@abnarQuantifyingAttentionFlow2020]] is given by:
$$
\begin{aligned}
\hat{\mathbf{A}}^{(b)} & =I+\mathbb{E}_h \mathbf{A}^{(b)} \\
\text { rollout } & =\hat{\mathbf{A}}^{(1)} \cdot \hat{\mathbf{A}}^{(2)} \ldots \ldots \cdot \hat{\mathbf{A}}^{(B)}
\end{aligned}
$$
We can observe that the result of rollout is fixed given an input sample, regardless of the target class to be visualized. In addition, it does not consider any signal, except for the pairwise attention scores.

The assumptions in the **rollout method** may be over-simplistic.

## Annotations 📖

“For heads judged to be important, we then attempt to characterize the roles they perform.” ([Voita et al., 2019, p. 5797](zotero://select/library/items/RGLMUHPA)) ([pdf](zotero://open-pdf/library/items/59NJCZ52?page=1&annotation=5DUWDRYB))

“We observe the following types of role: positional (heads attending to an adjacent token), syntactic (heads attending to tokens in a specific syntactic dependency relation) and attention to rare words (heads pointing to the least frequent tokens in the sentence)” ([Voita et al., 2019, p. 5797](zotero://select/library/items/RGLMUHPA)) ([pdf](zotero://open-pdf/library/items/59NJCZ52?page=1&annotation=XCFY4RUG))

“Previous work analyzing how representations are formed by the Transformer’s multi-head attention mechanism focused on either the average or the maximum attention weights over all heads (Voita et al., 2018; Tang et al., 2018), but neither method explicitly takes into account the varying importance of different heads.” ([Voita et al., 2019, p. 5798](zotero://select/library/items/RGLMUHPA)) ([pdf](zotero://open-pdf/library/items/59NJCZ52?page=2&annotation=Q7D7HL7U))

“but neither method explicitly takes into account the varying importance of different heads.” ([Voita et al., 2019, p. 5798](zotero://select/library/items/RGLMUHPA)) ([pdf](zotero://open-pdf/library/items/59NJCZ52?page=2&annotation=37XDUGQA))