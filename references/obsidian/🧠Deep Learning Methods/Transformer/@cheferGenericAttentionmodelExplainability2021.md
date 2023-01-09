*title:* Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers
*authors:* Hila Chefer, Shir Gur, Lior Wolf
*year:* 2021
*tags:* #transformer #self-attention #multiheaded-attention #attention-visualization #multiheaded-attention 
*status:* #📦 
*related:*
- [[@cheferTransformerInterpretabilityAttention2021]] (previous paper. This paper is more generic and seems to replace the previous paper which is more complex)
*code:*
https://www.youtube.com/watch?time_continue=7&v=bQTL34Dln-M&feature=emb_logo
https://github.com/hila-chefer/Transformer-MM-Explainability

*review:*

## Notes 📍
Since in the beginning, each token contains only information from itself:
$$
\begin{array}{ll}
\mathbf{R}^{i i}=\mathbb{1}^{i \times i}, & \mathbf{R}^{t t}=\mathbb{1}^{t \times t} \\
\mathbf{R}^{i t}=\mathbf{0}^{i \times t}, & \mathbf{R}^{t i}=\mathbf{0}^{t \times i}
\end{array}
$$
As attention layers mix tokens, R matrix helps keep track how attention is mixed. Each attention layer adds context from other tokens, for example for a **self-attention layer**:
$$
\mathbf{R}^{q q} \leftarrow \mathbf{R}^{q q}+\overline{\mathbf{A}} \cdot \mathbf{R}^{q q}
$$
For each attention layer, they integrate the gradients and attention maps to average across attention heads.
![[agg-attention-heads.png]]

Requires a single forward pass only. 
## Annotations 📖
“In this work, we propose the first method to explain prediction by any Transformer-based architecture, including bi-modal Transformers and Transformers with co-attentions. We provide generic solutions and apply these to the three most commonly used of these architectures: (i) pure selfattention, (ii) self-attention combined with co-attention, and (iii) encoder-decoder attention.” ([Chefer et al., 2021, p. 1](zotero://select/library/items/76APCGJW)) ([pdf](zotero://open-pdf/library/items/3SMPJ6RN?page=1&annotation=VFSGSXNM))

“a residual connection, as shown in Fig. 1, we accumulate the relevancies by adding each layer’s contribution to the aggregated relevancies, similar to [1] in which the identity matrix is added to account for residual connections.” ([Chefer et al., 2021, p. 3](zotero://select/library/items/76APCGJW)) ([pdf](zotero://open-pdf/library/items/3SMPJ6RN?page=3&annotation=GH9YZQXU))

“Our method uses the attention map A of each attention layer to update the relevancy maps. Since each such map is comprised of h heads, we follow [5] and use gradients to average across heads. Note that Voita et al. [41] show that attention heads differ in importance and relevance, thus a simple average across heads results in distorted relevancy maps. The final attention map ̄ A ∈ Rs×q of our method is then defined as follows: ̄ A = Eh((∇A A)+) (5) where is the Hadamard product, ∇A := ∂yt ∂A for yt which is the model’s output for the class we wish to visualize t, and Eh is the mean across the heads dimension. Following [5] we remove the negative contributions before averaging.” ([Chefer et al., 2021, p. 3](zotero://select/library/items/76APCGJW)) ([pdf](zotero://open-pdf/library/items/3SMPJ6RN?page=3&annotation=VZZ6LEUF))

“For self-attention layers that satisfy ̄ A ∈ Rs×s the update rules for the affected aggregated relevancy scores are: Rss = Rss + ̄ A · Rss (6) Rsq = Rsq + ̄ A · Rsq (7)” ([Chefer et al., 2021, p. 3](zotero://select/library/items/76APCGJW)) ([pdf](zotero://open-pdf/library/items/3SMPJ6RN?page=3&annotation=TQE6XEC8))

“In Eq. 6 we account for the fact that the tokens were already contextualized in previous attention layers by applying matrix multiplication with the aggregated self-attention matrix Rss, as done in [1, 5].” ([Chefer et al., 2021, p. 3](zotero://select/library/items/76APCGJW)) ([pdf](zotero://open-pdf/library/items/3SMPJ6RN?page=3&annotation=TH74KCXU))

“Since we initialized Rxx = Ix×x, and Eq. 6 accumulates the relevancy matrices at each layer, we can consider an aggregated self-attention matrix Rxx as a matrix comprised of two parts, the first is the identity matrix from the initialization, and the second, ˆ Rxx = Rxx − Ix×x is the matrix created by the aggregation of self-attention across the layers” ([Chefer et al., 2021, p. 3](zotero://select/library/items/76APCGJW)) ([pdf](zotero://open-pdf/library/items/3SMPJ6RN?page=3&annotation=RNJEQV7K))

“Since Eq. 5 uses gradients to average across heads, the values of ˆ Rxx are typically reduced. We wish to account equally both for the fact that each token influences itself and for the contextualization by the selfattention mechanism. Therefore, we normalize each row in ˆ Rxx so that it sums to 1. Intuitively, row i in ˆ Rxx disclosed the self-attention value of each token w.r.t. the i-th token, and the identity matrix Ix×x sets that value for each token” ([Chefer et al., 2021, p. 3](zotero://select/library/items/76APCGJW)) ([pdf](zotero://open-pdf/library/items/3SMPJ6RN?page=3&annotation=56VWVPGT))
