Recall from our discussion on token embeddings, that embeddings are not yet context-sensitive. The Transformer relies on an *attention mechanism* to let tokens gather information from other tokens within the sequence and thereby encode the context onto the embeddings.

**Attention**
Attention can be thought of as a mapping between a query and a set of key-value pairs to an output. In general, the current token is first projected onto a query vector, and all tokens in the context are mapped to key and value vectors. Similar to a soft dictionary lookup, the goal is to retrieve the values from tokens in the context for which the keys are similar to the query and return an aggregate estimate of the values weighted by the similarity of the keys and the query. Naturally, if a token in the context is important for predicting the queried token, indicated by a high similarity, the value of the context token has a large contribution to the output.  ([[@phuongFormalAlgorithmsTransformers2022]]5) and ([[@vaswaniAttentionAllYou2017]]3)

Attention first appeared in ([[@bahdanauNeuralMachineTranslation2016]]4) and was popularized by ([[@vaswaniAttentionAllYou2017]]4). The latter introduced an attention mechanism, known as *scaled dot-product attention*, which we cover in detail.

**Scaled Dot-Product Attention**
Analogous to before, *scaled dot-product attention* estimates the similarity between queries and keys, now as the dot product. The resulting attention scores are divided by some constant, and normalized using a softmax function to obtain the attention weights. A multiplication of the attention weights with the values yields the outputs. Scaled dot-product attention is visualized in cref-fig (left).

For computational efficiency, attention is performed simultaneously over multiple queries. Thus, the authors group queries, keys, and values in matrices. In matrix notation outputs are estimated as:
$$
\begin{align}
\operatorname{Attention}(\mathbf{X},\mathbf{Z}) &= \mathbf{V} \operatorname{softmax}\left(\mathbf{S} / \sqrt{d_{attn}}\right)\\
 \mathbf{S} &= \mathbf{K}^{\top} \mathbf{Q}
\end{align}
$$
where $\mathbf{X} \in \mathbb{R}^{d_X\times \ell_X}$ and $\mathbf{Z} \in \mathbb{R}^{d_Z\times \ell_Z}$  are vector representation the primary input sequence and of the context sequence. Both the primary and the context sequence are identical for the encoder, but are different for the decoder. The query, key, and value matrices $\mathbf{Q}=\mathbf{W}_q \mathbf{X} + \mathbf{b}_q\mathbf{1}^{\top}$, $\mathbf{K}=\mathbf{W}_k \mathbf{Z} + \mathbf{b}_k\mathbf{1}^{\top}$, and $\mathbf{V}=\mathbf{W}_v \mathbf{Z} + \mathbf{b}_v\mathbf{1}^{\top}$ are linear projections of the input and context sequences, and $\mathbf{W}_q, \mathbf{W}_k \in \mathbb{R}^{d_{\mathrm{attn}\times d_{X}}}$; $\mathbf{W}_v \in \mathbb{R}^{d_{\mathrm{dout}\times d_{Z}}}$; $\mathbf{b}_q, \mathbf{b}_k \in \mathbb{R}^{d_{\mathrm{attn}}}$, and $\mathbf{b}_v \in \mathbb{R}^{d_{\mathrm{dout}}}$ are learnable parameters. The dimensionality of the attention mechanism, $d_{attn}$, is typically a fraction of the model dimensionality to accelerate computation. Likewise, the output dimension, $d_{out}$, is another hyperparameter to the models. The attention scores are $\mathbf{S}$, which are scaled by $\sqrt{d_{\text{attn}}}$ to avoid gls-vanishing-gradients, and the softmax activation normalizes all scores to a $[0,1]$ range. As (normalized) attention scores have a clear interpretation as the probability (...), the attention mechanism provides a window into the model. We explore this idea further in cref-[[🧭Attention map]].

**Multi-Head Attention**
Rather than relying on a single attention function, ([[@vaswaniAttentionAllYou2017]]4--5) introduce multiple *attention heads*, which perform attention in parallel on $H$ *different* linear projections of the queries, keys, and values. The *multi-head attention* enables the model to learn richer representations of the input, as attention heads operate independently, they can pick up unique patterns or focus on different positions in the sequence at once. Multi-head attention is visualized in cref-fig (centre).

Exemplary for machine translation, ([[@voitaAnalyzingMultiHeadSelfAttention2019]]5795) show, that heads serve indeed distinct purposes like learning positional or syntactic relations between tokens. For tabular data this could transfer to dependencies between features. In practice, Transformers may not leverage all attention heads and some heads could even be pruned without impacting the performance ([[@michelAreSixteenHeads2019]]9) ([[@voitaAnalyzingMultiHeadSelfAttention2019]]5805).

Multi-head attention can be computed as:
$$
\begin{align}
\operatorname{MHAttention}(\mathbf{X}, \mathbf{Z}) &= \mathbf{W}_{o}\left[\mathbf{Y}^{1};\mathbf{Y}^{2};\ldots;\mathbf{Y}^{H} \right] + \mathbf{b}_{o}\mathbf{1}^{\top}\\
\mathbf{Y}^{h} &= \operatorname{Attention}(\mathbf{Q}^h, \mathbf{K}^h, \mathbf{V}^h) 
\end{align}
$$

The query, key, and value matrices  $\mathbf{Q}^{h}=\mathbf{W}^h_q \mathbf{X} + \mathbf{b}^h_q\mathbf{1}^{\top}$, $\mathbf{K}^{h}=\mathbf{W}_k^h \mathbf{Z} + \mathbf{b}_k^h\mathbf{1}^{\top}$, and $\mathbf{V}^{h}=\mathbf{W}_v^h \mathbf{Z} + \mathbf{b}_v^h\mathbf{1}^{\top}$ are obtained from linear projections of the input and context sequences unique per head. Again, $\mathbf{W}^{h}_{q} \in \mathbb{R}^{d_{\mathrm{attn}}\times d_{X}}$; $\mathbf{W}^{h}_{k}, \mathbf{W}^{h}_{v} \in \mathbb{R}^{d_{\mathrm{attn}}\times d_Z}$, $\mathbf{b}_q, \mathbf{b}_k \in \mathbb{R}^{d_{\mathrm{attn}}}$; $\mathbf{b}^h_v, \mathbf{b}^h_k \in \mathbb{R}^{d_{\mathrm{attn}}}$, and $\mathbf{b}^h_k \in \mathbb{R}^{d_{\mathrm{mid}}}$ are used for projection. Typically, the dimensionality of the attention heads, $d_{\mathrm{mid}}$, is reduced to $d_{\mathrm{attn}} / H$ to match the computational cost of single-head attention. The output dimensionality $d_{\text{out}}$ is restored with a final linear projection through the weight matrix $\mathbf{W}_{o} \in \mathbb{R}^{d_{\mathrm{out}}\times Hd_{\mathrm{mid}}}$ and bias $\mathbf{b}_o \in \mathbb{R}^{d_{\mathrm{out}}}$ applied to the concatenated results of the attention heads. 

The concatenated, and projected output of the attention layer is then passed to the point-wise feed-forward networks, which enables interaction between the head's outputs. We discuss position-wise feed-forward networks in cref-[[🎱Position-wise FFN]].

**Masked Self-Attention and Cross-Attention**
In the cref-attention above, tokens can attend to any preceding or subsequent token without restrictions. Thus, the full *bidirectional context* is used. This design is optimal for the encoder, where the entire input sequence shall serve as the context. 

For the decoder, the self-attention is modified to *masked self-attention* and *cross-attention* mechanism. First, a causal masking is required to achieve the autoregressive sequence generation in the decoder. The context is now *unidirectional*, where a token is only allowed to attend to itself or all previously generated tokens. Second, the decoder uses *cross-attention* to connect between the encoder and decoder. Other than in the self-attention mechanism, where keys, values and queries are generated from same sequence, key and values come from the encoder and queries are provided by the decoder (see cref-fig). As our focus is on encoder-only architectures, we refer the reader to ([[@raffelExploringLimitsTransfer2020]]16--17) or ([[@vaswaniAttentionAllYou2017]]5) for an in-depth treatment on both topics.

**Notes:**
[[🅰️attention notes]]

