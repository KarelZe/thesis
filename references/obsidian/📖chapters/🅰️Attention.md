Recall from our discussion on token embeddings, that embeddings are not yet context-sensitive. The Transformer relies on an *attention mechanism* to let tokens gather information from other tokens within the sequence and thereby encode the context onto the embeddings.

**Attention**
Attention can be thought of as a mapping between a query and a set of key-value pairs to an output. In general, the current token is first projected onto a query vector, and all tokens in the context are mapped to key and value vectors. Similar to a soft dictionary lookup, the goal is to retrieve the values from tokens in the context for which the keys are similar to the query and return an aggregate estimate of the values weighted by the similarity of the keys and the query. Naturally, if a token in the context is important for predicting the queried token, indicated by a high similarity, the value of the context token has a large contribution to the output. ~~The output is a contextualized version of the query. ~~([[@phuongFormalAlgorithmsTransformers2022]]5) and ([[@vaswaniAttentionAllYou2017]]3)

Attention first appeared in ([[@bahdanauNeuralMachineTranslation2016]]4) and was popularized by ([[@vaswaniAttentionAllYou2017]]4). The latter introduced an attention mechanism known as *scaled dot-product attention*, which we cover in detail.

**Scaled Dot-Product Attention**
Analogous to before, *scaled dot-product attention* estimates the similarity between queries and keys, now as the dot product. The resulting attention scores are divided by $\sqrt{d_{\text{attn}}}$, and normalized using a softmax function to obtain the attention weights. A multiplication of the attention weights with the values yields the outputs. 

For computational efficiency, attention is performed simultaneously over multiple queries. Thus, the authors group queries, keys, and values in matrices. In matrix notation outputs are estimated as:
$$
\tilde{\mathbf{V}} = \mathbf{V} \operatorname{softmax}\left(\frac{1}{\sqrt{d_{\text{attn}}}}\mathbf{K}^{T} \mathbf{Q}\right)
$$


The scaling factor is int
Softmax

Instead of considering a single query, key, and value at a time, the authors perform attention on matrices, where multiple queries, keys, and values into matrices. 

and divides by some scaling factor. The so-obtained normalize the so obtained attention weights using the softmax function and multiply with the values to obtain the output.


Here, d

Like 

row-wise

Scaled dot-product attention, requires queries, keys, and, and values. 

Typically, 


The similarity between the queries, and the 

([[@vaswaniAttentionAllYou2017]]4) introduce a 

The attention weights are estimated as the dot product between Q and K. 


“The attention matrix A = QK> is chiefly responsible for learning alignment scores between tokens in the sequence. In this formulation, the dot product between each element/token in the query (Q) and key (K) is taken. This drives the self-alignment process in self-attention whereby tokens learn to gather from each other.” (Tay et al., 2022, p. 4)

The attention mechanism 


![[Pasted image 20230419123045.png]]
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22]. End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34]. To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].


**Multi-head Attention**
Rather than relying on a single attention function, ([[@vaswaniAttentionAllYou2017]]4--5) introduce multiple *attention heads*, which perform attention in parallel on *different* linear projections of the queries, keys, and values. The *multi-head attention* enables the model to learn richer representations of the input, as attention heads operate independently, they can pick up unique patterns or focus on different positions in the sequence at once. 

Examplary for machine translation, ([[@voitaAnalyzingMultiHeadSelfAttention2019]]5795) show, that heads serve distinct purposes like learning positional or syntactic relations between tokens. In practice, Transformers may not leverage all attention heads and some heads could even be pruned without impacting the performance (cp. [[@michelAreSixteenHeads2019]]9) ([[@voitaAnalyzingMultiHeadSelfAttention2019]]5805). We come back to these observations in cref-[[🧭Attention map]].

footnote-(Split is only done logically with each of the attention heads operating on the same data matrix, but in different subsections.)


(How it's done + formula)

$$
\operatorname{MHAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{W}_{O}\left[\mathbf{Y}^{1};\mathbf{Y}^{2};\ldots;\mathbf{Y}^{H} \right] + \mathbf{b}_{O}\mathbf{1}^{\top}
$$
$$
\mathbf{Y}^{h} = \operatorname{Attention}() 
$$
![[Pasted image 20230419114118.png]]
![[Pasted image 20230419114045.png]]

![[Pasted image 20230419110346.png]]
![[Pasted image 20230419110505.png]]
![[Pasted image 20230419112529.png]]
<mark style="background: #FFF3A3A6;">“However, to keep the number of parameters constant, dh is typically set to d Nh , in which case MHA can be seen as an ensemble of low-rank vanilla attention layers2. In the following, we use Atth(x) as a shorthand for the output of head h on input x.” (Michel et al., 2019, p. 2)</mark>

<mark style="background: #FFB86CA6;">“Multi-head attention. In the Transformer, instead of performing a single attention function with dmodel-dimensional keys, values and queries, one linearly projects the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. Attention is applied to each of these projected versions of queries, keys and values in parallel, yielding dvdimensional output values. These are concatenated and once again projected, resulting in the final values. This mechanism is known as multi-head attention.” (Kitaev et al., 2020, p. 2)</mark>

Finally, the output of the $h$ individual attention heads is concatenated and projected back to output dimension. The output of the attention layer is then passed to the point-wise feed-forward networks, which enables interaction between the head's outputs. We discuss position-wise feed-forward networks in cref-[[🎱Position-wise FFN]].

**Masked Self-Attention and Cross-Attention**
In the cref-attention above, tokens can attend to any preceding or subsequent token without restrictions. Thus, the full *bidirectional context* is used. This design is optimal for the encoder, where the entire input sequence shall serve as the context. 

For the decoder, the self-attention is modified to *masked self-attention* and *cross-attention* mechanism. First, a causal masking is required to achieve the autoregressive sequence generation in the decoder. The context is now *unidirectional*, where a token is only allowed to attend to itself or all previously generated tokens. Second, the decoder uses *cross-attention* to connect between the encoder and decoder. Other than in the self-attention mechanism, where keys, values and queries are generated from same sequence, key and values come from the encoder and queries are provided by the decoder (see cref-fig). As our focus is on encoder-only architectures, we refer the reader to ([[@raffelExploringLimitsTransfer2020]]16--17) or ([[@vaswaniAttentionAllYou2017]]5) for an in-depth treatment on both topics.

**Notes:**
[[🅰️attention notes]]

