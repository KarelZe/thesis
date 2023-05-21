*title:* Reformer: The Efficient Transformer
*authors:* Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya
*year:* 2020
*tags:* #transformer #self-attention #multiheaded-attention #efficinece
*status:* #📦 
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖

“The above estimate includes only per-layer memory and input activations cost and does not take into account the following major sources of memory use in the Transformer. • Memory in a model with N layers is N -times larger than in a single-layer model due to the fact that activations need to be stored for back-propagation. • Since the depth dff of intermediate feed-forward layers is often much larger than the depth dmodel of attention activations, it accounts for a large fraction of memory use. • Attention on sequences of length L is O(L2) in both computational and memory complexity, so even for a single sequence of 64K tokens can exhaust accelerator memory.” ([Kitaev et al., 2020, p. 1](zotero://select/library/items/D93TNTMS)) ([pdf](zotero://open-pdf/library/items/5F5L22PR?page=1&annotation=GGWJFMPX))

“Dot-product attention. The standard attention used in the Transformer is the scaled dot-product attention (Vaswani et al., 2017). The input consists of queries and keys of dimension dk, and values of dimension dv. The dot products of the query with all keys are computed, scaled by √dk, and a softmax function is applied to obtain the weights on the values. In practise, the attention function on a set of queries is computed simultaneously, packed together into a matrix Q. Assuming the keys and values are also packed together into matrices K and V , the matrix of outputs is defined as: Attention(Q, K, V ) = softmax( QKT √dk )V” ([Kitaev et al., 2020, p. 2](zotero://select/library/items/D93TNTMS)) ([pdf](zotero://open-pdf/library/items/5F5L22PR?page=2&annotation=YLGCST6K))

“Multi-head attention. In the Transformer, instead of performing a single attention function with dmodel-dimensional keys, values and queries, one linearly projects the queries, keys and values h times with different, learnt linear projections to dk, dk and dv dimensions, respectively. Attention is applied to each of these projected versions of queries, keys and values in parallel, yielding dvdimensional output values. These are concatenated and once again projected, resulting in the final values. This mechanism is known as multi-head attention.” ([Kitaev et al., 2020, p. 2](zotero://select/library/items/D93TNTMS)) ([pdf](zotero://open-pdf/library/items/5F5L22PR?page=2&annotation=P89D5VB5))