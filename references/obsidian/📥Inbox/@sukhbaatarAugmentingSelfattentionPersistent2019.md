*title:* Augmenting Self-attention with Persistent Memory
*authors:* Sainbayar Sukhbaatar, Edouard Grave, Guillaume Lample, Herve Jegou, Armand Joulin
*year:* 2019
*tags:* #transformer #layernorm #residual-connections
*status:* #📦 
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖

(23/01/2023, 13:27:46)

“Both the multi-head self-attention and the feed-forward layer are followed by an add-norm operation. This transformation is simply a residual connection [17] followed by layer normalization [23]. The layer normalization computes the average and standard deviation of the output activations of a given sublayer and normalizes them accordingly. This guarantees that the input yt of the following sublayer is well conditioned, i.e., that yT t 1 = 0 and yT t yt = √d.” ([Sukhbaatar et al., 2019, p. 3](zotero://select/library/items/A7XG93GC)) ([pdf](zotero://open-pdf/library/items/I76F65M4?page=3&annotation=AZJKFERG))

“More precisely, the AddNorm operation is defined as: AddNorm(xt) = LayerNorm(xt + Sublayer(xt)), (6) where Sublayer is either a multi-head self-attention or a feedforward sublayer.” ([Sukhbaatar et al., 2019, p. 3](zotero://select/library/items/A7XG93GC)) ([pdf](zotero://open-pdf/library/items/I76F65M4?page=3&annotation=DZQX69NL))

“Transformer layer. The overall transformer layer has the following set of equations: zt = AddNorm(MultiHead(xt)), (7) yt = AddNorm(FF(zt)), (8) where MultiHead is the multi-head self-attention sublayer. This is shown on the left panel of Fig.” ([Sukhbaatar et al., 2019, p. 4](zotero://select/library/items/A7XG93GC)) ([pdf](zotero://open-pdf/library/items/I76F65M4?page=4&annotation=6G4EEYF3))

“These vectors are added to capture information that does not depend on the immediate context, like general knowledge about the task. They are shared across the data and, in some sense, forms a persistent memory similar to the feedforward layer. Therefore we call them persistent vectors. More precisely, the persistent vectors are a set of N pairs of key-value vectors, respectively stacked in two dh × N dimensional matrices Mk and Mv. As discussed in Section 4.1, Mk and Mv can be interpreted as V and U of a feedforward sublayer” ([Sukhbaatar et al., 2019, p. 4](zotero://select/library/items/A7XG93GC)) ([pdf](zotero://open-pdf/library/items/I76F65M4?page=4&annotation=XJVKDKN9))

“The right panel of Fig. 1 summarize the all-attention layer in the case of a single head: we remove the feedforward sublayer and add unconditioned persistent vectors to the self-attention sublayer. While the persistent vectors are directly comparable to a feedforward sublayer in the case of a single head, a multi-head version is more comparable to multiple small feedforward layers working in parallel.” ([Sukhbaatar et al., 2019, p. 5](zotero://select/library/items/A7XG93GC)) ([pdf](zotero://open-pdf/library/items/I76F65M4?page=5&annotation=XJ39NBIL))

“It extends the self-attention layer of a transformer with a set of persistent vectors that are capable of storing information that is complementary to the short term information in contexts. We also show that these persistent vectors can replace the feedforward layers in a transformer network with no loss of performance.” ([Sukhbaatar et al., 2019, p. 9](zotero://select/library/items/A7XG93GC)) ([pdf](zotero://open-pdf/library/items/I76F65M4?page=9&annotation=VSNQEW6M))