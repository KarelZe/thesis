*title:* Transformers without Tears: Improving the Normalisation of Self-Attention
*authors:* Toan Q. Nguyen, Julian Salazar
*year:* 2019
*tags:* #transformer #layernorm #residual-connections 
*status:* #📦 
*related:*
*code:*
- Interesting presentation: https://tnq177.github.io/data/transformers_without_tears.pdf
*review:*

## Notes 📍

## Annotations 📖

“Our work demonstrates additional consequences in the base (≤6-layer encoder) Transformer regime. We show that PRENORM enables warmup-free, validation-based training with large learning rates even for small batches, in contrast to past work on scaling NMT (Ott et al., 2018). We also partly reclaim POSTNORM’s stability via smaller initializations, although PRENORM is less sensitive to this magnitude and can improve performance. However, despite PRENORM’s recent adoption in many NMT frameworks, we find it degrades base Transformer performance on WMT '14 English-German.” ([Nguyen and Salazar, 2019, p. 1](zotero://select/library/items/EDLX35I6)) ([pdf](zotero://open-pdf/library/items/2PAADYR7?page=1&annotation=HQ986YIT))

“Residual connections (He et al., 2016a) were first introduced to facilitate the training of deep convolutional networks, where the output of the \`-th layer F\` is summed with its input: x\`+1 = x\` + F\`(x\`). (1) The identity term x\` is crucial to greatly extending the depth of such networks (He et al., 2016b). If one were to scale x\` by a scalar λ\`, then the contribution of x\` to the final layer FL is (∏L−1 i=\` λi)x\`. For deep networks with dozens or even hundreds of layers L, the term ∏L−1 i=\` λi becomes very large if λi > 1 or very small if λi < 1, for enough i. When backpropagating from the last layer L back to \`, these multiplicative terms can cause exploding or vanishing gradients, respectively. Therefore they fix λi = 1, keeping the total residual path an identity map.” ([Nguyen and Salazar, 2019, p. 2](zotero://select/library/items/EDLX35I6)) ([pdf](zotero://open-pdf/library/items/2PAADYR7?page=2&annotation=NCSKYHZT))

“We conjecture this has caused past convergence failures (Popel and Bojar, 2018; Shazeer and Stern, 2018), with LAYERNORMs in the residual path acting similarly to λi 6= 1; furthermore, warmup was needed to let LAYERNORM safely adjust scale during early parts of training.” ([Nguyen and Salazar, 2019, p. 2](zotero://select/library/items/EDLX35I6)) ([pdf](zotero://open-pdf/library/items/2PAADYR7?page=2&annotation=RQP4GPTS))

“Inspired by He et al. (2016b), we apply LAYERNORM immediately before each sublayer (PRENORM): x\`+1 = x\` + F\`(LAYERNORM(x\`)). (3) This is cited as a stabiliser for Transformer training (Chen et al., 2018; Wang et al., 2019) and is already implemented in popular toolkits (Vaswani et al., 2018; Ott et al., 2019; Hieber et al., 2018), though not necessarily used by their default recipes. Wang et al. (2019) make a similar argument to motivate the success of PRENORM in training very deep Transformers. Note that one must append an additional normalisation after both encoder and decoder so their outputs are appropriately scaled.” ([Nguyen and Salazar, 2019, p. 2](zotero://select/library/items/EDLX35I6)) ([pdf](zotero://open-pdf/library/items/2PAADYR7?page=2&annotation=HGX5V3N6))
