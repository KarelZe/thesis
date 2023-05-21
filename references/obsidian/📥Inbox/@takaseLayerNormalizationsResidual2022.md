*title:* On layer normalizations and residual connections in transformers
*authors:* Sho Takase, Shun Kiyono, Sosuke Kobayashi, Jun Suzuki
*year:* 2022
*tags:* #transformer #layernorm #pre-norm #post-norm
*status:* #📦 
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖
“the impact of the layer normalisation positions [32, 33]. There are currently two major layer normalisation positions in Transformers: Pre-Layer Normalisation (Pre-LN) and Post-Layer Normalisation (Post-LN). Pre-LN applies the layer normalisation to an input for each sub-layer, and Post-LN places the layer normalisation after each residual connexion. The original Transformer [28] employs PostLN. However, recent studies often suggest using Pre-LN [32, 2, 5] because the training in Post-LN with deep Transformers (e.g., ten or more layers) often becomes unstable, resulting in useless models. Figure 1 shows an actual example; loss curves of training 18L-18L Transformer encoder-decoders on a widely used WMT English-to-German machine translation dataset. Here, XL-Y L represents the number of layers in encoder and decoder, where X and Y correspond to encoder and decoder, respectively. These figures clearly show that 18L-18L Post-LN Transformer encoder-decoder fails to train the model. However, in contrast, Liu et al. [13] reported that Post-LN consistently achieved better performance than Pre-LN in the machine translation task when they used 6L-6L (relatively shallow) Transformers.” ([Takase et al., 2022, p. 2](zotero://select/library/items/9867KZ4B)) ([pdf](zotero://open-pdf/library/items/JBR6QM9N?page=2&annotation=IGTZ6SLR))
