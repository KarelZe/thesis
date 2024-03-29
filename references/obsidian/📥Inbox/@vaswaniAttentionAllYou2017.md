*title:* Attention is all you need
*authors:* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, \L{}ukasz Kaiser, Illia Polosukhin
*year:* 2016
*tags:* #transformer #encoder #attention #sequence-modelling
*status:* #📥
*related:*
- [[@arikTabnetAttentiveInterpretable2020]]
- [[@huangTabTransformerTabularData2020]]
- [[@gorishniyRevisitingDeepLearning2021]]
# Notes 
- On a high level network consists of encoder and decoder
- Multi-headed attention is the key component of transformer
- Input is send into three different inputs to the multiheaded attention: values, keys, queries.
    - For first input it's the same for key, values and queries
- Also has step connections (similar to resnet)
- output of encoder is sent to so multiheaded attention  in the decoder
- decoder is similarly structured (composed of transformer) + other parts multi-headed and Add & Norm
- Decoder and Encoder block can be repeated multiple times, before it is being passed to decoder
- Positional Encoding is responsible for positional encoding as transformer is positional invariant
- part of the input will be masked, to prevent the decoder from learning simple mapping
- first then input is split into chunks. A scaled dot-product attention is derived from the queries keys and the values. This is sent to a linear layer which is then the output of the multi-headed attention

# Annotations

- cover dot-product attention and sequential attention
- multi-headed attention
- self-attention