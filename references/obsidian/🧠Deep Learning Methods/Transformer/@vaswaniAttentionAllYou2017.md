
title: Attention Is All You Need
authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
year: 2017
tags:


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
