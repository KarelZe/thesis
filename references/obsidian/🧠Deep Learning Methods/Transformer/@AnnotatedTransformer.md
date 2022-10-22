
title: The Annotated Transformer
authors: Havard NLP research group
year: 2018


Tags: #Transformer #positional-encoding

- Learning dependencies between distant positions is difficult. Transformers reduce it to a constant number of operations at the cost of reduced effective resoultion.
- Self-attention is an attention mechanism elating different positions of a single sequence in order to compute a representation of the sequence.
- The basic architecture consists of a encoder-decoder structure. Most competitive neural sequence transduction models have an encoder-decoder structure (cite). Here, the encoder maps an input sequence of symbol representations $\left(x_{1}, \ldots, x_{n}\right)$ to a sequence of continuous representations $z=\left(z_{1}, \ldots, z_{n}\right)$. Given $\mathbf{z}$, the decoder then generates an output sequence $\left(y_{1}, \ldots, y_{m}\right)$ of symbols one element at a time.
- The transformer follows this overall architecture using stacked self-attention and pint-wise, fully connected layers for both the encoder and decoder.

- The attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values and output are all vectors.


## Positional encoding
- since the model contains no recurrence and no convolution, in order to make use of the order of the sequence, some information about the relative or absolute position of the tokens in the sequence must be injected.
- Therefore positional encodings are added to the input embeddings at the bottom of the encoder and decoder stacks.
- They chose the sinusoidal version because it allows the model to extrapolate to sequence lengths longer than the ones encountered during training