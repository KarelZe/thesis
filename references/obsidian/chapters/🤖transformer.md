
![[classical_transformer_architecture.png]]
(own drawing after [[@daiTransformerXLAttentiveLanguage2019]], use L instead of N)

In the subsequent sections we introduce the classical Transformer of [[@vaswaniAttentionAllYou2017]]. Our focus on introducing the central building blocks like self-attention and multi-headed attention.  We then transfer the concepts to the tabular domain by covering [[🤖TabTransformer]] and [[🤖FTTransformer]]. Throughout the work we adhere to a notation suggested in [[@phuongFormalAlgorithmsTransformers2022]].

- encoder/ decoder models $\approx$ sequence-to-sequence model
- both encoders and decoders can be used separately. Might name prominent examples.
- note the practical effect of cot

Cross-check understanding against:
- https://www.baeldung.com/cs/transformer-text-embeddings
- Check my understanding of transformers with https://huggingface.co/course/chapter1/5?fw=pt
- I like how to describe the architecture from a coarse-level to a very fine level. Especially, how it's done visually. Could be helpful for my own explanations as well.
- http://nlp.seas.harvard.edu/annotated-transformer/
- a bit of intuition why it makes sense https://blog.ml6.eu/transformers-for-tabular-data-hot-or-not-e3000df3ed46
- https://ai.stanford.edu/blog/contextual/
- https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3


Components:
[[🛌Token Embedding]]
[[🧵Positional encoding]]
[[🅰️Attention]]
[[🎱Point-wise FFN]]

Specialized variants:
[[🤖TabTransformer]]
[[🤖FTTransformer]]

Open:
- [ ] Attention
- [ ] Research the intuition behind attention
- [ ] Self-attention / multi-headed attention
- [ ] Residual connections
- [ ] Layer Norm, Pre-Norm, and Post-Norm
- [x] TabTransformer
- [ ] FTTransformer
- [ ] Pre-Training
- [ ] Embeddings of categorical / continuous data
- [ ] Selection of supervised approaches
- [ ] Selection of semi-supervised approaches


- Tabular data is different from ... due to being invariant to ...
- What is the purpose of the encoder and the decoder? Introduce the term contextualized embeddings thoroughly.


## Architecture
Here, the encoder maps an input sequence of symbol representations $\left(x_{1}, \ldots, x_{n}\right)$ to a sequence of continuous representations $\mathrm{z}=\left(z_{1}, \ldots, z_{n}\right)$. Given $\mathrm{z}$, the decoder then generates an output sequence $\left(y_{1}, \ldots, y_{m}\right)$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1 respectively.


Encoder: The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm $(x+$ Sublayer $(x))$, where Sublayer $(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text {model }}=512$.

Decoder: The decoder is also composed of a stack of $N=6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.


Visualization of norm-first and norm last (similar in [[@xiongLayerNormalizationTransformer2020]]):
![[layer-norm-first-last.png]]
![[norm-first-norm-last-big-picture.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)

Layer norm / batch norm / instance norm:
![[layer-batch-instance-norm.png]]
![[viz-of image-embedding.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)


## Multiheaded Attention
- What is the effect of multi-headed attention?
https://transformer-circuits.pub/2021/framework/index.html


## Point-wise Feed-Forward Networks

[[🧵Positional encoding]]


## Optimizer
- Adam


- Go "deep" instead of wide
- Explain how neural networks can be adjusted to perform binary classification.
- use feed-forward networks to discuss central concepts like loss function, back propagation etc.
- Discuss why plain vanilla feed-forward networks are not suitable for tabular data. Why do the perform poorly?
- How does the chosen layer and loss function to problem framing
- How are neural networks optimized?
- Motivation for Transformers
- For formal algorithms on Transformers see [[@phuongFormalAlgorithmsTransformers2022]]
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://www.youtube.com/watch?v=EixI6t5oif0
- https://transformer-circuits.pub/2021/framework/index.html
- On efficiency of transformers see: https://arxiv.org/pdf/2009.06732.pdf
- Mathematical foundation of the transformer architecture: https://transformer-circuits.pub/2021/framework/index.html
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- On implementation aspects see: https://arxiv.org/pdf/2007.00072.pdf
- batch nromalization is not fully understood. See [[@zhangDiveDeepLearning2021]] (p. 277)
- https://e2eml.school/transformers.html

feature importance evaluation is a non-trivial problem due to missing ground truth. See [[@borisovDeepNeuralNetworks2022]] paper for citation
- nice visualization / explanation of self-attention. https://peltarion.com/blog/data-science/self-attention-video

- intuition behind multi-head and self-attention e. g. cosine similarity, key and querying mechanism: https://www.youtube.com/watch?v=mMa2PmYJlCo&list=PL86uXYUJ7999zE8u2-97i4KG_2Zpufkfb





- General Introduction: [[@vaswaniAttentionAllYou2017]]
- What is Attentition?
- What is the difference between LSTMs and Transformers? Why are transformers preferable?

- discuss the effects of layer pre-normalization vs. post-normalization (see [[@tunstallNaturalLanguageProcessing2022]])

## Notes from Huggingface 🤗
https://huggingface.co/course/chapter1/4
-   **Encoder (left)**: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
-   **Decoder (right)**: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.


#### Notes on Talk with Łukasz Kaiser 

- https://www.youtube.com/watch?v=rBCqOTEfxvg

- RNNs suffer from vanishing gradients
- Some people used CNNs, but path length is still logarithmic (going down a tree). Is limited to position.
- Attention: make a query with your vector and look at similar things in the past. Looks at everything, but choose things, that are similar.
- Encoder attention allows to go from one word to another. (Encoder Self-Attention)
- MaskedDecoder Self-Attention (is a single matrix multiply with a mask) to mask out all prev. elements not relevant
- Attention A(Q, K, V) (q = query vector) (K, V matrices= memory) (K = current word working on, V = all words generated before). You want to use q to find most similar k and get values that correspond to keys. (QK^T) gives a probability distribution over keys, which is then multiplied with values
- n^2 * d complexity
- to preserve the order of words they use multi-head attention
- attention heads can be interpreted (see winograd schemas)

#### Attention in Dive into Deep Learning (Zhang et al)
- framework for designing attention mechanisms consists of:
    - volitional (~free) cues = queries
    - sensory inputs  = keys
    - nonvolitional cue of sensory input = keys
- attention pooling mechanism  enables a given query (volitional cue) to interact with keys (nonvolitional cues) which guides a biased selection over values (sensory inputs)

- self attention enjoys both parallel computation and the shortest maximum path length. Which makes it appealing to design deep architectures by using self-attention. Do not require a convolutional layer or recurrent layer.

- It's an instance of an encoder-decoder architecture. Input and output sequence embeddings are added with positional encoding before being fed into the encoder and the decoder that stack modules based on self-attention.

## Advantages over LSTM
- More hardware friendly LSTMs require 4 linear layers for the gates and the cell state, for every timestep.


## Points to consider
- consistst of encoder and decoder
- uses an attention mechanism
- No need for recusive loops as with RNN and LSTM
- faster processing due to parallelization
- handle long-term depndencies
- decoder takes a sequence as an input, parallel processing inside encoder, input for the decoder, output of a sequence
- the sequence length of encoder and decoder is flexible (compare translation)
- the decoder processes autoregressive, meaning it considers previous outputs $y_i$, output before $y_i < y_j$.
- components.
	- encoder:
		- consists of 6 encoder blocks. Every encoder block consits of multi-head atttention and fully connected layers, and a normalization layer
		- residual connection?
	- self-attention
		- key mechanism to transfer sequences
		- from a sequence with variable size $x$ onto a sequence with the same size $I$ with the property ...