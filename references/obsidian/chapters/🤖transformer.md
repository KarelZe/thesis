
![[classical_transformer_architecture.png]]
(own drawing after [[@daiTransformerXLAttentiveLanguage2019]], use L instead of N, left encoder and right decoder)

In the subsequent sections we introduce the classical Transformer of [[@vaswaniAttentionAllYou2017]]. Our focus on introducing the central building blocks like self-attention and multi-headed attention.  We then transfer the concepts to the tabular domain by covering [[🤖TabTransformer]] and [[🤖FTTransformer]]. Throughout the work we adhere to a notation suggested in [[@phuongFormalAlgorithmsTransformers2022]].

## Points to consider
- encoder/ decoder models $\approx$ sequence-to-sequence model
- both encoders and decoders can be used separately. Might name prominent examples.
- consistst of encoder and decoder
- uses an attention mechanism
- No need for recusive loops as with RNN and LSTM
- faster processing due to parallelization
- handle long-term depndencies
- decoder takes a sequence as an input, parallel processing inside encoder, input for the decoder, output of a sequence
- the sequence length of encoder and decoder is flexible (compare translation)
- the decoder processes autoregressive, meaning it considers previous outputs $y_i$, output before $y_i < y_j$.
- why were Transformers introduced?
- What is the purpose of the encoder and the decoder? Introduce the term contextualized embeddings thoroughly.
- What are the parts of the architecture?
- Introduce pre-norm. What is bad with it? Why should it maybe be adjusted?
- components.
	- encoder:
		- consists of 6 encoder blocks. Every encoder block consits of multi-head atttention and fully connected layers, and a normalization layer
		- residual connection?
	- self-attention
		- key mechanism to transfer sequences
		- from a sequence with variable size $x$ onto a sequence with the same size $I$ with the property ...


%%
Nice visuals: https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers

ResNet paper on residual learning / residual connections. Discusses in general the problems that arise with learning deep neural networks: https://arxiv.org/pdf/1512.03385.pdf
Nice explanation: https://stats.stackexchange.com/a/565203/351242
%%


## Resources
Cross-check understanding against:
- https://www.baeldung.com/cs/transformer-text-embeddings
- Check my understanding of transformers with https://huggingface.co/course/chapter1/5?fw=pt
- I like how to describe the architecture from a coarse-level to a very fine level. Especially, how it's done visually. Could be helpful for my own explanations as well.
- http://nlp.seas.harvard.edu/annotated-transformer/
- a bit of intuition why it makes sense https://blog.ml6.eu/transformers-for-tabular-data-hot-or-not-e3000df3ed46
- https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3
- Explain how neural networks can be adjusted to perform binary classification.
- Discuss why plain vanilla feed-forward networks are not suitable for tabular data. Why do the perform poorly?
- How does the chosen layer and loss function to problem framing
- How are neural networks optimized?
- Motivation for Transformers
- For formal algorithms on Transformers see [[@phuongFormalAlgorithmsTransformers2022]]
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://www.youtube.com/watch?v=EixI6t5oif0
- https://transformer-circuits.pub/2021/framework/index.html
- On efficiency of transformers see: [[🧠Deep Learning Methods/Transformer/@tayEfficientTransformersSurvey2022]] Also good sanity check for own understanding
- Mathematical foundation of the transformer architecture: https://transformer-circuits.pub/2021/framework/index.html
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- batch nromalization is not fully understood. See [[@zhangDiveDeepLearning2021]] (p. 277)

- nice visualization / explanation of self-attention. https://peltarion.com/blog/data-science/self-attention-video
- intuition behind multi-head and self-attention e. g. cosine similarity, key and querying mechanism: https://www.youtube.com/watch?v=mMa2PmYJlCo&list=PL86uXYUJ7999zE8u2-97i4KG_2Zpufkfb
- Our analysis starts from the observation: the original Transformer (referred to as Post-LN) is less robust than its Pre-LN variant2 (Baevski and Auli, 2019; Xiong et al., 2019; Nguyen and Salazar, 2019). (from [[@liuUnderstandingDifficultyTraining2020]])
- motivation to switch 
- General Introduction: [[@vaswaniAttentionAllYou2017]]
- What is Attention?
- discuss the effects of layer pre-normalization vs. post-normalization (see [[@tunstallNaturalLanguageProcessing2022]])

Components:
[[🛌Token Embedding]]
[[🧵Positional encoding]]
[[🅰️Attention]]
[[🎱Point-wise FFN]]

Specialized variants for tabular data:
[[🤖TabTransformer]]
[[🤖FTTransformer]]



## Notes from Harvard🎓
http://nlp.seas.harvard.edu/2018/04/03/attention.html
- self-attention is a an attention mechaism of realting different positions of a single sequence to compute a representation of the sequence.
- Encoder maps an input sequenc of symbol representations (x1, .... xn) to a a seuqence of continuous representations z=(z1, ..., zn). The decoder generates from z an output sequence(y1,...ym) of symbols one element at a time. Each step is auto-regressive as generated symbols are used as additional input when generating text.
- Dropout is applied to the output of each sublayer, before it is added to the sub-layer input and normalized. Requires all sublayers of the model to have a fixed dimensions to make residual connnections (which add elemnt-wisely?) possible.
- Each layer has two sub-layers. The first is a multi-headed self attention mechanism and the second is a simple, point-wise feed-forward network.
- the decoder inserts a third layer, which performs multi-head attention over the output of the encoder stack. Again layer normalization and residual connections are used.
- To prevent the decoder from attending to subsequent  poistions a musk is used, so that the predictions only depend on known outputs at positions less than i.
- 

## Notes from Tunstall
[[@tunstallNaturalLanguageProcessing2022]]
- it's based on the encoder-decoder architecture, which is commonly used in machine translation, where a sequence is translated from language to another.
- **Encoder:** converts an input sequence of tokens into sequence of embedding vectors which is the context.
-  **Decoder:** use encoder's hidden state to iteratively generate an output sequence of tokens (auto-regressively?)
- Both encoder and decoder consists of multiple stacked transformer blocks
- the encoder's output is fed to the decoder layer and the decoder gnerates a prediction for the most probable next token in the sequence. The output is then fed back into the decoder to generate the next token until the end of EOS token is reached.
- Enoder-only architecture that is well suited for text-classification and named-entity recognition. A sequence of text is converted into a rich numerical representation => bidirectional attention
- Other types => decoder-only => autoregressive attention and encoder-decoder => good for machine translation and summarization
- The encoder feeds the input through sublayers:
	- multi-head self-attention layer
	- fully-connected feed-forward layer
- The purpose of the encoder is to update the input embeddings and to produce representations that encode some contextual information of the sequence. Input dimensions and output dimensions are the same. 
- skip connections and layer normalization is used to train deep neural networks efficiently.


## Notes from Rothman
[[@rothmanTransformersNaturalLanguage2021]]
- multi-head attention sub-layer contains eight attention heads and is followed by post-layer normalization, which will add residual connections to the output of the sublayer and normalize
- performing attention using a single head is slow. Given the size of the embeddings, we would have to make huge computations. A better way is to divide the dimensions (embedding dim) among the heads.  Output of the multi-headed attention module must be concatenated. 
- Inside each head of the attention mechanism, each word vector has three representations a Query vector, key and value.

## Notes from e2ml
(https://e2eml.school/transformers.html#second_order_matrix_mult)
- De-embedding is done the same way embeddings are done, with a projection from one space to another, that is, a matrix multiplication.
- The softmax is helpful here for three reasons. First, it converts our de-embedding results vector from an arbitrary set of values to a probability distribution. Softmax exaggerates the difference between values. Preserves though if some words are equally likely. And we can perform backpropagation.
- Linear layers are used to project the matricces. To make multi-headed self-attention even possible.
- Skip connections add a copy of the input to the output of a set of calculations. The input to the attention block are added back in to its outputs. The inputs to the element-wise forward blcoks are added to the inputs. Makes the overall pipeline robust.
- Skip connections help to smooth out saddle points and ridges of the gradient. The problem is taht attention is a filter, that blocks out most what tries to pass through and may lead to large areas where the gradient is flat. Slopes of loss function hills are much smoother uniform, if skip connections are used. Also, it could happen, that an attention filter forgets entirely about the most recent word. Skip connections therefore enforce the signal and add the word back in.
- Inputs are shifted to have a zero mean and scaled to std dev of 1. It's needed cause it matters, for e. g., softmax what ranges values have and if they are balanced.
- layer normalization maintains a consistent distribution of signal values each step of the way throughout many-layered neural nets.
- Intuively, multiple attention layers allow for multiple paths to good set of transformer params. More layers lead to better results, but improvements are marginal with more than six layers.
- We can not make any judgements about the performance of the encoder, as the result is only a sequence of vectors in the embedded space.
- Cross-attention is similar to self-attention but with the exception taht the key matrix, value  matrix are from the fianl encoder layer
- As all layers get the same embedded source sequence in the decoder, it can be said that succesive layers provide redundancy and cooperate to perform the same task.

## Architecture
Here, the encoder maps an input sequence of symbol representations $\left(x_{1}, \ldots, x_{n}\right)$ to a sequence of continuous representations $\mathrm{z}=\left(z_{1}, \ldots, z_{n}\right)$. Given $\mathrm{z}$, the decoder then generates an output sequence $\left(y_{1}, \ldots, y_{m}\right)$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1 respectively.


Encoder: The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm $(x+$ Sublayer $(x))$, where Sublayer $(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text {model }}=512$.

Decoder: The decoder is also composed of a stack of $N=6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

Visualization of norm-first and norm last (similar in [[@xiongLayerNormalizationTransformer2020]]):
![[layer-norm-first-last.png]]
![[formulas-layer-norm.png]]
![[norm-first-norm-last-big-picture.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)

Layer norm / batch norm / instance norm:
![[layer-batch-instance-norm.png]]
![[viz-of image-embedding.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)


## Notes from Huggingface 🤗
https://huggingface.co/course/chapter1/4
-   **Encoder (left)**: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
-   **Decoder (right)**: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

## Notes from Baeldung 🍁
https://www.baeldung.com/cs/transformer-text-embeddings
- input sequence is input to the encoding block to obtain rich embeddings for each token, which is then fed to the decoding block to obtain the output
- initial layers in the encoder capture more basic patterns, latter blocks capture more sophisticated ones (as only filtered signal is passed?)
- Encoder takes one vector per token in the sequence as in put and returns a vector per token of the same shape. Thus, intuitively, the encoder returns the same input vectors, but enriched with more complex information.
- Self-attention layer detects related tokens in the sequence. (sequential)
- Next are the add and normalization layers, as the entire sequence is required and normalized. FFN follows (parallel) and another add and normalization. The only part that can be normalized are the feed-forward parts.
- decoder is like the encoder but with an additional encoder-decoder-attention layer. The attention mechanism provides insights into which tokens of the input sequence are more relevant to the current output token. It is followed by an add and normalize layer.
- the current decoder input will be processed producing and output, which will feed the next decoder
- The last decoder is connected to the output layer, generating the next output token, until the EOS token is reached.
- The decoder outputs a stack of float vectors. This vector is connect to a linear layer to project the output vector into a vector the size of the vocabulary. By applying softmax, we obtain the probabilities for every token to be the next token in the sequence. The token with the highest probability is chosen.

## Notes on Talk with Łukasz Kaiser 🎙️
(see here: https://www.youtube.com/watch?v=rBCqOTEfxvg)

- RNNs suffer from vanishing gradients
- Some people used CNNs, but path length is still logarithmic (going down a tree). Is limited to position.
- Attention: make a query with your vector and look at similar things in the past. Looks at everything, but choose things, that are similar.
- Encoder attention allows to go from one word to another. (Encoder Self-Attention)
- MaskedDecoder Self-Attention (is a single matrix multiply with a mask) to mask out all prev. elements not relevant
- Attention A(Q, K, V) (q = query vector) (K, V matrices= memory) (K = current word working on, V = all words generated before). You want to use q to find most similar k and get values that correspond to keys. (QK^T) gives a probability distribution over keys, which is then multiplied with values
- n^2 * d complexity
- to preserve the order of words they use multi-head attention
- attention heads can be interpreted (see winograd schemas)

## Notes from talk with Lucas Beyer / Google🎙️
(see https://www.youtube.com/watch?v=EixI6t5oif0)
- attention was originally introduced in the Bahdu paper. But was not the most central part.
- attention is like a (convoluted (soft)) dictionary lookup. like in a dict we have keys and values and want to query the dictionary. keys and values are a vector of quotes. measure the similarity with the dotproduct. we measure similarity between query and key (attention weights) and the result is normalized. We take weighted average of all values weighted by the attention weights. Note output can also be average over multiple 
![[attention-visualization.png]]
- q (word to translate), k, v (words in source language)
- We not just have one query, but multiple. Also an attention matrix. We use multi-head attention
- Multi-head attention splits the queries along the embedding dimension. Also outputs are split. Works empirically better. Requires less compute. (only implementation details. Not the gist of attention.)
- Architecture is heavily inspired by the translation task / community. This is helpful, as it resulted in encoder / decoder architecture.
- Every token from the input sequence is linearily projected. Each vector looks around to see what vectors are there and calculates the output. (self-attention)
- Every token individually is sent to a oint-wise MLP. It's done individually for every token. Stores knowledge. There is a paper. <mark style="background: #FFB8EBA6;">(search for it) </mark>"Gives the model processing power to think about what it has seen." Larger hidden size gives better results.
- skip processing. We have input and update it with our processing. (See residual stream in (mathematical foundations of Transformers))
- Layer norm is technically important.
- It's not clear, which variant of layer-norm is better.
- Decoder learns to sample from all possible outputs of the target language. 10 most likely translation etc. Computationally infeasible. To solve we look at one token at a time. Decoder works auto-regressively. Choose most likely token. and update all the inputs / things we have computed so far.
- All inputs are passed into the decoder at once to reduce training times. We multiply with mask matrix to lookup future tokens. In generation time we can not implement this trick and have to implement token by token.
- Cross-attention. Tokens from decoder become queries and keys and values come from the encoder. Look at the tokens from the source language (keys, values). 
- Flexible architecture. Needs loads of data. Are computationally efficient. That's true.

## Notes from Zhang
(see here: [[🧠Deep Learning Methods/@zhangDiveDeepLearning2021]])

- framework for designing attention mechanisms consists of:
    - volitional (~free) cues = queries
    - sensory inputs  = keys
    - nonvolitional cue of sensory input = keys
- attention pooling mechanism  enables a given query (volitional cue) to interact with keys (nonvolitional cues) which guides a biased selection over values (sensory inputs)
- self attention enjoys both parallel computation and the shortest maximum path length. Which makes it appealing to design deep architectures by using self-attention. Do not require a convolutional layer or recurrent layer.
- It's an instance of an encoder-decoder architecture. Input and output sequence embeddings are added with positional encoding before being fed into the encoder and the decoder that stack modules based on self-attention.
