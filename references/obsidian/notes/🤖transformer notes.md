
## Notes from Sukhabaatar
(see [[@sukhbaatarAugmentingSelfattentionPersistent2019]])
Feedforward sublayer. The second element of a transformer layer is a fully connected feedforward layer. This sublayer is applied to each position $t$ in the input sequence independently, and consists of two affine transformations with a pointwise non-linear function in between:
$$
\mathrm{FF}\left(\mathbf{x}_t\right)=\mathbf{U} \sigma\left(\mathbf{V} \mathbf{x}_t+\mathbf{b}\right)+\mathbf{c},
$$
where $\sigma(x)=\max (0, x)$ is the ReLU activation function; $\mathbf{V}$ and $\mathbf{U}$ are matrices of dimension $d \times d_f$ and $d_f \times d$ respectively; $\mathbf{b}$ and $\mathbf{c}$ are the bias terms. Typically, $d_f$ is set to be 4 times larger than $d$.
Add-norm. Both the multi-head self-attention and the feed-forward layer are followed by an add-norm operation. This transformation is simply a residual connection [17] followed by layer normalization [23]. The layer normalization computes the average and standard deviation of the output activations of a given sublayer and normalizes them accordingly. This guarantees that the input $\mathbf{y}_t$ of the following sublayer is well conditioned, i.e., that $\mathbf{y}_t^T 1=0$ and $\mathbf{y}_t^T \mathbf{y}_t=\sqrt{d}$. More precisely, the AddNorm operation is defined as:
$$
\operatorname{AddNorm}\left(\mathbf{x}_t\right)=\operatorname{LayerNorm}\left(\mathbf{x}_t+\operatorname{Sublayer}\left(\mathbf{x}_t\right)\right) \text {, }
$$
where Sublayer is either a multi-head self-attention or a feedforward sublayer.


## Notes from Tay
(see [[@tayEfficientTransformersSurvey2022]])
- transformers are a multi-layered architecture formed by stacking transformer blocks on top of one another.
- Transformer blocks are characterized by a multi-head sel-attention mechanism, a poistion-wise feed-forward network, layer norm modules ([[@baLayerNormalization2016]]) and residual connectors ([[@heDeepResidualLearning2015]])
- The input is passed through an embedding layer and converts one-hot tokens into a $d_{\text{model}}$ dimensional embedding. The tensor is composed with a positional encoding and passed through a multi-headed self-attention module. 
- Inputs and outputs oft he multi-headed self-attention module are connected by residual connectors and a layer normalization layer. the output of the multi-headed self-attention module is then passed to a two-layered feed forward network which has it inputs / outputs similarily connected in a residual fashion with layer normalization. 

## Notes from Harvard🎓
http://nlp.seas.harvard.edu/2018/04/03/attention.html
- self-attention is a an attention mechaism of realting different positions of a single sequence to compute a representation of the sequence.
- Encoder maps an input sequenc of symbol representations (x1, .... xn) to a a seuqence of continuous representations z=(z1, ..., zn). The decoder generates from z an output sequence(y1,...ym) of symbols one element at a time. Each step is auto-regressive as generated symbols are used as additional input when generating text.
- Dropout is applied to the output of each sublayer, before it is added to the sub-layer input and normalized. Requires all sublayers of the model to have a fixed dimensions to make residual connnections (which add elemnt-wisely?) possible.
- Each layer has two sub-layers. The first is a multi-headed self attention mechanism and the second is a simple, point-wise feed-forward network.
- the decoder inserts a third layer, which performs multi-head attention over the output of the encoder stack. Again layer normalization and residual connections are used.
- To prevent the decoder from attending to subsequent  poistions a musk is used, so that the predictions only depend on known outputs at positions less than i.

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

## Notes from ML6
(see: https://blog.ml6.eu/transformers-for-tabular-data-hot-or-not-e3000df3ed46)
- Why it makes sense to use embeddings / transformers for tabular data:
	- In a lot of tabular “languages”, there are meaningful feature interactions. The value of one feature impacts the way another feature should be interpreted.Decision trees naturally lend themselves to model these kinds of interactions because of their sequential decision making process. A decision deeper in the tree depends on all previous decisions since the root, so previous feature values impact the current feature interpretation. That’s why a transformer also explicitly models token interactions through its multi head self-attention mechanism. In that way, the model produces _contextual embeddings_.
	- use powerful semi-supervised training techniques from natural language processing
	- A final advantage of transformers is that they excel in handling missing and noisy features.

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

^54aa8a
(see https://www.youtube.com/watch?v=EixI6t5oif0)
- attention was originally introduced in the Bahdu paper. But was not the most central part.
- attention is like a (convoluted (soft)) dictionary lookup. like in a dict we have keys and values and want to query the dictionary. keys and values are a vector of quotes. measure the similarity with the dotproduct. we measure similarity between query and key (attention weights) and the result is normalized. We take weighted average of all values weighted by the attention weights. Note output can also be average over multiple 
![[attention-visualization.png]]
- q (word to translate), k, v (words in source language)
- We not just have one query, but multiple. Also an attention matrix. We use multi-head attention
- Multi-head attention splits the queries along the embedding dimension. Also outputs are split. Works empirically better. Requires less compute. (only implementation details. Not the gist of attention.)
- Architecture is heavily inspired by the translation task / community. This is helpful, as it resulted in encoder / decoder architecture.
- Every token from the input sequence is linearily projected. Each vector looks around to see what vectors are there and calculates the output. (self-attention)
- Every token individually is sent to a oint-wise MLP. It's done individually for every token. Stores knowledge. There is a paper. (references in [[@gevaTransformerFeedForwardLayers2021]] are the best I could find?) Gives the model processing power to think about what it has seen." Larger hidden size gives better results.
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