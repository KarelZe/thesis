
![[classical_transformer_architecture.png]]
(own drawing after [[@daiTransformerXLAttentiveLanguage2019]], <mark style="background: #FFB8EBA6;">use L instead of N, left encoder and right decoder. Add label.</mark>)

## Overview

The *Transformer* is a neural network architecture proposed by [[@vaswaniAttentionAllYou2017]] (p. 2 f.) for sequence-to-sequence modelling. Since it introduction it has become ubiquitous in natural language processing ([[@lampleLargeMemoryLayers2019]], p. 3; ...), among other domains (...). The wide success has lead to adaptions / seen wide adoptions for image representations, tabular representations, ... .

“The transformer network [44] is the current workhorse of Natural Language Processing (NLP): it is employed ubiquitously across a large variety of tasks. Transformers are built by stacking blocks composed of self-attention layers followed by fully connected layers (dubbed FFN), as shown in Figure 3.” (Lample et al., 2019, p. 3) ([[@lampleLargeMemoryLayers2019]])

The classical Transfomer follows an encoder-decoder architecture, as visualized in Figure(...).
<mark style="background: #FFB8EBA6;">FIXME:</mark>
-   **Encoder (left)**: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
-   **Decoder (right)**: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs

(Explain how encoder and decoder are intertwined. What is mask multi-head self-attention.)
(“Overall, the decoder is structured similarly to the encoder, with the following changes: First, the self-attention mechanisms are “causal” which prevents the decoder from looking at future items from the target sequence when it is fed in during training” (Narang et al., 2021, p. 15))

- the decoder processes autoregressive, meaning it considers previous outputs $y_i$, output before $y_i < y_j$.
- why were Transformers introduced?

## Transformer modes

(Transformer modes -> nicely explained in [[@tayEfficientTransformersSurvey2022]]. They refer to [[@raffelExploringLimitsTransfer2020]]).

For it's original application, machine translation, both the encoder and decoder are required, as the input sequence in the source language must first mapped to a rich numerical representation to later generate the output in the target language <mark style="background: #FFF3A3A6;">(Note: This is over-simplifying and lines are blurry.)</mark> Yet, the modular design, allows to adapt Transformers to a much wider range of use cases, some of which only require the encoder or decoder. The necessity is highly dependent on the task to solve i. e., if a enriched representation of the input suffices, or if inversely new output must be generated. We refer to these truncated architectures as *encoder-only* or *decoder-only*. 

## Transformer architectures
- motivation to switch discuss the effects of layer pre-normalization vs. post-normalization (see [[@tunstallNaturalLanguageProcessing2022]])

Both the encoder and decoder stack $L$ Transformer blocks. Each of these blocks consists of two sub-layers: a multi-head self-attention layer, followed by a fully-connected feed-forward network. Each of these sub-layer are connected by residual connections ([[@heDeepResidualLearning2015]]) and followed by layer normalization ([[@baLayerNormalization2016]]). The specific layer arrangement is referred to as *Post Layer Normalization* (Post-LN) derived from the placement of the normalization layer.

First, we apply layer normalization before the selfattention and feedforward blocks instead of after. This small change has been unanimously adopted by all current Transformer implementations because it leads to more effective training (Baevski and Auli, 2019; Xiong et al., 2020). [[@narangTransformerModificationsTransfer2021]]


Layer normalization improves the trainability of the Transformer by keeping.

![[layer-norm-first-last.png]]
Visualization of norm-first and norm last (similar in [[@xiongLayerNormalizationTransformer2020]]):

- Update residual stream, refine inputs from previous layers? See [[@elhage2021mathematical]]

(brittle training, requirement for warm-up stages)

“Different orders of the sub-layers, residual connection and layer normalization in a Transformer layer lead to variants of Transformer architectures. One of the original and most popularly used architecture for the Transformer and BERT (Vaswani et al., 2017; Devlin et al., 2018) follows “selfattention (FFN) sub-layer → residual connection → layer normalization”, which we call the Transformer with PostLayer normalization (Post-LN Transformer), as illustrated in Figure 1.” (Xiong et al., 2020, p. 3)

*Pre-LN* is known to be particullary hard
- Our analysis starts from the observation: the original Transformer (referred to as Post-LN) is less robust than its Pre-LN variant2 (Baevski and Auli, 2019; Xiong et al., 2019; Nguyen and Salazar, 2019). (from [[@liuUnderstandingDifficultyTraining2020]])
Addnorm operation.

How it's done in [[@tayEfficientTransformersSurvey2022]]:
The inputs and output of the multi-headed self-attention module are connected by residual connectors and a layer normalization layer. The output of the multi-headed selfattention module is then passed to a two-layered feed-forward network which has its inputs/outputs similarly connected in a residual fashion with layer normalization. The sublayer residual connectors with layer norm is expressed as:
$$
X=\operatorname{LayerNorm}\left(F_S(X)\right)+X
$$
where $F_S$ is the sub-layer module which is either the multi-headed self-attention or the position-wise feed-forward layers.



“Both the multi-head self-attention and the feed-forward layer are followed by an add-norm operation. This transformation is simply a residual connection [17] followed by layer normalization [23]. The layer normalization computes the average and standard deviation of the output activations of a given sublayer and normalizes them accordingly. This guarantees that the input yt of the following sublayer is well conditioned, i.e., that yT t 1 = 0 and yT t yt = √d.” (Sukhbaatar et al., 2019, p. 3)

The later, is commonly known as pre-norm.

![[formulas-layer-norm.png]]

[[@xiongLayerNormalizationTransformer2020]]
[[@nguyenTransformersTearsImproving2019]]
[[@wangLearningDeepTransformer2019]]

https://stats.stackexchange.com/a/565203/351242 ResNet paper ([[@heDeepResidualLearning2015]]) on residual learning / residual connections. Discusses in general the problems that arise with learning deep neural networks.

2.3 Putting it all together (from [[@tayEfficientTransformersSurvey2022]])
Each Transformer block can be expressed as:
$$
\begin{aligned}
& \left.X_A=\text { LayerNorm(MultiheadAttention }(X, X)\right)+X \\
& X_B=\operatorname{LayerNorm}\left(\operatorname{PositionFFN}\left(X_A\right)\right)+X_A
\end{aligned}
$$
where $X$ is the input of the Transformer block and $X_B$ is the output of the Transformer block. Note that the MultiheadAttention() function accepts two argument tensors, one for query and the other for key-values. If the first argument and second argument is the same input tensor, this is the MultiheadSelfAttention mechanism.


The classical Transformer of [[@vaswaniAttentionAllYou2017]] features 

- layer norm is the same as batch norm except that it normalizes the feature dimension ([[@zhangDiveDeepLearning2021]] p. 423)

As mentioned earlier, the Transformer architecture makes use of layer normalization and skip connections. The former normalizes each input in the batch to have zero mean and unity variance. Skip connections pass a tensor to the next layer of the model without processing and add it to the processed tensor. When it comes to placing the layer normalization in the encoder or decoder layers of a transformer, there are two main choices adopted in the literature: Post layer normalization This is the arrangement used in the Transformer paper; it places layer normalization in between the skip connections. This arrangement is tricky to train from scratch as the gradients can diverge. For this reason, you will often see a concept known as learning rate warm-up, where the learning rate is gradually increased from a small value to some maximum value during training. Pre layer normalization This is the most common arrangement found in the literature; it places layer normalization within the span of the skip connections. This tends to be much more stable during training, and it does not usually require any learning rate warm-up. The difference between the two arrangements is illustrated in Figure 3-6. (unknown)

“To train a Transformer however, one usually needs a carefully designed learning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyperparameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications.” (Xiong et al., 2020, p. 1)

Our analysis starts from the observation: the original Transformer (referred to as Post-LN) is less robust than its Pre-LN variant2 (Baevski and Auli, 2019; Xiong et al., 2019; Nguyen and Salazar, 2019). We recognize that gradient vanishing issue is not the direct reason causing such difference, since fixing this issue alone cannot stabilize PostLN training. It implies that, besides unbalanced gradients, there exist other factors influencing model training greatly [[@liuUnderstandingDifficultyTraining2020]]

leading to a brittle optimization. 
A variant known as pre-norm, 

Why do we employ residual connections? add input back in. Requires the 

Besides the decoder also contains a third sub-layer.

<mark style="background: #FFB86CA6;">“Residual connections (He et al., 2016a) were first introduced to facilitate the training of deep convolutional networks, where the output of the `-th layer F` is summed with its input: x`+1 = x` + F`(x`). (1) The identity term x` is crucial to greatly extending the depth of such networks (He et al., 2016b). If one were to scale x` by a scalar λ`, then the contribution of x` to the final layer FL is (∏L−1 i=` λi)x`. For deep networks with dozens or even hundreds of layers L, the term ∏L−1 i=` λi becomes very large if λi > 1 or very small if  for enough i. When backpropagating from the last layer L back to `, these multiplicative terms can cause exploding or vanishing gradients, respectively. Therefore they fix λi = 1, keeping the total residual path an identity map.” (Nguyen and Salazar, 2019, p. 2) [[@nguyenTransformersTearsImproving2019]]</mark>

<mark style="background: #ADCCFFA6;">“Inspired by He et al. (2016b), we apply LAYERNORM immediately before each sublayer (PRENORM): x`+1 = x` + F`(LAYERNORM(x`)). (3) This is cited as a stabilizer for Transformer training (Chen et al., 2018; Wang et al., 2019) and is already implemented in popular toolkits (Vaswani et al., 2018; Ott et al., 2019; Hieber et al., 2018), though not necessarily used by their default recipes. Wang et al. (2019) make a similar argument to motivate the success of PRENORM in training very deep Transformers. Note that one must append an additional normalization after both encoder and decoder so their outputs are appropriately scaled.” (Nguyen and Salazar, 2019, p. 2)</mark> [[@nguyenTransformersTearsImproving2019]]

<mark style="background: #FFB86CA6;">Skip Connection bypasses the gradient exploding or vanishing problem and tries to solve the model optimization problem from the perspective of information transfer. It enables the delivery and integration of information by adding an identity mapping from the input of the neural network to the output, which may ease the optimization and allow the error signal to pass through the non-linearities.</mark> (https://aclanthology.org/2020.coling-main.320.pdf)

<mark style="background: #FFF3A3A6;">The residual connection is crucial in the Transformer architecture for two reasons:

1.  Similar to ResNets, Transformers are designed to be very deep. Some models contain more than 24 blocks in the encoder. Hence, the residual connections are crucial for enabling a smooth gradient flow through the model.
    
2.  Without the residual connection, the information about the original sequence is lost. Remember that the Multi-Head Attention layer ignores the position of elements in a sequence, and can only learn it based on the input features. Removing the residual connections would mean that this information is lost after the first attention layer (after initialization), and with a randomly initialized query and key vector, the output vectors for position � has no relation to its original input. All outputs of the attention are likely to represent similar/same information, and there is no chance for the model to distinguish which information came from which input element. An alternative option to residual connection would be to fix at least one head to focus on its original input, but this is very inefficient and does not have the benefit of the improved gradient flow.</mark> (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

In the subsequent sections we introduce the classical Transformer of [[@vaswaniAttentionAllYou2017]] more thoroughly. Our focus on the central building blocks, attention, multi-head self-attention, and cross-attention (see Chapter [[🅰️Attention]]) as well as feed-forward networks (chapter [[🎱Position-wise FFN]]). In the subsequent chapters we show, that the self-attention mechanism and embeddings are generic enough to be transferred to the tabular domain. With the [[🤖TabTransformer]] ([[@huangTabTransformerTabularData2020]], p. 1 f.) and [[🤖FTTransformer]] ([[@gorishniyRevisitingDeepLearning2021]] p. 1) we introduce two promising alternatives. For consistency we adhere to a notation suggested in [[@phuongFormalAlgorithmsTransformers2022]] (p. 1 f) throughout the work.

## Resources
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- mathematical foundations [[@elhage2021mathematical]]
- general description of setup in [[@zhangDiveDeepLearning2021]]
- https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3
- http://nlp.seas.harvard.edu/2018/04/03/attention.html

Components in Embedding:
[[🛌Token Embedding]]
[[🧵Positional encoding]]

Components in Transformer block:
[[🅰️Attention]]
[[🎱Position-wise FFN]]

Specialized variants for tabular data:
[[🤖TabTransformer]]
[[🤖FTTransformer]]


## Viz



![[norm-first-norm-last-big-picture.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)

Layer norm / batch norm / instance norm:
![[layer-batch-instance-norm.png]]
![[viz-of image-embedding.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)

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
