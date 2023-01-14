#word-2-vec #embeddings #data-preprocessing 

**Related:**
- See [[@gorishniyEmbeddingsNumericalFeatures2022]]
- See [[@phuongFormalAlgorithmsTransformers2022]]
- See [[@bengioNeuralProbabilisticLanguage]]
![[classical_transformer_architecture.png]]
(own drawing after [[@daiTransformerXLAttentiveLanguage2019]])

 ^dbc00b

[[@vaswaniAttentionAllYou2017]] train the Transformer on pre-trained token embeddings. To obtain token embeddings from raw input sequences, the sequence is first split into individual vocabulary elements, so-called *tokens*. Depending on the tokenization strategy, tokens can be as fine-grained as individual characters, or more coarse-grained sub-words (cite some famous papers), or words. The vocabulary $V$ consists of $N_{V}=|V|$ elements and maps tokens onto unique integer keys, referred to as token-ids. [[@phuongFormalAlgorithmsTransformers2022]]

The vocabulary may include special tokens, like the $\texttt{[UNK]}$ token to handle out-of-vocabulary items, the $\texttt{[EOS]}$ token to mark the end of sequence, or $\texttt{[CLS]}$ token for storing an aggregate representation of the sequence for classification (used in [[@devlinBERTPretrainingDeep2019]]; p. 4). 

Consider the input sequence "Kings and Queens"; a small vocabulary of $V=[0,N_v]$, and a mapping between token and token-id of $\text{queen}\mapsto 0$; $\text{king}\mapsto 1$ . Applying tokenizing by words, after common pre-processing like stemming and stopword removal, yields a sequence of token-ids $x$ given by $[1, 0]$. <mark style="background: #FFB8EBA6;">(What about EOS / BOS?; let indices start at one!)</mark>[^ 1] 

Subsequently, the sequence of token-ids is converted into a sequence of *token embeddings*. Pioneered by [[@bengioNeuralProbabilisticLanguage]] (p. 1,139), an embedding maps each token - here a word - into a high-dimensional space. Through representing every word as a vector, semantic and syntactic relationships between words can be encoded. As such, similar words share a similar embedding vector [[@bengioNeuralProbabilisticLanguage]] (p. 1,139). Also, word embeddings are semantically meaningful and can capture linguistic regularities, like the gender, with vector offsets [[@mikolovLinguisticRegularitiesContinuous2013]]  (p. 748 f.).

The embedding layer from Figure [[#^dbc00b]] is ultimately a lookup table to retrieve the embedding vector $e \in \mathbb{R}^{d_{\mathrm{e}}}$  from a learned, embedding matrix $W_e \in \mathbb{R}^{d_{\mathrm{e}} \times N_{\mathrm{V}}}$ with a token-id $v \in V \cong\left[N_{\mathrm{V}}\right]$ as shown :
$$
\tag{1}
e=W_e[:, v].
$$
The dimension of the embedding $d_e$ is an important tunable hyperparameter (see [[💡Hyperparameter tuning]]) .

Concluding the example from above with synthetic embeddings of dimensionality $e^d=3$:
$$
\tag{2}
\begin{aligned}
e_{\text{king}}&=W_e[:,1] = [0.01, 0.20, 0.134]^T\\
e_{\text{queen}}&=W_e[:,0] = [0.07, 0.157, 0.139]^T\\
\end{aligned}
$$
are likely to be close in space with cosine-similarity of $\approx 1$ due to their high semantic similarity with regard to profession. 
%%
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
x = np.array([0.01, 0.20, 0.134]).reshape(-1,1)
y = np.array([0.07, 0.157, 0.139]).reshape(-1,1)
cosine_sim = cosine_similarity(x, y)
print(cosine_sim)

[[1. 1. 1.] [1. 1. 1.] [1. 1. 1.]]

```
%%

Our running example uses word embeddings, motivated by the domain in which Transformers were proposed. However, the novel idea of capturing semantics as  embedding vectors extends to other discrete entities. We explore embedding categorical data, like the option's underlying in Chapter [[🤖TabTransformer]] and [[🤖FTTransformer]] and (discretized?), continuous data in the Chapter [[🤖FTTransformer]].

Embeddings can only encode the semantic relationship of tokens, but they do not provide a clue to the model about the relative and absolute ordering of tokens within a sequence. (Check if this is also true for contextual embeddings used in BERT etc.) The later later is vital in natural language processing and must be induced to the model using a [[🧵Positional encoding]], as later stages of the encoder are position-invariant (see [[@tunstallNaturalLanguageProcessing2022]] or [[@phuongFormalAlgorithmsTransformers2022]]) <mark style="background: #FFB8EBA6;">(check formulation could also be equivariant)</mark>.

[^1:]Note that there is a subtle difference between tokens and words. Token could be be words including punctation marks. But words can also be split into multiple tokens (Compare sub-words). Also notice the subtlety of words being reduced to their stem and lower-cased. (Provide the standard nlp reference for further info on this topic.)

---


## Notes from Phuong and Hutter
(see [[@phuongFormalAlgorithmsTransformers2022]])
![[token-embedding.png]]

## Notes from e2eml

In an embedding, those word points are all taken and rearranged (**projected**, in linear algebra terminology) into a lower-dimensional space. The picture above shows what they might look like in a 2-dimensional space for example. Now, instead of needing _N_ numbers to specify a word, we only need 2. These are the (_x_, _y_) coordinates of each point in the new space. Here's what a 2-dimensional embedding might look like for our toy example, together with the coordinates of a few of the words.

![](https://e2eml.school/images/transformers/embedded_words.png)

A good embedding groups words with similar meanings together. A model that works with an embedding learns patterns in the embedded space. That means that whatever it learns to do with one word automatically gets applied to all the words right next to it. This has the added benefit of reducing the amount of training data needed. Each example gives a little bit of learning that gets applied across a whole neighborhood of words

A good embedding groups words with similar meanings together. **A model that works with an embedding learns patterns in the embedded space.** That means that whatever it learns to do with one word automatically gets applied to all the words right next to it. This has the added benefit of reducing the amount of training data needed. Each example gives a little bit of learning that gets applied across a whole neighborhood of words.

## Notes from Chris Olah
From https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/:

I’d like to start by tracing a particularly interesting strand of deep learning research: word embeddings. In my personal opinion, word embeddings are one of the most exciting area of research in deep learning at the moment, although they were originally introduced by Bengio, _et al._ more than a decade ago.(see [[@bengioNeuralProbabilisticLanguage]]) Beyond that, I think they are one of the best places to gain intuition about why deep learning is so effective.


A word embedding $W:$ words $\rightarrow \mathbb{R}^n$ is a paramaterized function mapping words in some language to high-dimensional vectors (perhaps 200 to 500 dimensions). For example, we might find:
$$
\begin{aligned}
& W(\text { "cat" })=(0.2,-0.4,0.7, \ldots) \\
& W(\text { "mat" })=(0.0,0.6,-0.1, \ldots)
\end{aligned}
$$
(Typically, the function is a lookup table, parameterized by a matrix, $\theta$, with a row for each word: $\left.W_\theta\left(w_n\right)=\theta_{n-}\right)$
$W$ is initialized to have random vectors for each word. It learns to have meaningful vectors in order to perform some task.
WW is initialized to have random vectors for each word. It learns to have meaningful vectors in order to perform some task.

Word embeddings exhibit an even more remarkable property: analogies between words seem to be encoded in the difference vectors between words. For example, there seems to be a constant male-female difference vector:
$$
\begin{aligned}
& W(\text { ''woman" })-W(\text { ''man") }) \simeq W(\text { ''aunt" })-W(\text { ''uncle" }) \\
& W(\text { ''woman" })-W(\text { 'man" }) \simeq W(\text { 'queen" })-W(\text { ('king") }
\end{aligned}
$$
This may not seem too surprising. After all, gender pronouns mean that switching a word can make a sentence grammatically incorrect. You write, "she is the aunt" but "he is the uncle." 

Example is adapted from [[@mikolovLinguisticRegularitiesContinuous2013]]


## Notes from Vaswani
From [[@vaswaniAttentionAllYou2017]] :
“Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.” (Vaswani et al., 2017, p. 5)

## Notes from Rothman
From [[@rothmanTransformersNaturalLanguage2021]]:

The embedding sub-layer works like other standard *transduction models*. A tokenizer will transform a sentence into tokens.

“The Transformer contains a learned embedding sub-layer. Many embedding methods can be applied to the tokenized input. I chose the skip-gram architecture of the word2vec embedding approach Google made available in 2013 to illustrate the embedding sublayer of the Transformer.” (Rothman, 2021, p. 9)

“A skip-gram will focus on a center word in a window of words and predicts context words. For example, if word(i) is the center word in a two-step window, a skipgram model will analyze word(i-2), word(i-1), word(i+1), and word(i+2). Then the window will slide and repeat the process. A skip-gram model generally contains an input layer, weights, a hidden layer, and an output containing the word embeddings of the tokenized input words.” (Rothman, 2021, p. 9)

“To verify the word embedding produced for these two words, we can use cosine similarity to see if the word embeddings of the words black and brown are similar. Cosine similarity uses Euclidean (L2) norm to create vectors in a unit sphere. The dot product of the vectors we are comparing is the cosine between the points of those two vectors.” (Rothman, 2021, p. 10)

“The Transformer's subsequent layers do not start empty-handed. They have learned word embeddings that already provide information on how the words can be associated. However, a big chunk of information is missing because no additional vector or information indicates a word's position in a sequence.” (Rothman, 2021, p. 10)