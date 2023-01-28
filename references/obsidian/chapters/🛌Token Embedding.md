#word-2-vec #embeddings #data-preprocessing 

Instead of using the NLP-specific term "token embedding" use term "feature embedding." E. g., [[@somepalliSAINTImprovedNeural2021]] write <mark style="background: #FFB86CA6;">SAINT is inspired by the transformer encoder of Vaswani et al. [41], designed for natural language, where the model takes in a sequence of feature embeddings and outputs contextual representations of the same dimension.</mark>

(Define with a catchy phrase what an embedding is -> Example from Marais -> use similar “An embedding is a layer which maps a discrete input to a numeric vector representation. It was first used in NLP in order to represent words as numbers so that they may be processed by numeric models.” (Marais, p. 51))

[[@vaswaniAttentionAllYou2017]] train the Transformer on pre-trained token embeddings. To obtain token embeddings from raw input sequences, the sequence is first split into individual vocabulary elements, so-called *tokens*. Depending on the tokenization strategy, tokens can be as fine-grained as individual characters, or more coarse-grained sub-words (cite some famous papers), or words. The vocabulary $V$ consists of $N_{V}=|V|$ elements and maps tokens onto unique integer keys, referred to as token-ids. [[@phuongFormalAlgorithmsTransformers2022]]

The vocabulary may include special tokens, like the $\texttt{[UNK]}$ token to handle out-of-vocabulary items, the $\texttt{[EOS]}$ token to mark the end of sequence, or $\texttt{[CLS]}$ token for storing an aggregate representation of the sequence for classification (used in [[@devlinBERTPretrainingDeep2019]]; p. 4). 

Consider the input sequence "Kings and Queens"; a small vocabulary of $V=[0,N_v]$, and a mapping between token and token-id of $\text{queen}\mapsto 0$; $\text{king}\mapsto 1$ . Applying tokenizing by words, after common pre-processing like stemming and stopword removal, yields a sequence of token-ids $x$ given by $[1, 0]$. <mark style="background: #FFB8EBA6;">(What about EOS / BOS?; let indices start at one!)</mark>[^ 1] 

Subsequently, the sequence of token-ids is converted into a sequence of *token embeddings*. Pioneered by [[@bengioNeuralProbabilisticLanguage]] (p. 1,139), an embedding maps each token - here a word - into a high-dimensional space. Through representing every word as a vector, semantic and syntactic relationships between words can be encoded. As such, similar words share a similar embedding vector [[@bengioNeuralProbabilisticLanguage]] (p. 1,139). Also, word embeddings are semantically meaningful and can capture linguistic regularities, like the gender, with vector offsets [[@mikolovLinguisticRegularitiesContinuous2013]]  (p. 748 f.).

The embedding layer from Figure [[#^dbc00b]] is ultimately a lookup table to retrieve the embedding vector $e \in \mathbb{R}^{d_{\mathrm{e}}}$  from a learned, embedding matrix $W_e \in \mathbb{R}^{d_{\mathrm{e}} \times N_{\mathrm{V}}}$ with a token-id $v \in V \cong\left[N_{\mathrm{V}}\right]$ as shown :
$$
\tag{1}
e=W_e[:, v].
$$

^4bee48

<mark style="background: #FFB8EBA6;">(What do we mean by learnable-> randomly initialize, let the model figure out how to adjust embedding)</mark> The dimension of the embedding $d_e$ is an important tunable hyperparameter (see [[💡Hyperparameter tuning]]) .

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
TODO: Use dot-product instead, to be coherent to the idea used in attention? Not scaled by magnitude?
See here. https://datascience.stackexchange.com/questions/744/cosine-similarity-versus-dot-product-as-distance-metrics
%%

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

Embeddings can only encode the semantic relationship of tokens, but they do not provide a clue to the model about the relative and absolute ordering of tokens within a sequence. (Check if this is also true for contextual embeddings used in BERT etc.) The later later is vital in natural language processing and must be induced to the model using a [[🧵Positional Embedding]], as later stages of the encoder are position-invariant (see [[@tunstallNaturalLanguageProcessing2022]] or [[@phuongFormalAlgorithmsTransformers2022]]) <mark style="background: #FFB8EBA6;">(check formulation could also be equivariant)</mark>.

[^1:]Note that there is a subtle difference between tokens and words. Token could be be words including punctation marks. But words can also be split into multiple tokens (Compare sub-words). Also notice the subtlety of words being reduced to their stem and lower-cased. (Provide the standard nlp reference for further info on this topic.)

**Notes:**
[[🛌 Token embeddings notes]]