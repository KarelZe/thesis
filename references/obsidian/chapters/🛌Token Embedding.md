As explained in the previous chapter, transformers operate on sequences of numeric vector representations, the *embeddings*, rather than on raw strings. More specifically, [[@vaswaniAttentionAllYou2017]] feed pre-trained token embeddings into the model. To obtain token embeddings from the raw input sequences, the sequence is first split into constituent vocabulary elements, the *tokens*. 

All known tokens are stored in a vocabulary. The vocabulary $V$ consists of $N_{V}=|V|$ elements and maps tokens onto their unique integer keys, referred to as *token-ids* [[@phuongFormalAlgorithmsTransformers2022]]. Apart from tokens in the training corpus, the vocabulary may include special tokens, like the $\texttt{[UNK]}$ token to handle out-of-vocabulary items, the $\texttt{[EOS]}$ token to mark the end of the sequence, or $\texttt{[CLS]}$ token for storing an aggregate representation of the sequence for classification (used in [[@devlinBERTPretrainingDeep2019]]; p. 4). 

Depending on the tokenization strategy, tokens can be as fine-grained as individual characters, or more coarse-grained sub-words ([[@bojanowskiEnrichingWordVectors2017]]) (p. 3), or words. For ease of explanation, we use tokens and words interchangeably.
Consider the following example with a small vocabulary of $V=[1,N_v]$, with a mapping between token and token-id of $\text{queen}\mapsto 1$; $\text{king}\mapsto 2$. For the sample sequence »Kings and Queens«, the sequence of token-ids $x$ would be by $[2, 1]$, after applying tokenizing by words and common pre-processing like lower-casing, and removal of the stop word »and« [^ 1].

Subsequently, the sequence of token ids is converted into a sequence of *token embeddings*. Pioneered by [[@bengioNeuralProbabilisticLanguage]] (p. 1,139), an embedding maps each word into a high-dimensional space. By representing every word as a vector, semantic and syntactic relationships between tokens can be encoded. As such, similar words share a similar embedding vector [[@bengioNeuralProbabilisticLanguage]] (p. 1,139). Also, word embeddings are semantically meaningful and can capture linguistic regularities, like gender through offsets between vectors [[@mikolovLinguisticRegularitiesContinuous2013]]  (p. 748 f.). 

The embedding layer from Figure [[#^dbc00b]] is ultimately a lookup table to retrieve the embedding vector $e \in \mathbb{R}^{d_{\mathrm{e}}}$  from a learned, embedding matrix $W_e \in \mathbb{R}^{d_{\mathrm{e}} \times N_{\mathrm{V}}}$ with a token-id $v \in V \cong\left[N_{\mathrm{V}}\right]$ as shown: [^2]
$$
\tag{1}
e=W_e[:, v].
$$

^4bee48
The weights of $W_e$ are initialized randomly and updated using gradient descent to obtain the learned embeddings. The dimension of the embedding $d_e$, affects the expressiveness of the network and is thus an important tuneable hyperparameter of the model (see [[💡Hyperparameter tuning]]). Concluding the example from above with artificial embeddings of dimensionality $e^d=3$:
$$
\tag{2}
\begin{aligned}
e_{\text{king}}&=W_e[:,2] = [0.01, 0.20, 0.134]^T\\
e_{\text{queen}}&=W_e[:,1] = [0.07, 0.157, 0.139]^T\\
\end{aligned}
$$
are likely to be close in space with cosine-similarity of $\approx 1$ due to their high semantic similarity. Embeddings can only encode the semantic relationship of tokens, but they do not provide a clue to the model about the relative and absolute ordering of tokens in which they appear in the sequence, since all stages of the encoder and decoder are invariant to the token's position (see [[@tunstallNaturalLanguageProcessing2022]] (p. 72) or [[@phuongFormalAlgorithmsTransformers2022]]). To preserve the ordering, positional information must be induced to the model using a [[🧵Positional Embedding]]. Another limitation of embeddings is, that identical tokens share their embedding, even if they are ambiguous and their meaning is different from the context in which they appear. To resolve this issue, embeddings get contextualized in the self-attention mechanism (see chapter [[🅰️Attention]]).

Our running example uses word embeddings, motivated by the domain in which transformers were proposed. However, the novel idea of capturing semantics as embedding vectors extends to other discrete entities, as we explore in chapter [[💤Embeddings for tabular data]].

---

[^1:]There is a subtle difference between tokens and words. A token can be words including punctuation marks. But words can also be split into multiple tokens, which are known as sub-words. To decrease the size of the vocabulary, words may be reduced to their stems, lower-cased, and stop words be removed. See <mark style="background: #FF5582A6;">(...)</mark> for in-depth coverage of pre-processing techniques.

[^2:] Throughout this work, we adhere to a notation suggested in [[@phuongFormalAlgorithmsTransformers2022]] (p. 1 f) to maintain consistency.

**Notes:**
[[🛌 Token embeddings notes]]