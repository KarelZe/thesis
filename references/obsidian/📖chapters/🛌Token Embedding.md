As explained previously, Transformers operate on sequences of numeric vector representations, the *token embeddings*, rather than on raw strings.  The classical Transformer was trained on *word embeddings* for the purpose of translation. Nevertheless, token embeddings are generic and arbitrary inputs can be embedded and processed by the Transformer.  In the spirit of ([[@vaswaniAttentionAllYou2017]]5), we first explore word embeddings for textual data, before adapting embeddings to the tabular domain.

**Word embeddings**
To obtain token embeddings from the raw input sequences i. e., a sentence, the sequence is first split into constituent vocabulary elements, the *tokens*. All known tokens are stored in a vocabulary. The vocabulary $V$ consists of $N_{V}=|V|$ elements and maps tokens onto their unique integer keys, referred to as *token-ids* ([[@phuongFormalAlgorithmsTransformers2022]]3). Apart from tokens in the training corpus, the vocabulary may include special tokens, like the $\texttt{[UNK]}$ token to handle out-of-vocabulary items or $\texttt{[CLS]}$ token for storing an aggregate representation of the sequence for classification (cp.[[@devlinBERTPretrainingDeep2019]]4). The result is a sequence of token ids.

Depending on the tokenization strategy, tokens can be as fine-grained as individual characters, more coarse-grained sub-words ([[@bojanowskiEnrichingWordVectors2017]]3), or words. For ease of explanation, we treat tokens as words. Consider the following example with a small vocabulary of $V=[1,N_v]$, with a mapping between token and token-id of $\text{queen}\mapsto 1$; $\text{king}\mapsto 2$. For the sample sequence »Kings and Queens«, the sequence of token-ids $x$ would be $[2, 1]$, after applying tokenizing by words and common pre-processing like lower-casing, and the removal of the stop word »and« [^ 1].  

The conversion to token ids, however, loses the semantics, as token ids may be assigned arbitrarily or ordering by semantics not be feasible. This limitation, can be overcome by embeddings, asioneered by ([[@bengioNeuralProbabilisticLanguage]]1139), which map each tokenid into a high-dimensional space. By representing words as a vector, semantic and syntactic relationships between tokens can be encoded. As such, similar words share a similar embedding vector ([[@bengioNeuralProbabilisticLanguage]]1139). Also, word embeddings are semantically meaningful and can capture linguistic regularities, like gender through offsets between vectors ([[@mikolovLinguisticRegularitiesContinuous2013]]748--749). 

The embedding layer from Figure [[#^dbc00b]] is ultimately a lookup table to retrieve the embedding vector $e \in \mathbb{R}^{d_{\mathrm{e}}}$  from a learnt, embedding matrix $W_e \in \mathbb{R}^{d_{\mathrm{e}} \times N_{\mathrm{V}}}$ with a token-id $v \in V \cong\left[N_{\mathrm{V}}\right]$ as shown: [^2]
$$
\tag{1}
e=W_e[:, v].
$$

^4bee48
The weights of $W_e$ are initialised randomly and updated using gradient descent to obtain the learnt embeddings. The dimension of the embedding $d_e$, affects the expressiveness of the network and is thus an important tuneable hyperparameter of the model. 

Concluding the example from above with artificial embeddings of dimensionality $d_{e}=3$:
$$
\tag{2}
\begin{aligned}
e_{\text{king}}&=W_e[:,2] = [0.01, 0.20, 0.134]^T\\
e_{\text{queen}}&=W_e[:,1] = [0.07, 0.157, 0.139]^T\\
\end{aligned}
$$
are likely to be close in space with cosine-similarity of $\approx 1$ due to their high semantic similarity. 

As this work is concerned with trade classification on tabular datasets, the aforementioned concepts must be evolved. Differently from textual data, where all tokens come from the same vocabulary and a homogeneous embedding procedure suffices, tabular data is flexible with regard to the columns, their data type, and their semantics. While features maintain a shared meaning across rows, no universal semantics can be assumed across columns. For instance, every sample in a trade data set may contain the previous trade price, yet the meaning of the trade price is different from other columns, urging the need for heterogeneous embeddings. Also, columns may be categorical or continuous.

**Continuous embedding** 🔢
Transformer networks can handle continuous features, such as the trade price, by mapping the scalar value to a high-dimensional embedding vector and process sequences thereof ([[@gorishniyEmbeddingsNumericalFeatures2022]]). In the simplest case, a learnt linear projection is utilised to obtain the embedding. Linear embeddings of numerical features were previously explored in ([[@kossenSelfAttentionDatapointsGoing2021]]1), ([[@somepalliSaintImprovedNeural2021]]1), or ([[@gorishniyRevisitingDeepLearning2021]]1).  

More sophisticated approaches rely on parametric embeddings, like the *piece-wise linear encoding* or the *periodic encoding* of ([[@gorishniyEmbeddingsNumericalFeatures2022]]10). Both enforce a non-linearity. The authors show that these can alleviate the model's performance but at a considerable increase of computational cost. 

More generally, the works of ([[@gorishniyEmbeddingsNumericalFeatures2022]]1) and ([[@somepalliSaintImprovedNeural2021]]1) suggest, that numerical embedding can significantly improve performance and robustness to missing values or noise. Their work however provide no theoretical explanation. ([[@grinsztajnWhyTreebasedModels2022]]8--9) fill this void and attribute the increased robustness to the broken rotational invariance. 

**Categorical embeddings** 🗃️

Datasets often comprise not only continuous features, but also categorical features such as the underlying. In the context of tabular Transformers, learnt categorical embeddings are widely used, which are similar to word embeddings ([[@wangTransTabLearningTransferable]]1), ([[@gorishniyRevisitingDeepLearning2021]]1), ([[@huangTabTransformerTabularData2020]]1), [[@somepalliSaintImprovedNeural2021]]1). Similar to the word embedding, each category is mapped to an embedding vector using a learnt, embedding matrix. Due to the heterogeneous nature of tabular data, embeddings may not be shared between features.

These categorical embeddings can potentially capture intrinsic properties of categorical variables by arranging similar items closer in the embedding space.  For instance, consider the underlyings $\mathtt{GOOGL}$ (Alphabet Inc.), $\mathtt{MSFT}$ (Microsoft Inc.), and $\mathtt{K}$ (Kellogg Company). Due to the overlapping field of operations, one would anticipate greater similarity between Alphabet and Google.

Despite these advantages, high-cardinal features present a challenge for embeddings since they are typically learnt from few samples, which can lead to overfitting. Handling high-dimensional categorical data remains an open research problem, as noted by ([[@borisovDeepNeuralNetworks2022]]2).

One limitation of all learnt embeddings is, that they contribute to the parameter count of the model through the embedding matrix or increased layer capacity of subsequent layers. The dimension of the embedding $d_{e}$ affects the expressiveness and complexity of the network and is an important hyerparameter. 

Embeddings can only encode the semantic relationship of tokens, but they do not provide a clue to the model about the relative and absolute ordering of tokens in which they appear in the sequence, since all stages of the encoder and decoder are invariant to the token's position (see [[@tunstallNaturalLanguageProcessing2022]] (p. 72) or [[@phuongFormalAlgorithmsTransformers2022]]). To preserve the ordering, positional information must be induced to the model using a [[🧵Positional Embedding]]. Another limitation of embeddings is, that identical tokens share their embedding, even if they are ambiguous and their meaning is different from the context in which they appear. To resolve this issue, embeddings get contextualised in the self-attention mechanism (see chapter [[🅰️Attention]]).


[^1:]There is a subtle difference between tokens and words. A token can be words including punctuation marks. But words can also be split into multiple tokens, which are known as sub-words. To decrease the size of the vocabulary, words may be reduced to their stems, lower-cased, and stop words be removed. See <mark style="background: #FF5582A6;">(...)</mark> for in-depth coverage of pre-processing techniques.

[^2:] Throughout this work, we adhere to a notation suggested in [[@phuongFormalAlgorithmsTransformers2022]] (p. 1 f) to maintain consistency.

**Notes:**
[[🛌 Token embeddings notes]]