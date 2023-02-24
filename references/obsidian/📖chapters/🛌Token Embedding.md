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
are likely to be close in space with cosine-similarity of $\approx 1$ due to their high semantic similarity. 

Embeddings can only encode the semantic relationship of tokens, but they do not provide a clue to the model about the relative and absolute ordering of tokens in which they appear in the sequence, since all stages of the encoder and decoder are invariant to the token's position (see [[@tunstallNaturalLanguageProcessing2022]] (p. 72) or [[@phuongFormalAlgorithmsTransformers2022]]). To preserve the ordering, positional information must be induced to the model using a [[🧵Positional Embedding]]. Another limitation of embeddings is, that identical tokens share their embedding, even if they are ambiguous and their meaning is different from the context in which they appear. To resolve this issue, embeddings get contextualized in the self-attention mechanism (see chapter [[🅰️Attention]]).

Our running example uses word embeddings, motivated by the domain in which transformers were proposed. However, the novel idea of capturing semantics as embedding vectors extends to other discrete entities, as we explore in chapter [[💤Embeddings For Tabular Data]].

---

[^1:]There is a subtle difference between tokens and words. A token can be words including punctuation marks. But words can also be split into multiple tokens, which are known as sub-words. To decrease the size of the vocabulary, words may be reduced to their stems, lower-cased, and stop words be removed. See <mark style="background: #FF5582A6;">(...)</mark> for in-depth coverage of pre-processing techniques.

[^2:] Throughout this work, we adhere to a notation suggested in [[@phuongFormalAlgorithmsTransformers2022]] (p. 1 f) to maintain consistency.


------
In the chapter [[🤖Transformer]] we have shown that processing token embeddings and contextualizing them, is the core idea behind Transformers. Yet, [[🛌Token Embedding]]s  are tailored towards textual data. With all tokens coming from the same vocabulary, a homogeneous embedding procedure suffices. As this work is concerned with trade classification on tabular datasets containing both numerical and categorical features, the aforementioned concept is not directly applicable and must be evolved to a generic feature embedding. We do this separately for categorical and numerical features.

Tabular data is flexible with regard to the columns, their data type, and their semantics. While features maintain a shared meaning across rows or samples, no universal semantics can be assumed across columns. For instance, every sample in a trade data set may contain the previous trade price, yet the meaning of the trade price is different from other columns, urging the need for heterogeneous embeddings. 

**Numerical embedding** 🔢
Columns may be categorical or numerical. Transformer-like architectures handle numerical features by mapping the scalar value to a high-dimensional embedding vector and process sequences thereof [[@gorishniyEmbeddingsNumericalFeatures2022]]. In the simplest case, a learned linear projection is utilized to obtain the embedding. Linear embeddings of numerical features were previously explored in [[@kossenSelfAttentionDatapointsGoing2021]], [[@somepalliSAINTImprovedNeural2021]], [[@chengWideDeepLearning2016]], or [[@gorishniyRevisitingDeepLearning2021]]. More sophisticated approaches rely on parametric embeddings, like the *piece-wise linear encoding* or the *periodic encoding* of [[@gorishniyEmbeddingsNumericalFeatures2022]]. Both enforce non-linear mapping. [[@gorishniyEmbeddingsNumericalFeatures2022]] show that these can alleviate the model's performance but at an additional computational cost. Alternatively, numerical features can be processed as a scalar in non-transformer-based networks and therefore independent from other features. We explore this idea as part of our discussion on [[🤖TabTransformer]]. 

Despite this simpler alternative, numerical embeddings are desirable, as a recent line of research, e. g.,  [[@gorishniyEmbeddingsNumericalFeatures2022]] and [[@somepalliSAINTImprovedNeural2021]] suggests, that numerical embedding can significantly improve performance and robustness to missing values or noise of Transformers. Exemplary, [[@somepalliSAINTImprovedNeural2021]] report an increase *AUC* (ROC) from 89.38 % to 91.72 % merely through embedding numerical features. Their work however offers no theoretical explanation. [[@grinsztajnWhyTreebasedModels2022]] (p. 8f.) fill this void. The authors find, that the mere use of embeddings breaks rotation invariance. *Rotational invariance* in the spirit of [[@ngFeatureSelectionVs2004]] refers to the model's dependency,  <mark style="background: #FFF3A3A6;">(...)</mark>.

**Categorical embeddings** 🗃️
Recall from the chapter [[🍪Selection Of Supervised Approaches]] that categorical data is data, that is divided into groups. In the context of trade classification, the option type is categorical and takes values $\{\text{'C'},\text{'P'}\}$ for calls and puts. Similar to a token, a category, e. g., $\text{'P'}$ in the previous example, must be represented as a multi-dimensional vector to be handled by the Transformer. Even when processed in other types of neural networks, categories need to be converted to real-valued inputs first, in order to optimize parameters with gradient descent.

A classical strategy is to apply one-hot-encoding to categorical features, whereby each category is mapped to a sparse vector, which can then be processed by a neural network. While this approach is conceptually simple and frequently employed in neural network architectures, it has several drawbacks like resulting in sparse vectors, where the cardinality of feature directly affects the one-hot vector. For instance, applying one-hot-encoding to the categorical underlyings $\texttt{GOOGL}$ (Alphabet Inc.), $\texttt{MSFT}$ (Microsoft Inc.), and $\texttt{K}$ (Kellogg Company) would result in sparse vectors equidistant in terms of cosine distance. Naturally, one would expect a greater similarity between the first two underlyings due to the overlapping field of operations. 

For Transformers learned, categorical embeddings are common, which are a direct adaption of the token embeddings ([[@wangTransTabLearningTransferable]], [[@gorishniyRevisitingDeepLearning2021]], [[@huangTabTransformerTabularData2020]], [[@somepalliSAINTImprovedNeural2021]]). A category is mapped to an embedding vector using a learned, embedding matrix, as in Equation [[🛌Token Embedding#^4bee48]]. These embeddings can potentially capture intrinsic properties of categorical variables by arranging similar items closer in the embedding space. For high cardinal variables, learned embeddings also have the advantage of being memory efficient, as the length of the embedding vector is untied from the cardinality of the variable [[@guoEntityEmbeddingsCategorical2016]] (p. 1). Despite these advantages, learned, categorical embeddings still lack a sound theoretical foundation and remain an open research problem [[@hancockSurveyCategoricalData2020]] (p. 28). In a similar vein, [[@borisovDeepNeuralNetworks2022]] (p. 2) note, that handling high-dimensional categoricals has not been resolved by existing approaches. Being dependent on a few samples, high cardinality is equally problematic for learned embeddings. We come back to this issue in later chapters.


Like in chapter [[🛌Token Embedding]] the dimension of the embedding $e_{d}$ affects the expressiveness of the network and is a tunable hyerparameter. One major drawback of learned embeddings is, that they contribute to the parameter count of the model through the embedding matrix or increased layer capacity of subsequent layers. 

To this end, embeddings are non-exclusive to Transformer-based architectures, and can be used in other deep learning-based approaches, and even classical machine learning models, like [[🐈Gradient Boosting]]. Covering these combinations is outside the scope of this work. We refer the reader to [[@gorishniyEmbeddingsNumericalFeatures2022]] for an in-depth comparison. Next, our focus is on two concrete examples of Transformers for tabular data.


**Notes:**
[[🛌 Token embeddings notes]]