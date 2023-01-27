#transformer #embeddings #numerical #continous #categorical #tabular

In the chapter [[🤖Transformer]] we have shown that processing [[🛌Token Embedding]]s and contextualizing them, is the core idea behind transformers. Yet, [[🛌Token Embedding]]s  are tailored towards data in text representation. With all tokens coming from the same vocabulary, a homogeneous embedding procedure suffices. As this work is concerned about trade classification on tabular datasets containing both numerical and categorical features, the aforementioned concept is not directly applicable and must be evolved to a generic feature embedding. We do this separately for categorical and numerical features.

Tabular data is flexible with regard to the columns, their data type, their distribution, and their semantics. While features maintain a shared meaning across rows, or samples, no universal semantics can be assumed across columns. For instance, every sample in a trade data set may contain the previous trade price, yet the the meaning of the trade price is different from other columns, urging the need for heterogeneous embeddings. 

**Embeddings of numerical data:** 🔢
Columns may be categorical or numerical. Transformer-like architectures handle numerical features by mapping the scalar value to a high-dimensional embedding vector [[@gorishniyEmbeddingsNumericalFeatures2022]]. In the simplest case, a learned linear projection is utilized to obtain the embedding. Linear embeddings of numerical features were previously explored in [[@kossenSelfAttentionDatapointsGoing2021]], [[@somepalliSAINTImprovedNeural2021]], [[@chengWideDeepLearning2016]], or [[@gorishniyRevisitingDeepLearning2021]]. More sophisticated approaches rely on parametric embeddings, like the *piece-wise linear encoding* or the *periodic encoding* of [[@gorishniyEmbeddingsNumericalFeatures2022]]. Both enforce a non-linear mapping. [[@gorishniyEmbeddingsNumericalFeatures2022]] show that these can alleviate model's performance, but at an additional computational cost. Alternatively, numerical features can be processed as a scalar in non-transformer-based networks and therefore independent from other features. We explore this idea as part of our discussion on [[🤖TabTransformer]]. Despite this simpler alternative, numerical embeddings are desirable, as a recent line of research, e. g.,  [[@gorishniyEmbeddingsNumericalFeatures2022]] and [[@somepalliSAINTImprovedNeural2021]] suggests, that numerical embedding can significantly performance and robustness to missing values or noise of transformers. Exemplary, [[@somepalliSAINTImprovedNeural2021]] report an increase *AUC* (ROC) from 89.38 % to 91.72 % through embedding numerical features alone. Their work offers no theoretical explanation. [[@grinsztajnWhyTreebasedModels2022]] (p. ) fill this gap.  <mark style="background: #ABF7F7A6;">identify the rotational variance (...) are more.</mark>  

**Embeddings of categorical data:** 🗃️
Unless columns posses an identical relationship, they must be embedded separately. 
<mark style="background: #FFF3A3A6;">(Encoding the data from [[@somepalliSAINTImprovedNeural2021]] explains nicely why every feature requires its own embedding.)
</mark>
Embeddings may be ordered 

Embedding categorical data requires a different treatment from numerical features, due to the (...?). (Nominal and ordinal)
Categorical data is different due to its nominal properties.
Features may share categories, but with a different 

One common strategy is to apply one-hot-encoding, whereby each category is mapped to a sparse vector, which can then be processed in a Transformer block or multi-layer perceptrons. While this approach is conceptually simple and frequently used in with neural network architectures or classical embeddings (See [[🐈extensions-to-gradient-boosting]] chapter), it has several major drawbacks like resulting in sparse vectors, where the cardinality of feature directly affects the one-hot vector. <mark style="background: #FFB8EBA6;">(remove this paragraph?)</mark>

A more efficient way is to use learned embeddings, whereby categorical  For instance, applying a one-hot-encoding to the underlyings "GOOGL" (Alphabet Inc.), "MSFT" (Microsoft Inc.), and "K" (Kellogg Company) would result in sparse vectors equally similar <mark style="background: #ABF7F7A6;">/ distant appart</mark> from each other in terms of <mark style="background: #FFB86CA6;">cosine similarity</mark>. One would naturally expect a greater similarity between the first two underlyings due to similar operations. On the other hand, learned embedding can potentially capture such intrinsic properties of categorical variables, by arrange similar items closer in the embedding space. For high cardinal variables the learned embeddings also have the advantage of being memory efficient, as the length of the embedding vector untied from the cardinality of the variable.  
[[@hancockSurveyCategoricalData2020]] conclude, that embedding categoricals  matrix is unexplored and lacks a theoretical foundation.

Like in chapter, [[🛌Token Embedding]] the dimension of the embedding $e_{d}$ can affect the expressiveness of the network and is a tunable hyerparameter. One major drawback of learned embeddings is, that they contribute to parameter count of the model through the embedding matrix or increased layer capacity. As brought up by [[@wangLearningDeepTransformer2019]] (p. 2), the presented feature embeddings are restricted to the table and do not transfer across different tables.

**Positional embeddings in tabular data:**
In chapter [[🧵Positional Embedding]] we reconed that using solely the token embeddings, would lose the ordering of the sequence and applied a positional encoding to resolve this issue. In tabular dataset columns may be arranged in an arbitrary order, where the (Gesamtheit matters?)  This is necessary as all  

To this end, embeddings are non-exclusive to transformer-based architectures, and can be used in other deep learning-based approaches, and even classical machine learning models, like [[🐈gradient-boosting]]. Covering these combinations is outside the scope of this work. We refer the reader to [[@gorishniyEmbeddingsNumericalFeatures2022]] for an in-depth comparison. Our focus is on two concrete examples of transformers for tabular data. We pick up concepts from this chapter. 

## Notes
[[💤Embeddings for tabular data notes]]

