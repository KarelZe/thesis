#transformer #embeddings #numerical #continous #categorical #tabular

In the chapter [[🤖Transformer]] we have shown that processing [[🛌Token Embedding]] and contextualizing them, is the core idea behind transformers. Yet, [[🛌Token Embedding]]s  are tailored towards data in text representation. In the context natural language processing, embeddings are a efficient way to represent tokens in a sequence. As this work is concerned about trade classification on tabular datasets containing both numerical and categorical features, the aforementioned concept is not directly applicable and must be enhanced to a generic *feature embedding* / <mark style="background: #ABF7F7A6;">generalize embedding vector</mark>. We do this separately for categorical and numerical features. 

Tabular data is highly flexible with regard to the columns, their data type, and their relationship to each other. While features maintain a shared meaning across rows, or samples, no shared meaning / <mark style="background: #ADCCFFA6;">(non-universal semantics?)</mark> can be assumed across columns. For instance, every sample in a trade data set may contain the previous trade price, yet the the meaning of the trade price is different from other columns, such as the exchange, urging the need for separate embeddings.

**Numerical data:** 🔢
Transformer-like architectures handle numerical features by mapping the scalar values to a high-dimensional embedding vector [[@gorishniyEmbeddingsNumericalFeatures2022]]. In the simplest case, a learned linear projection is used to obtain the embedding vector, which was previously employed in [[@kossenSelfAttentionDatapointsGoing2021]], [[@somepalliSAINTImprovedNeural2021]], or [[@gorishniyRevisitingDeepLearning2021]]. <mark style="background: #ADCCFFA6;">(Give formula) </mark>More sophisticated approaches like the *piece-wise linear encoding* or a *periodic encoding* [[@gorishniyEmbeddingsNumericalFeatures2022]]  elevate through non-linear mappings and can elevate the model's performance further. (...)

Alternatively, numerical features can be processed as a scalar in non-transformer-based networks, separate from the remaining feature. We explore this idea as part of our discussion on [[🤖TabTransformer]]. However, recent line of research, e. g.,  [[@gorishniyEmbeddingsNumericalFeatures2022]] and [[@somepalliSAINTImprovedNeural2021]], suggest that numerical embeddings can improve significantly performance of transformer-based models. [[@somepalliSAINTImprovedNeural2021]], report an increase *AUC* from 89.38 % to 91.72 % through embedding numerical features. Yet, their work offers no theoretical explanation / <mark style="background: #ADCCFFA6;">justification</mark>, why embeddings »work«. [[@grinsztajnWhyTreebasedModels2022]] (p. ) reason, identify the rotation invariance (...) are more. Also learned, numerical embeddings

**Categorical data:** 🗃️
Categorical data is different due to its nominal properties.

One common strategy is to apply one-hot-encoding, whereby each category is mapped to a sparse vector. While this approach is simple, it has been successfully used in conjunction with in several architectures including transformers.

The use of embeddings . For instance, a similarity between categories can  in the embedding space. 

For instance, applying a one-hot-encoding to the underlyings "GOOGL" (Alphabet Inc.), "MSFT" (Microsoft Inc.), and "K" (Kellogg Company) would result in sparse vectors equally similar <mark style="background: #ABF7F7A6;">/ distant appart</mark> from each other in terms of <mark style="background: #FFB86CA6;">cosine similarity</mark>. One would naturally expect a greater similarity between the first two underlyings due to similar operations. On the other hand, learned embedding can potentially capture such intrinsic properties of categorical variables, by arrange similar items closer in the embedding space. For high cardinal variables the learned embeddings also have the advantage of being memory efficient, as the length of the embedding vector untied from the cardinality of the variable.  


[[@hancockSurveyCategoricalData2020]] conclude, that embedding categoricals  matrix is unexplored and lacks a theoretical foundation.

The use of numerical and categorical feature embeddings is not exclusive to Transformers and has been explored in deep learning and classical machine learning. Noticable works include [[@chengWideDeepLearning2016]] .

Like in chapter, [[🛌Token Embedding]] the dimension of the embedding $e_{d}$ can affect the expressiveness of the network and is a tunable hyerparameter. One major drawback of learned embeddings is, that they contribute to parameter count of the model through the embedding matrix or increased layer capacity. As addressed by [[@wangLearningDeepTransformer2019]] (p. 2), the presented feature embeddings are restricted to the table and do not transfer across different tables.

To this end, embeddings non-exclusive to transformer-based architectures, and can be to other deep learning-based approaches, and even classical machine learning models, like [[🐈gradient-boosting]]. Covering those approaches is outside the scope of this work.



In the subsequent chapters we present for two concrete examples of transformers for the tabular data. We pick up concepts from this chapter. 


The idea of using learned embeddings for categorical variables is not particularily new.


1. Recall from chapter (...), that tabular data contains both categorical and continuous data
2. many sota approaches are transformer based are built around the idea of embedding
3. Repeat what an embedding is?
4. embeddings generally lead to a better performance.
5. embeddings for tabular architectures are different from token embeddings.
6. Why do we want to learn embeddings instead of just using scalars?
7. Different architectures implement embeddings to different degree. Present them afterwards.
8. One-hot-encoding of categoricals with high cardinality leads to sparse matrices, high number of parameters. Label encoding would lead to poor results. Situation is different for numerical data. Here embedding numerical data increases the parameter count.
9. Explain what is different for embedding continuous / categoricals?

## Notes
[[💤Embeddings for tabular data notes]]

