![[ft_transformer.png]]
(own drawing somewhat inspired by figure [[@gorishniyRevisitingDeepLearning2021]] and page 5; introduce CLS token? to differentiate from previous architectures?, not N but L) ^6a12a6

The FTTransformer of [[@gorishniyRevisitingDeepLearning2021]] is an adaption of the classical Transformer ([[@vaswaniAttentionAllYou2017]]) and BERT ([[@devlinBERTPretrainingDeep2019]]) for the tabular domain. Opposed to the [[TabTransformer]], the FTTransformer, introduces a  *feature tokenizer*, for embedding both continuous and categorical inputs, that are contextualized in a stack of Transformer layers, as shown in Figure [[#^6a12a6]].  In contrast to TabTransformer, where all contextualized embeddings contribute to the prediction, the output probabilities are only derived from the final hidden state of a single, specialized token, the $\texttt{[CLS]}$ token. We explain both aspects in greater detail in the next paragraphs. <mark style="background: #FF5582A6;">TODO: (Encoder-only)</mark>

The *feature tokenizer* transforms all features of $x$ to their embeddings. If the $j$-th feature, $x_j$, is **numerical**, it is projected to its embedding $e_j \in \mathbb{R}^{e_d}$ by element-wise multiplication with a learned vector $W_j \in \mathbb{R}^{e_d}$ and the addition of a feature-dependent bias term $b_j \in \mathbb{R}$, as in Equation (1). (bias should be a vector as per tehir NIPS talk, shifting is needed for good performance https://slideslive.com/38968794/revisiting-deep-learning-models-for-tabular-data?ref=recommended)

For **categorical** inputs, the embedding is implemented as a lookup table, similar to the techniques from Chapter [[🛌Token Embedding]] and [[🤖TabTransformer]]. We denote the cardinality of the $j$-th feature with $N_{C_j}$. The specific embeddings $e_j$ are queried with a unique integer key $c_j \in C_j \cong\left[N_{\mathrm{C_j}}\right]$ from the learned embedding matrix $W_j \in \mathbb{R}^{e_d \times N_{C_j}}$. Finally a feature-specific bias term $b_j$ is added <mark style="background: #FFB86CA6;">(TODO: lookup if bias is a scalar or vector?).</mark>  Overall for $x_j$:
%%
Exemplary, the encoding of the option type could be  $\text{P}\mapsto 1$; $\text{C}\mapsto 2$, which would result in a selection of the second column of the embedding matrix whenever a put is traded. 
%%
$$
\tag{1}
e_j= 
\begin{cases}
    W_j x_j +b_j, & \text{if } x_j \text{ is numeric}\\
    W_j[:c_j] +b_j,              & x_j \text{ is categorical}.
\end{cases}

$$
%%
$e_j = W_j x_j +b_j$ (if $x_j$ is numerical with $W_j \in \mathbb{R}^{e_d}$ (element-wise multiplication))
$e_j = W_j[:c_j] +b_j$ (if $x_j$ is numerical with $W_j \in \mathbb{R}^d$ ( $W_j \in \mathbb{R}^{e_d \times c_j}$ look up table. The size is dependent on the embedding dim and the cardinality of the categorical feature.  multiplication with one-hot-encoded vector; or just access the columns as done in chapter [[🛌Token Embedding]] ); $S_j$ is not introduced in paper. 

For original notation see [[@gorishniyRevisitingDeepLearning2021]].
%%

![[mashup-tokenizer-fttransformer.png]]
(by [[@devlinBERTPretrainingDeep2019]] (p. 4185) and [[@gorishniyRevisitingDeepLearning2021]] (p. 4))

![[viz-of image-embedding.png]]
(from https://github.com/dvgodoy/PyTorchStepByStep)
![[vison-transformer.png]]
(from [[@dosovitskiyImageWorth16x162021]])

For classification tasks in the language representation model BERT, [[@devlinBERTPretrainingDeep2019]] (p. 4,174) propose to append a specialized $\texttt{[CLS]}$ token to every sequence, that stores its aggregate representation. Like any other token, the $\texttt{[CLS]}$ token is embedded first (see chapter [[🛌Token Embedding]]), and contextualized in the Transformer layers. Its final representation is then used in the classification task. 

<mark style="background: #FFB86CA6;">“Similar to BERT’s [class] token, we prepend a learnable embedding to the sequence of embedded patches (z00 = xclass), whose state at the output of the Transformer encoder (z0 L) serves as the image representation y (Eq. 4).” (Dosovitskiy et al., 2021, p. 3)</mark> <mark style="background: #FFB8EBA6;">-> prepend is the word I was looking for</mark>

[[@gorishniyRevisitingDeepLearning2021]] (p. 4) adapt the idea of a $\texttt{[CLS]}$ token for tabular representation models. Similar to a categorical or continuous feature, the embedding of the $[\texttt{CLS}]$ token $e_\texttt{[CLS]} \in \mathbb{R}^{e_d}$ is appended to the column embeddings with $X = \left[e_\texttt{[CLS]}, e_1, e_2, \ldots e_{n}\right]$ , where $X \in \mathbb{R}^{e_d \times n +1}$. $X$ is passed through a stack of $L$ transformer layers. The then updated representation of the (CLS) token is used for prediction:
$$
\hat{y}=\operatorname{Linear}\left(\operatorname{ReLU}\left(\texttt{layer\_norm}\left(X[:,0]\right)\right)\right).
$$

The layer arrangement is depicted in Figure (...). It slightly differs from the one described in chapter [[🅰️Attention]] and [[🤖Transformer]].  [[@gorishniyRevisitingDeepLearning2021]] (p. 17) use *PreNorm*, or layer normalization at the beginning of the residual connection, which is easier to optimize ([[@xiongLayerNormalizationTransformer2020]], [[@wangLearningDeepTransformer2019]]). Pre-norm <mark style="background: #FFB8EBA6;">(Discuss influences on the results?)</mark> <mark style="background: #FF5582A6;">(TODO: look into code what is done in the classification case, guess there must be a sigmoid to turn logits into probs)</mark> 

Embedding both categorical and continuous inputs enables the Transformer to consider all features in the self-attention mechanism, but at an increase in computational cost, that may only be justified by higher classification *accuracies*. While the linear embedding of continuous inputs (see Eq. (1)) is straightforward and also found in <mark style="background: #FFF3A3A6;">(... see citations in [[@gorishniyEmbeddingsNumericalFeatures2022]], e.g., [[@somepalliSAINTImprovedNeural2021]], [[@guoEmbeddingLearningFramework2021]] etc.)</mark> <mark style="background: #FFF3A3A6;">(poor paper by https://openreview.net/attachment?id=SJlyta4YPS&name=original_pdf)</mark>, <mark style="background: #ADCCFFA6;">(see also [[@wangTransTabLearningTransferable]]. 
“propose to include column names into the tabular modeling. As a result, TransTab treats any tabular data as the composition of three elements: text (for categorical & textual cells and column names), continuous values (for numerical cells), and boolean values (for binary cells) .” ([Wang and Sun, p. 4](zotero://select/library/items/38EXIFQ9)) ([pdf](zotero://open-pdf/library/items/C9P6BQ9N?page=4&annotation=2L6GMYKW)))</mark>

They learn embeddings for continuous, categorical and binary data and include the column headings) a recent work of a [[@gorishniyEmbeddingsNumericalFeatures2022]] (p. 8) suggests, that periodic embeddings or piecewise-linear embeddings, can improve the model's performance for classification tasks. This idea has not yet been widely adopted by the research community. We compare all previously introduced architectures in [[🏅Results]] . In the subsequent section we discuss techniques, how [[🐈gradient-boosting]], [[🤖TabTransformer]], and the [[🤖FTTransformer]] can be extended for learning on partially-labelled data.


[^1:] In the official implementation of [[@gorishniyRevisitingDeepLearning2021]], the cls token is appended to the tensor at the end. Recall from chapter [[🅰️Attention]] that order is preserved. Also, the tensor is expanded, thus final column is repeated to provide an initial embedding for the cls token. As it get's contextualized, it shouldn't matter match, with what it is initialized. For explanation in this work, we stick to the explanation in the paper. 