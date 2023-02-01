![[ft_transformer.png]]
(own drawing somewhat inspired by figure [[@gorishniyRevisitingDeepLearning2021]] and page 5; introduce CLS token? to differentiate from previous architectures?, not N but L; <mark style="background: #FFB86CA6;">Check if Softmax is correct? Where is ReLU?</mark>) ^6a12a6

The FTTransformer of [[@gorishniyRevisitingDeepLearning2021]] is an adaption of an encoder-only Transformer for the tabular domain. Opposed to the [[TabTransformer]], the FTTransformer, introduces a feature tokenizer for embedding both numerical and categorical inputs, that are contextualized in a stack of Transformer layers, as shown in Figure [[#^6a12a6]].  In contrast to the TabTransformer, where all contextualized embeddings contribute to the prediction, the output probabilities are only derived from the final hidden state of a single, specialized token, the $\texttt{[CLS]}$ token. 

The feature tokenizer transforms all features of $x$ to their embeddings. If the $j$-th feature, $x_j$, is numerical, it is projected to its embedding $e_j \in \mathbb{R}^{e_d}$ by element-wise multiplication with a learned vector $W_j \in \mathbb{R}^{e_d}$ . More over, a feature-dependent bias term $b_j \in \mathbb{R}$ is added, as noted in Equation (1).

For categorical inputs, the embedding is implemented as a lookup table, similar to the embedding in the chapter [[💤Embeddings For Tabular Data]]. We denote the cardinality of the $j$-th feature with $N_{C_j}$. The specific embeddings $e_j$ are queried with a unique integer key $c_j \in C_j \cong\left[N_{\mathrm{C_j}}\right]$ from the learned embedding matrix $W_j \in \mathbb{R}^{e_d \times N_{C_j}}$. Finally, a feature-specific bias term $b_j$ is added. Overall for $x_j$:
$$
\tag{1}
e_j= 
\begin{cases}
    W_j x_j +b_j, & \text{if } x_j \text{ is numeric}\\
    W_j[:c_j] +b_j,              & x_j \text{ is categorical}.
\end{cases}

$$



![[mashup-tokenizer-fttransformer.png]]
(by [[@devlinBERTPretrainingDeep2019]] (p. 4185) and [[@gorishniyRevisitingDeepLearning2021]] (p. 4)) ^23bb5c

For classification tasks in the language representation model BERT, [[@devlinBERTPretrainingDeep2019]] (p. 4,174) propose to prepend a specialized $\texttt{[CLS]}$ token to every sequence, that stores its aggregate representation. Like any other token, the $\texttt{[CLS]}$ token is embedded first (see chapter [[🛌Token Embedding]]), and contextualized in the subsequent Transformer layers. Its final representation is then used in the classification task.

[[@gorishniyRevisitingDeepLearning2021]] (p. 4) adapt the idea of a $\texttt{[CLS]}$ token for tabular representation models, as visualized in Figure [[🤖FTTransformer#^23bb5c]]. Similar to the embeddings of a categorical or continuous feature, the embedding of the $[\texttt{CLS}]$ token $e_\texttt{[CLS]} \in \mathbb{R}^{e_d}$ is prepended to the column embeddings with $X = \left[e_\texttt{[CLS]}, e_1, e_2, \ldots e_{n}\right]$ , where $X \in \mathbb{R}^{e_d \times n +1}$. Like before, $X$ is passed through a stack of $L$ Transformer layers. The then updated representation of the (CLS) token is used for prediction:
$$
P=\texttt{linear}\left(\texttt{ReLU}\left(\texttt{layer\_norm}\left(X[:,0]\right)\right)\right).
$$
<mark style="background: #BBFABBA6;">(Why ReLU; check code for output probabilities; see BERT code) -> to turn logits into probs, softmax must be applied?</mark>

The specific layer arrangement is different from the one of [[@vaswaniAttentionAllYou2017]], as depicted in Figure [[#^6a12a6]]. Most notably the FTTransformer uses pre-norm for easier optimization [[@gorishniyRevisitingDeepLearning2021]] (p. 17). Moreover, the authors remove the first normalization from the first Transformer block due to a propitious performance. 

[[@gorishniyRevisitingDeepLearning2021]] can achieve state-of-the-art performance through numerical and categorical embeddings. Embedding both categorical and continuous inputs enables the Transformer to consider all features in the self-attention mechanism, but at an increase in computational cost, that may only be justified by higher classification accuracies. While the linear embedding of continuous inputs (see Equation (1)) is conceptually simple, later works of [[@gorishniyEmbeddingsNumericalFeatures2022]] (p. 8) suggests, that periodic embeddings or piece-wise-linear embeddings, can improve the model's performance for classification tasks. This idea has not yet been widely adopted by the research community. We compare all previously introduced architectures in the section [[🏅Results]]. In the subsequent section, we discuss techniques, how [[🐈gradient-boosting]], [[🤖TabTransformer]], and the FTTransformer can be extended for learning on partially-labelled data.

[^1:] In the official implementation of [[@gorishniyRevisitingDeepLearning2021]], the $\texttt{[CLS]}$ token is appended to the tensor at the end. Recall from the chapter [🅰️Attention]] that order is preserved. Also, the tensor is expanded, thus final column is repeated to provide an initial embedding for the $\texttt{[CLS]}$ token. For the explanation in this work, we stick to the explanation in the paper. 