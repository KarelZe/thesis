![[ft_transformer.png]]
(own drawing somewhat inspired by figure [[@gorishniyRevisitingDeepLearning2021]] and page 5; introduce CLS token? to differentiate from previous architectures?, not N but L; <mark style="background: #FFB86CA6;">Check if Softmax is correct? Where is ReLU?</mark>) ^6a12a6

[[🤖TabTransformer]] is limited by the ability to process only categorical embeddings in the Transformer unit. With FTTransformer ([[@gorishniyRevisitingDeepLearning2021]] 5) propose a competing architecture, that pairs an embedding unit for both numerical and categorical inputs, dubbed the feature tokenizer, with a Transformer. The complete architecture is depicted in cref-fig. Notably, the Transformer units use a pre-norm setup for easier optimisation, whereby the very first normalisation layer in the encoder is removed due to a propitious performance ([[@gorishniyRevisitingDeepLearning2021]]17). The upstream feature tokenizer transforms all features of $\boldsymbol{x}$ to their embeddings. If the $j$-th feature, $x_j$, is numerical, it is projected to its embedding $e_j \in \mathbb{R}^{e_d}$ by element-wise multiplication with a learnt vector $W_j \in \mathbb{R}^{d_{e}}$ . Moreover, a feature-dependent bias term $b_j \in \mathbb{R}$ is added, as noted in cref-eq.
$$
e_j= W_j x_j +b_j
$$
For categorical inputs, the embedding is implemented as a lookup table, analogous to [[💤Embeddings For Tabular Data]]. The specific embeddings $e_j$ are queried with a unique integer key $c_j \in C_j \cong\left[N_{\mathrm{C_j}}\right]$ from the learnt embedding matrix $W_j \in \mathbb{R}^{e_d \times N_{C_j}}$. Finally, a feature-specific bias term $b_j$ is added as shown in cref-eq.

$$
e_{j}=W_j[:c_j] +b_j
$$
Recall from our discussion on self-attention (see cref [[🅰️Attention]]), that each token encodes the tokens within the sequence. Based on this notion, ([[@devlinBERTPretrainingDeep2019]]4174) prepend a specialised $\texttt{[CLS]}$ token to the sequence, which stores the sequence's aggregate representation. Like any other token, the $\texttt{[CLS]}$ token is embedded first (see cref [[🛌Token Embedding]]), and contextualised in the encoder. Its final hidden state is then used for classification. ([[@gorishniyRevisitingDeepLearning2021]]4) adapt the idea of a $\texttt{[CLS]}$ token for tabular representation models. Similar to the embeddings of categorical or continuous features, the embedding of the $[\texttt{CLS}]$ token $e_\texttt{[CLS]} \in \mathbb{R}^{d_{e}}$ is prepended to the column embeddings with $X = \left[e_\texttt{[CLS]}, e_1, e_2, \ldots e_{n}\right]$, where $X \in \mathbb{R}^{d_{e} \times n +1}$. Like before, $X$ is passed through a stack of Transformer layers. The updated representation of the $\texttt{[CLS]}$ token is used exclusively for prediction: 
$$
P=\texttt{linear}\left(\texttt{ReLU}\left(\texttt{layer\_norm}\left(X[:,0]\right)\right)\right).
$$
<mark style="background: #BBFABBA6;">(Why ReLU; check code for output probabilities; see BERT code) -> to turn logits into probs, softmax must be applied.-</mark>

([[@gorishniyRevisitingDeepLearning2021]]8) achieve state-of-the-art performance through numerical and categorical embeddings. Embedding both categorical and continuous inputs enables the Transformer to attend to all other features, but at an increase in computational cost, that may only be justified by higher classification accuracies. While the linear embedding of continuous inputs from cref (1a) is conceptually simple, later works of ([[@gorishniyEmbeddingsNumericalFeatures2022]]8) suggests, that more sophisticated periodic embeddings or piece-wise-linear embeddings, can further improve the model's performance for classification tasks. This idea has not yet been widely adopted by the research community, we focus on linear embeddings instead.

In the subsequent section, extend all previous models including FTTransformer for learning on partially-labelled data.