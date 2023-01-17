#tabular-data #tabtransformer #transformer #embeddings #categorical #continuous #tabular


![[tab_transformer.png]]
(own drawing. Inspired by [[@huangTabTransformerTabularData2020]]. Top layers a little bit different. They write MLP. I take the FFN with two hidden layers and an output layer. <mark style="background: #FFB8EBA6;">Better change label to MLP</mark>; Also they call the <mark style="background: #FFB8EBA6;">input embedding a column embedding, use L instead of N)</mark> ^87bba0

Motivated by the success of (cp. [[@devlinBERTPretrainingDeep2019]]; [[@liuRoBERTaRobustlyOptimized2019]]) of contextual embeddings in natural language processing, [[@huangTabTransformerTabularData2020]]  propose with *TabTransformer* an adaption of the classical Transformer for the tabular domain. *TabTransformer* is *encoder-only* and features a stack of Transformer layers (see chapter [[🤖Transformer]] or [[@vaswaniAttentionAllYou2017]]) to learn contextualized embeddings of categorical features from their parametric embeddings, as shown in Figure ([[#^87bba0]]]).  The transformer layers, are identical to those found in [[@vaswaniAttentionAllYou2017]] featuring multi-headed self-attention and a norm-last layer arrangement. Continuous inputs are normalized using layer norm ([[@baLayerNormalization2016]]) , concatenated with the contextual embeddings, and input into a multi-layer peceptron. More specifically, [[@huangTabTransformerTabularData2020]] (p. 4; 12) use a feed-forward network with two hidden layers, whilst other architectures and even non-deep models, such as [[🐈gradient-boosting]], are applicable. Thus, for strictly continuous inputs, the network collapses to a multi-layer perceptron with layer normalization.

Due to the tabular nature of the data, with features arranged in a row-column fashion, the token embedding (see chapter [[🛌Token Embedding]]) is replaced for a *column embedding*. Also the notation needs to be adapted to the tabular domain. We denote the data set with $D:=\left\{\left(\mathbf{x}_k, y_k\right) \right\}_{k=1,\cdots N}$ identified with $\left[N_{\mathrm{D}}\right]:=\left\{1, \ldots, N_{\mathrm{D}}\right\}$.  Each tuple $(\boldsymbol{x}, y)$ represents a row in the data set, and consist of the binary classification target $y_k \in \mathbb{R}$ and the vector of features 
$\boldsymbol{x} = \left\{\boldsymbol{x}_{\text{cat}}, \boldsymbol{x}_{\text{cont}}\right\}$, where $x_{\text{cont}} \in \mathbb{R}^c$ denotes all $c$ continuous features and $\boldsymbol{x}_{\text{cat}}\in \mathbb{R}^{m}$ all $m$ categorical features. We denote the cardinality of the $j$-th feature with $j \in 1, \cdots m$ with $N_{C_j}$. 

In chapter [[🛌Token Embedding]], one lookup table suffices for storing the embeddings of all tokens within the sequence. Due to the heterogenous nature of tabular data, every categorical column is independent from all $m-1$ other categorical columns. Thus, every column requires learning their own embedding matrix. 
The *feature-specific embeddings* are queried with a unique integer key $c_j \in C_j \cong\left[N_{\mathrm{C_j}}\right]$ from the learned embedding matrix $W_j \in \mathbb{R}^{e_d \times N_{C_j}}$ of the categorical column. Similar to the [[🛌Token Embedding]], a previous label encoding must be employed, to map the categories to their unique integer keys.
%%
They use +1 class, for NaN. Should already be addressed in pre-processing or imputed i. e. become their own category. Thus, I think it's ok, to not dwell on this here, as it is part of NC already?
%%
Additionally, a *shared embedding* is learned. This embedding is equal for all categories of one feature and is added or concatenated to the feature-specific embeddings to enable the model to distinguish classes in one column from those in other columns ([[@huangTabTransformerTabularData2020]] p. 10). For the variant, where the shared embedding is added element-wisely, the embedding matrix $W_S$ is of dimension $\mathbb{R}^{e_d \times m}$ .

Overall, the joint *column embedding* of $x_j$ is given by:
$$
e_j = W_j[:c_j] + W_S[:j].
$$
%%
Notation adapted from [[@prokhorenkovaCatBoostUnbiasedBoosting2018]], [[@huangTabTransformerTabularData2020]]) and [[@phuongFormalAlgorithmsTransformers2022]]
Classification (ETransformer). Given a vocabulary $V$ and a set of classes $\left[N_{\mathrm{C}}\right]$, let $\left(x_n, c_n\right) \in$ $V^* \times\left[N_{\mathrm{C}}\right]$ for $n \in\left[N_{\text {data }}\right]$ be an i.i.d. dataset of sequence-class pairs sampled from $P(x, c)$. The goal in classification is to learn an estimate of the conditional distribution $P(c \mid x)$.

Notation. Let $V$ denote a finite set, called a $v o-$ cabulary, often identified with $\left[N_{\mathrm{V}}\right]:=\left\{1, \ldots, N_{\mathrm{V}}\right\}$

where $\boldsymbol{x} \equiv$ $\left\{\boldsymbol{x}_{\text {cat }}, \boldsymbol{x}_{\text {cont }}\right\}$.
The analogon for a sequence, i. e.  a row in the tabular dataset. 

Assume we observe a dataset of examples $\mathcal{D}=\left\{\left(\mathbf{x}_k, y_k\right)\right\}_{k=1 . . n}$, where $\mathbf{x}_k=\left(x_k^1, \ldots, x_k^m\right)$ is a random vector of $m$ features and $y_k \in \mathbb{R}$ is a target, which can be either binary or a numerical response. (from catboost paper [[@prokhorenkovaCatBoostUnbiasedBoosting2018]])
Let $(\boldsymbol{x}, y)$ denote a feature-target pair, where $\boldsymbol{x} \equiv$ $\left\{\boldsymbol{x}_{\text {cat }}, \boldsymbol{x}_{\text {cont }}\right\}$. The $\boldsymbol{x}_{\text {cat }}$ denotes all the categorical features and $x_{\text {cont }} \in \mathbb{R}^c$ denotes all of the $c$ continuous features. Let $\boldsymbol{x}_{\text {cat }} \equiv\left\{x_1, x_2, \cdots, x_m\right\}$ with each $x_i$ being a categorical feature, for $i \in\{1, \cdots, m\}$. (from [[@huangTabTransformerTabularData2020]] )
%%

Note that categorical columns may be arranged in an arbitrary order and that the Transformer blocks are (... equivariant?), Thus, no [[🧵Positional encoding]] is required to inject the order. Analogous to chapter [[🤖Transformer]], the embedding of each row, or $X = [e_1, \cdots, e_m]$, are subsequently passed through several transformer layers, ultimately resulting in contextualized embeddings.  At the end of the encoder, the contextual embeddings are flattened and concatenated with the continuous inputs into a ($e_{d}  \times m + c$)-dimensional vector, which serves as input to the multi-layer perceptron ([[@huangTabTransformerTabularData2020]] (p. 3)). Like before, a linear layer and softmax activation <mark style="background: #FFB8EBA6;">(actually it's just a sigmoid due to the binary case, which is less computationally demanding for the binary case)</mark> are used to retrieve the class probabilities.

In large-scale experiments [[@huangTabTransformerTabularData2020]]  (p. 5 f.) can show, that the use of contextual embeddings elevates both the robustness to noise and missing data of the model. For various binary classification tasks, the TabTransformer outperforms other deep learning models e. g., vanilla multi-layer perceptrons in terms of *area under the curve* (AUC) and can compete with [[🐈gradient-boosting]].  

Yet, embedding and contextualizing categorical inputs remains imperfect, as no continuous data is considered for the contextualized embeddings and correlations between categorical and continuous features are lost due to the precessing in different subnetworks ([[@somepalliSAINTImprovedNeural2021]]; p. 2). Also, the robustness to noise in continous data is hardly improved. In a small experimental setup, [[@somepalliSAINTImprovedNeural2021]] (p. 8) address this concern for the TabTransformer by also embedding continuous inputs, which leads the substantial improvements in (AUC) . 

Their observation integrates with a wider strand of literature that suggests models can profit from embedding continuous features ([[@somepalliSAINTImprovedNeural2021]] (p. 8), [[@gorishniyRevisitingDeepLearning2021]] (p. ), [[@gorishniyEmbeddingsNumericalFeatures2022]] (p. )). To dwell on this idea, we introduce the [[🤖FTTransformer]], a transformer that contextualizes embeddings of all inputs in the subsequent section.

---
Related:
- [[@huangTabTransformerTabularData2020]] propose TabTransformer
- [[@vaswaniAttentionAllYou2017]] propose the Transformer architecture
- [[@cholakovGatedTabTransformerEnhancedDeep2022]] Rubish paper that extends the TabTransformer
- [[@gorishniyRevisitingDeepLearning2021]] propose FTTransformer, which is similar to the TabTransformer