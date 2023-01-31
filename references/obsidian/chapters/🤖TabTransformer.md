

![[tab_transformer.png]]
(own drawing. Inspired by [[@huangTabTransformerTabularData2020]]. Top layers a little bit different. They write MLP. I take the FFN with two hidden layers and an output layer. Change final feed forward network to multi-layer perceptron, as it could be anything. Also they call the <mark style="background: #FFB8EBA6;">input embedding a column embedding, use L instead of N)</mark> ^87bba0

Motivated by the success of contextual embeddings in natural language processing (cp. [[@devlinBERTPretrainingDeep2019]]; [[@liuRoBERTaRobustlyOptimized2019]]), [[@huangTabTransformerTabularData2020]] propose with *TabTransformer* an adaption of the classical Transformer for the tabular domain. 

*TabTransformer*, uses the Transformer encoder to learn contextualized embeddings for categorical features, as shown in Figure ([[#^87bba0]]]).  Featuring a multi-headed self-attention and post-norm arrangement, the transformer blocks, are identical to those found in [[@vaswaniAttentionAllYou2017]] (p. 3). Numerical inputs are processed separately from the categorical features. Numerical inputs are normalized using layer norm ([[@baLayerNormalization2016]]), concatenated with the contextual embeddings, and passed through a multi-layer perceptron. More specifically, [[@huangTabTransformerTabularData2020]] (p. 4; 12) use a feed-forward network with two hidden layers, whilst other architectures and even non-deep models, such as [[🐈gradient-boosting]], are applicable. For strictly numerical inputs, the model thus collapses to a multi-layer perceptron with layer normalization.

As discussed in chapter [[💤Embeddings for tabular data]], the [[🛌Token Embedding]] can be adapted for the tabular domain. Due to the tabular nature of the data, with features arranged in a row-column fashion, the token embedding (see chapter [[🛌Token Embedding]]) is replaced for a *column embedding*. Also the notation needs to be adapted to the tabular domain. We denote the data set with $D:=\left\{\left(\mathbf{x}_k, y_k\right) \right\}_{k=1,\cdots N}$ identified with $\left[N_{\mathrm{D}}\right]:=\left\{1, \ldots, N_{\mathrm{D}}\right\}$.  Each tuple $(\boldsymbol{x}, y)$ represents a row in the data set, and consist of the binary classification target $y \in \mathbb{R}$ and the vector of features $\boldsymbol{x} = \left\{\boldsymbol{x}_{\text{cat}}, \boldsymbol{x}_{\text{cont}}\right\}$, where $x_{\text{cont}} \in \mathbb{R}^c$ denotes all $c$ numerical features and $\boldsymbol{x}_{\text{cat}}\in \mathbb{R}^{m}$ all $m$ categorical features. We denote the cardinality of the $j$-th feature with $j \in 1, \cdots m$ with $N_{C_j}$.

In chapter [[🛌Token Embedding]], one lookup table suffices for storing the embeddings of all tokens within the sequence. Due to the heterogeneous nature of tabular data, every categorical column is independent of all $m-1$ other categorical columns. Thus, every feature requires its own logical embeddings.

The embedding approach of [[@huangTabTransformerTabularData2020]] (p. 3) is dichotomous, consisting of a *feature-specific embedding* and a *shared embedding*. It extends our observations in chapter [[💤Embeddings for tabular data]]. The *feature-specific embeddings* are queried with a unique integer key $c_j \in C_j \cong\left[N_{\mathrm{C_j}}\right]$ from the learned embedding matrix $W_j \in \mathbb{R}^{e_d \times N_{C_j}}$ of the categorical column [^1]. Similar to the [[🛌Token Embedding]], a previous label encoding must be employed, that maps each category to its unique integer keys. Besides feature-specific embedding, a *shared embedding* is learned. This embedding is equal for all categories of one feature and is added or concatenated to the feature-specific embeddings to enable the model to distinguish classes in one column from those in other columns ([[@huangTabTransformerTabularData2020]] p. 10). For the variant, where the shared embedding is added element-wisely, the embedding matrix $W_S$ is of dimension $\mathbb{R}^{e_d \times m}$. Overall, the joint *column embedding* of $x_j$ is given by:
$$
\tag{6}
e_j = W_j[:c_j] + W_S[:j].
$$
Alternatively, categorical embeddings can also be concatenated from the feature-specific and shared embeddings. To maintain the overall embedding dimension of $e_{d}$, the dimensionality of the feature-specific embedding must be reduced to $e_{d} - \gamma$  and the remaining dimensionality $\gamma$ is attributed to the shared embedding.  [[@huangTabTransformerTabularData2020]] (p. 12) recommend an overall ratio of $\frac{7}{8}$ feature-specific embeddings versus $\frac{1}{8}$ shared embeddings. With the embedding matrices $W_j \in \mathbb{R}^{(e_{d} -\gamma) \times N_{C_j}}$ and $W_S \in \mathbb{R}^{\gamma \times m}$ , the embedding is now given by:
$$
\tag{7}
e_j = \left[W_j[:c_j], W_S[:j]\right]^T.
$$
Both approaches from Equation $(6)$ and $(7)$, perform equally well in the experiments of [[@huangTabTransformerTabularData2020]] (p. 11). <mark style="background: #ADCCFFA6;">(What about high-cardinality issues?)</mark>

Following our observations in [[💤Embeddings for tabular data]], no positional embedding is required, as embeddings are not shared between features. Analogous to chapter [[🗼Overview Transformer (2 p)]], the embedding of each row, or $X = [e_1, \cdots, e_m$, is subsequently passed through several transformer layers, ultimately resulting in contextualized embeddings. At the end of the encoder, the contextual embeddings are flattened and concatenated with the numerical features into a ($e_{d}  \times m + c$)-dimensional vector, which serves as input to the multi-layer perceptron ([[@huangTabTransformerTabularData2020]] (p. 3)). Like before, a linear layer and a softmax activation are used to retrieve the class probabilities.

In large-scale experiments [[@huangTabTransformerTabularData2020]]  (p. 5 f.) can show, that the use of contextual embeddings elevates both the robustness to noise and missing data of the model. For various binary classification tasks, the TabTransformer outperforms other deep learning models e. g., vanilla multi-layer perceptrons in terms of *area under the curve* (AUC) and can compete with [[🐈gradient-boosting]].  

Yet, embedding and contextualizing of only the categorical inputs remains imperfect, as no numerical data is considered in the attention mechanism, and correlations between categorical and numerical features are lost due to the processing in different sub-networks ([[@somepalliSAINTImprovedNeural2021]]; p. 2). Also, the robustness to noise is hardly improved for numerical inputs. 

In a small experimental setup, [[@somepalliSAINTImprovedNeural2021]] (p. 8) address this concern for the TabTransformer by also embedding numerical inputs, which leads to a lift in AUC by 2.34 % merely through embedding. Their observation integrates with a wider strand of literature that suggests, that models can profit from numerical embeddings, as we derived in chapter [[💤Embeddings for tabular data]]. To dwell on this idea, we introduce the [[🤖FTTransformer]] next.

**Notes on TabTransformer:**
[[🤖tabtransformer notes]]


[^1:] Additional embeddings may be created. Similar to special tokens in the vocabulary, like the $\texttt{[UNK]}$ token for handing out-of-vocabulary items, an additional category can be reserved for unseen categories. Similarly, [[@huangTabTransformerTabularData2020]] use a separate embedding for missing categories.
