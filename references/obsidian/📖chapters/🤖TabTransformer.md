

![[tab_transformer.png]]
(own drawing. Inspired by [[@huangTabTransformerTabularData2020]]. Top layers a little bit different. They write MLP. I take the FFN with two hidden layers and an output layer. Change final feed forward network to gls-MLP, as it could be anything.) ^87bba0

TabTransformer of ([[@huangTabTransformerTabularData2020]]4) adapts the classical Transformer to the tabular domain to increase performance as well as robustness to noise and missing data over conventional deep learning approaches. In a post-norm encoder, the embeddings of categorical features are contextualized, while numerical inputs are processed as scalars. The former are normalized using layer norm, concatenated with the contextual embeddings, and passed through a multi-layer perceptron. More specifically, ([[@huangTabTransformerTabularData2020]]4--12) use a feed-forward network with two hidden layers, whilst other architectures and even non-deep models, are possible. For strictly numerical inputs, the model collapses to a gls-MLP with layer normalization. The joint architecture is depicted in [[#^87bba0]].

Categorical must be embedded first, to be accessible in the encoder. The embedding approach of ([[@huangTabTransformerTabularData2020]]3) is dichotomous and extends our observations from cref [[🛌Token Embedding]]. The embedding consists of a *feature-specific embedding* and a *shared embedding*. The *feature-specific embedding*, uniquely identifies each category. It is queried with an integer key $c_j \in C_j \cong\left[N_{\mathrm{C_j}}\right]$ from the embedding matrix $W_j \in \mathbb{R}^{e_d \times N_{C_j}}$ of the respective categorical column $j$ [^1]. Similar to the [[🛌Token Embedding]], a previous label encoding must be employed, that maps each category to its unique integer keys. 

Besides feature-specific embedding, a *shared embedding* is learned. This embedding is equal for all categories of one feature and is added or concatenated to the feature-specific embeddings to enable the model to distinguish classes in one column from those in other columns ([[@huangTabTransformerTabularData2020]]10). For the variant, that adds the shared embedding element-wisely, the embedding matrix $W_S$ is of dimension $\mathbb{R}^{e_d \times m}$. Overall, the joint embedding of $x_j$ is given by:
$$
\tag{6}
e_j = W_j[:c_j] + W_S[:j].
$$
Alternatively, categorical embeddings can also be concatenated from the feature-specific and shared embeddings. To maintain the overall embedding dimension of $d_{e}$, the dimensionality of the feature-specific embedding must be reduced to $d_{e} - \gamma$  and the remaining dimensionality $\gamma$ is attributed to the shared embedding. ([[@huangTabTransformerTabularData2020]]12) recommend an overall ratio of $\frac{7}{8}$ feature-specific embeddings versus $\frac{1}{8}$ shared embeddings. With the embedding matrices $W_j \in \mathbb{R}^{e_{d} -\gamma \times N_{C_j}}$ and $W_S \in \mathbb{R}^{\gamma \times m}$ , the embedding is now given by:
$$
\tag{7}
e_j = \left[W_j[:c_j], W_S[:j]\right]^T.
$$
Both approaches from Equation $(6)$ and $(7)$, achieve a similar performance experiments of ([[@huangTabTransformerTabularData2020]]11).  No additional positional embedding is required, as embeddings are unique per feature. 

Analogous to chapter [[🗼Overview Transformer]], the embedding of each row, or $X = [e_1, \cdots, e_m]$, is subsequently passed through several Transformer layers, ultimately resulting in contextualized embeddings. At the end of the encoder, the contextual embeddings are flattened and concatenated with the numerical features into a ($d_{e}  \times m + c$)-dimensional vector, which is input to the MLP ([[@huangTabTransformerTabularData2020]]3). Like before, a linear layer and a softmax activation are used to retrieve the class probabilities for $y$.

However, embedding and contextualizing of only the categorical inputs remains imperfect, as no numerical data is considered in the attention mechanism, and correlations between categorical and numerical features are lost due to the processing in different sub-networks ([[@somepalliSAINTImprovedNeural2021]]2). Also, the robustness to noise is hardly improved for numerical inputs. In a small experimental setup, ([[@somepalliSAINTImprovedNeural2021]]8) address this concern for the TabTransformer by also embedding numerical inputs, which leads to a lift in AUC by 2.34 % merely through embedding. Their observation integrates with a wider strand of literature that suggests, that models can profit from numerical embeddings, as we derived in chapter [[🛌Token Embedding]]. To dwell on this idea, we introduce the [[🤖FTTransformer]] next.

[^1:] The split across different embedding matrices is only *logically*. Also, additional embeddings may be created, which affects the dimensionality of the embedding metric. Similar to special tokens in the vocabulary, like the $\texttt{[UNK]}$ token for handing out-of-vocabulary items, an additional category can be reserved for unseen categories. ([[@huangTabTransformerTabularData2020]]10) use a separate embedding for missing categories.

**Notes on TabTransformer:**
[[🤖TabTransformer notes]]