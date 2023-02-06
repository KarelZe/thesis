
Exemplary, the encoding of the option type could be  $\text{P}\mapsto 1$; $\text{C}\mapsto 2$, which would result in a selection of the second column of the embedding matrix whenever a put is traded. 


bias should be a vector as per tehir NIPS talk, shifting is needed for good performance https://slideslive.com/38968794/revisiting-deep-learning-models-for-tabular-data?ref=recommended)

Exemplary, the encoding of the option type could be  $\text{P}\mapsto 1$; $\text{C}\mapsto 2$, which would result in a selection of the second column of the embedding matrix whenever a put is traded. 

$$
\tag{1}
e_j= 
\begin{cases}
    W_j x_j +b_j, & \text{if } x_j \text{ is numeric}\\
    W_j[:c_j] +b_j,              & x_j \text{ is categorical}.
\end{cases}

$$

$e_j = W_j x_j +b_j$ (if $x_j$ is numerical with $W_j \in \mathbb{R}^{e_d}$ (element-wise multiplication))
$e_j = W_j[:c_j] +b_j$ (if $x_j$ is numerical with $W_j \in \mathbb{R}^d$ ( $W_j \in \mathbb{R}^{e_d \times c_j}$ look up table. 

The size is dependent on the embedding dim and the cardinality of the categorical feature.  multiplication with one-hot-encoded vector; or just access the columns as done in chapter [[🛌Token Embedding]] ); $S_j$ is not introduced in paper.


encoder of  [[@vaswaniAttentionAllYou2017]] () and BERT ([[@devlinBERTPretrainingDeep2019]])


also found in (... see citations in [[@gorishniyEmbeddingsNumericalFeatures2022]], e.g., [[@somepalliSAINTImprovedNeural2021]], [[@guoEmbeddingLearningFramework2021]] etc.) (poor paper by https://openreview.net/attachment?id=SJlyta4YPS&name=original_pdf), (see also [[@wangTransTabLearningTransferable]]. 
“propose to include column names into the tabular modeling. As a result, TransTab treats any tabular data as the composition of three elements: text (for categorical & textual cells and column names), continuous values (for numerical cells), and boolean values (for binary cells) .” ([Wang and Sun, p. 4](zotero://select/library/items/38EXIFQ9)) ([pdf](zotero://open-pdf/library/items/C9P6BQ9N?page=4&annotation=2L6GMYKW)))

They learn embeddings for continuous, categorical and binary data and include the column headings) a recent work of a 