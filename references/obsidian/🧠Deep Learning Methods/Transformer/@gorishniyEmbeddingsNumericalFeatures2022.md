*title:* On Embeddings for Numerical Features in Tabular Deep Learning
*authors:* Yury Gorishniy, Ivan Rubachev, Artem Babenko
*year:* 2022
*tags:* #embeddings #transformer #tabular-data
*status:* #📥
*related:*
*code:* [https://github.com/Yura52/tabular-dl-num-embeddings](https://github.com/Yura52/tabular-dl-num-embeddings)
# Notes Sebastian Raschka
-   Instead of designing new architectures for end-to-end learning, the authors focus on embedding methods for tabular data: (1) a piecewise linear encoding of scalar values and (2) periodic activation-based embeddings.
-   Experiments show that the embeddings are not only beneficial for transformers but other methods as well – multilayer perceptrons are competitive to transformers when trained on the proposed embeddings.
-   Using the proposed embeddings, ResNet, multilayer perceptrons, and transformers outperform CatBoost and XGBoost on several (but not all) datasets.
-   Small caveat: I would have liked to see a control experiment where the authors trained CatBoost and XGboost on the proposed embeddings.

# Annotations

- Could include interesting ideas: [[@gorishniyEmbeddingsNumericalFeatures2022]]
