*title:* On Embeddings for Numerical Features in Tabular Deep Learning
*authors:* Yury Gorishniy, Ivan Rubachev, Artem Babenko
*year:* 2022
*tags:* #embeddings #transformer #tabular
*status:* #📥
*related:*
- article with explanations. https://towardsdatascience.com/transformers-for-tabular-data-part-3-piecewise-linear-periodic-encodings-1fc49c4bd7bc
*code:* [https://github.com/Yura52/tabular-dl-num-embeddings](https://github.com/Yura52/tabular-dl-num-embeddings)
# Notes Sebastian Raschka
-   Instead of designing new architectures for end-to-end learning, the authors focus on embedding methods for tabular data: (1) a piecewise linear encoding of scalar values and (2) periodic activation-based embeddings.
-   Experiments show that the embeddings are not only beneficial for transformers but other methods as well – multilayer perceptrons are competitive to transformers when trained on the proposed embeddings.
-   Using the proposed embeddings, ResNet, multilayer perceptrons, and transformers outperform CatBoost and XGBoost on several (but not all) datasets.
-   Small caveat: I would have liked to see a control experiment where the authors trained CatBoost and XGboost on the proposed embeddings.

# Annotations

“Transformer-like architectures have a specific way to handle numerical features of the data. Namely, they map scalar values of numerical features to high-dimensional embedding vectors, which are then mixed by the self-attention modules.” ([Gorishniy et al., 2022, p. 1](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=1&annotation=NINXYJZY))

“the existing architectures (Gorishniy et al., 2021; Somepalli et al., 2021; Kossen et al., 2021; Song et al., 2019; Guo et al., 2021) construct embeddings for numerical features using quite restrictive parametric mappings, e.g., linear functions, which can lead to suboptimal performance.” ([Gorishniy et al., 2022, p. 1](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=1&annotation=MFFC6S4Z))

“As another important finding, we demonstrate that the step of embedding the numerical features is universally beneficial for different deep architectures, not only for arXiv:2203.05556v2 [cs.LG] 15 Mar 202” ([Gorishniy et al., 2022, p. 1](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=1&annotation=7CKCYKGR))

“On Embeddings for Numerical Features in Tabular Deep Learning Transformer-like ones.” ([Gorishniy et al., 2022, p. 2](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=2&annotation=9QIH6IRL))

“To sum up, our contributions are as follows: 1. We show that embedding schemes for numerical features are an underexplored research question in tabular DL. Namely, we show that more expressive embedding schemes can provide substantial performance improvements over prior models. 2. We demonstrate that the profit from embedding numerical features is not specific for Transformer-like architectures, and proper embedding schemes benefit traditional models as well. 3. On a number of public benchmarks, we achieve the new state-of-the-art of tabular DL.” ([Gorishniy et al., 2022, p. 2](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=2&annotation=UI5Y5QHM))

“of tabular data requires mapping the scalar values of these features to high-dimensional embedding vectors. So far, the existing architectures perform this “scalar” → “vector” mapping by relatively simple computational blocks, which, in practice, can limit the model expressiveness.” ([Gorishniy et al., 2022, p. 2](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=2&annotation=T2YTSPF2))

“For instance, the recent FT-Transformer architecture (Gorishniy et al., 2021) employs only a single linear layer. In our experiments, we demonstrate that such embedding schemes can provide suboptimal performance, and more advanced schemes often lead to substantial profit.” ([Gorishniy et al., 2022, p. 2](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=2&annotation=XSUAFFCC))

“Feature binning. Binning is a discretization technique that converts numerical features to categorical features.” ([Gorishniy et al., 2022, p. 2](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=2&annotation=9BSQITEB))

“Periodic activations. Recently, periodic activation functions have become a key component in processing coordinates-like inputs, which is required in many applications. Examples include NLP (Vaswani et al., 2017), vision (Li et al., 2021), implicit neural representations (Mildenhall et al., 2020; Tancik et al., 2020; Sitzmann et al., 2020). In our work, we show that periodic activations can be used to construct powerful embedding modules for numerical features in tabular data problems.” ([Gorishniy et al., 2022, p. 2](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=2&annotation=7I8I5FY6))

“Importantly, contrary to some of the aforementioned papers, where components of the multidimensional coordinates are mixed (e.g. with linear layers) before passing them to periodic functions (Sitzmann et al., 2020; Tancik et al., 2020), we find it crucial to embed each feature separately before mixing them in the main backbone.” ([Gorishniy et al., 2022, p. 2](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=2&annotation=3MJWVU9F))

“We formalize the notion of ”embeddings for numerical features” in Equation 1: zi = fi ( x(num) i ) ∈ Rdi (1) where fi(x) is the embedding function for the i-th numerical feature, zi is the embedding of the i-th numerical feature and di is the dimensionality of the embedding” ([Gorishniy et al., 2022, p. 3](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=3&annotation=FFCWQQQY))

“While vanilla MLP is known to be a universal approximator (Cybenko, 1989; Hornik, 1991), in practice, due to optimization peculiarities, it has limitations in its learning capabilities (Rahaman et al., 2019). However, the recent work by Tancik et al. (2020) uncovers the case where changing the input space alleviates the above issue.” ([Gorishniy et al., 2022, p. 3](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=3&annotation=BREYEZN8))

“Namely, it allows existing DL backbones to achieve noticeably better results and significantly reduce the gap with Gradient Boosted Decision Trees.” ([Gorishniy et al., 2022, p. 10](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=10&annotation=DBBANJNW))

“We have also shown that traditional MLP-like models coupled with embeddings for numerical features can perform on par with attention-based models.” ([Gorishniy et al., 2022, p. 10](zotero://select/library/items/V9AJAB5T)) ([pdf](zotero://open-pdf/library/items/YMZCLKEQ?page=10&annotation=45RQWPSH))