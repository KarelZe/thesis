
title: TabTransformer: Tabular Data Modelling Using Contextual Embeddings
authors: Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin
year: 2020
*tags:* #deep-learning #gbm #semi-supervised #tabular #supervised-learning #transformer #pre-training #robustness
*status:* #📦 
*related:* 
- [[@borisovDeepNeuralNetworks2022]] 
- [[@gorishniyRevisitingDeepLearning2021]] (very similar idea, but look at differences)
*code:*
- https://github.com/jrzaurin/pytorch-widedeep
- https://github.com/lucidrains/tab-transformer-pytorch (faulty implementation; does not contain any shared embeddings; weird scaling)
- https://github.com/timeseriesAI/tsai/blob/be3c787d6e6d0e41839faa3e62d74145c851ef9c/tsai/models/TabTransformer.py (looks pretty reasonable)
- https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tab_transformer/tab_transformer.py (autogluon implementation)
- https://github.com/manujosephv/pytorch_tabular/blob/main/pytorch_tabular/models/tab_transformer/tab_transformer.py (pytorch tabular implementation)
- https://keras.io/examples/structured_data/tabtransformer/ (keras implementation)
*video:*
- Video by the author: https://www.youtube.com/watch?v=-ZdHhyQsvRc

## Notes
- TabTransformer is a tansformer-based architecture for tabular data. Using transformer layer ([[@vaswaniAttentionAllYou2017]]) it transforms the parametric embeddings of contextual data into robust contextual embeddings through the use of a sequence of multi-head attention-based transformer layers. The motivation for this step is, that highly correlated features within one column or across columns result in embedding pairs that are close in Euclidean space which could not be learnt for a vanilla MLP. Embeddings also increases robustness.
- TabTransformer only learns contextual embeddings on categorical features. Continous features are concatenated with the embeddings and fed into a vanilla neural network. [[@somepalliSaintImprovedNeural2021]] critize that information about correlations between categorical and continous features are lost, as continous features are not passed through the transformer block. Approches like FTTransformer (see [[@gorishniyRevisitingDeepLearning2021]]) embed both categorical and continous data and pass it through a transformer.
- Authors claim that TabTransformer improves by at least 1.0 % over other deep learning method and matches the performance of gbts. Unsupervised pre-training further improves performance with an improvement of 2.1 % over SOTA approaches.
- **Advantage:** 
	- The contextual embeddings makes the transformer highly robust against noise (i. e., random values) and missing values.
	- TabTransformer can be pre-trained using unlabeled data and fine-tuned with labelled data. This is advantage over other methods that require both labelled and unlabeled data in a single training pass.
- **Pro/ Cons of GBTs**
	- easy to interpret (+)
	- fast to train (+)
	- not suitable for continual training e. g., streaming (-)
	- not suitable due to their poor ability to estimate probabilities (-)
	- suboptimal for multi-modality data (e. g., image data + tabular data)
	- In their basic form not suitable for semi-supervised methods.
- Authors critize that the comparsion of GBTs and deep learning methods is mostly done on a limited number of datasets and often does not generalise. They actually show that GBTs are superior. 
- Their study finds that in large scale comparsion GBTs outperform recent architectures like [[@arikTabnetAttentiveInterpretable2020]]. Unsurprisingly TabTransformer shows superior performance to other-state-of-the-art deep learning methods and has a competitive perforance to tree-based ensembles. They test on 15 public data sets. They use five-fold cross validation with a 65/15/20% split. For hyperparameter tuning they make 20 runs for each fold. See hyperapram search space in appendix.
- **Architecture**
	- ![[tabtransformer-architecture.png]]
	- Architecture is based on the transformer architecture of [[@vaswaniAttentionAllYou2017]]
	- column embedding layer for categorical features. Parametric embedding is fed into $N$ transformer layers to obtain a contextual embedding. Each transformer layer consists of a multi-head self-attention layer and a position-wise feed-forward layer.
	- The contextual embeddings are concatenated along with continous features and input into a MLP.  The loss $\mathcal{L}$ is eigher mean square error or cross-entropy for classification.
	- Transformer uses self-attention / dot-product attention. 
	- For each categorical feature there is an own embedding lookup table with $d_i +1$ unique values. The additional value is required for missing values. The emeddings are different for each feature.
	- Architecture does not use positional encoding, as there is no ordering of features in tabular data. 
	- They perform an ablation study for different embedding strategies e. g.different choices for dimensions and adding unique identifiers and feature-value specific embeddings instead of concatenating them.

 - **Pre-training:**
	- They propose to pretrain using masked language modelling ([[@clarkElectraPretrainingText2020]]) and replaced token detection ([[@devlinBERTPretrainingDeep2019]])
	- Pre-training is only used in the semi-supervised scenario. They say, that there is no benefit if the entire data is labelled. Models only profit if unlabeled samples make up a large portion.
-
- **Visualisation of learnt embeddings on categorical features:**
	- To evaluate the effectiveness of the transformer layers, they visualise the (combined) contextual embeddings using a $t$-SNE plot calculated on the test set. Semantically similar classes are close to each other and form a cluster in the embedding space.
![[tab-transformer-embedding.png]]
- **Robustness to noise and missing values:**
	- Transformer is robust to noise. They tst by replacing a certain degree of values with random data. With more noise data, the TabTransformer performs better in prediction accuracy and is results deteriorate less than with MLPs. They suspect that the robustness comes from the learnt embeddings.
	- They also increase the number of missing values. Transformers are more robust to missing values than MLPs.
- **Semi-supervised results:**
	- For large number of unlabeled data pre-trained transformers outperform all other models in terms of AUC.
	- Tab-Transformer RTD performs better tahn TabTransformer-MLM as pre-training task is just a binary classification instead of a multi-class classification. They say, that the result is consistent with 
	- [[@clarkElectraPretrainingText2020]]

## Search space
- AdamW optimizer
- const learning rate
- early stopping after 15 epochs
- hidden (embedding) dim, no of layers, no attention heads. MLP sizes are 4x and 2x the input
- 2 , 4, 8 attention heads
- hid dim 32, 64, 128, 256
- no layers 1, 2, 3, 6, 12
- mlp selu, batchnorm, 2 hidden layers, capacity 8* l with l = d/8, second layer m*l with m in (1,3)

## Feature Engineering
- They write that learnt embeddings improve performance as long as the cardinality of variables is signifcantly less than the number of datapoints. If not the feature can cause overfitting effects.
- For scalar features they suggest to include three types of rescaled features and one categorical feature. They argue that redundant encodings for scalars don't cause overfitting, but can help differentiate between useful and unuseful features.

## Architecture
Let $(x, y)$ denote a feature-target pair, where $x \equiv$ $\left\{x_{\text {cat }}, x_{\text {cont }}\right\}$. The $x_{\text {cat }}$ denotes all the categorical features and $x_{\text {cont }} \in \mathbb{R}^c$ denotes all of the $c$ continuous features. Let $x_{\text {cat }} \equiv\left\{x_1, x_2, \cdots, x_m\right\}$ with each $x_i$ being a categorical feature, for $i \in\{1, \cdots, m\}$.

We embed each of the $x_i$ categorical features into a parametric embedding of dimension $d$ using Column embedding, which is explained below in detail. Let $e_{\phi_i}\left(x_i\right) \in \mathbb{R}^d$ for $i \in\{1, \cdots, m\}$ be the embedding of the $x_i$ feature, and $E_\phi\left(x_{\mathrm{cat}}\right)=\left\{e_{\phi_1}\left(x_1\right), \cdots, e_{\phi_m}\left(x_m\right)\right\}$ be the set of embeddings for all the categorical features.

Next, these parametric embeddings $\boldsymbol{E}_\phi\left(x_{\text {cat }}\right)$ are inputted to the first Transformer layer. The output of the first Transformer layer is inputted to the second layer Transformer, and so forth. Each parametric embedding is transformed into contextual embedding when outputted from the top layer Transformer, through successive aggregation of context from other embeddings. We denote the sequence of Transformer layers as a function $f_\theta$. The function $f_\theta$ operates on parametric embeddings $\left\{e_{\phi_1}\left(x_1\right), \cdots, e_{\phi_m}\left(x_m\right)\right\}$ and returns the corresponding contextual embeddings $\left\{h_1, \cdots, h_m\right\}$ where $h_i \in \mathbb{R}^d$ for $i \in\{1, \cdots, m\}$

The contextual embeddings $\left\{h_1, \cdots, h_m\right\}$ are concatenated along with the continuous features $x_{\text {cont }}$ to form a vector of dimension $(d \times m+c)$. This vector is inputted to an MLP, denoted by $g_\psi$, to predict the target $y$. Let $H$ be the cross-entropy for classification tasks and mean square error for regression tasks. We minimise the following loss function $\mathcal{L}(x, y)$ to learn all the TabTransformer parameters in an end-to-end learning by the first-order gradient methods. The TabTransformer parameters include $\phi$ for column embedding, $\theta$ for Transformer layers, and $\psi$ for the top MLP layer.
$$
\mathcal{L}(\boldsymbol{x}, y) \equiv H\left(g_{\boldsymbol{\psi}}\left(f_{\boldsymbol{\theta}}\left(\boldsymbol{E}_\phi\left(\boldsymbol{x}_{\text {cat }}\right)\right), \boldsymbol{x}_{\text {cont }}\right), y\right) .
$$
Below, we explain the Transformer layers and column embedding.

Important details on column embedding:
Column Embedding. The first study is on the choice of column embedding - shared parameters $\boldsymbol{c}_{\phi_i}$ across the embeddings of multiple classes in column $i$ for $i$ $\{1,2, \ldots, m\}$. In particular, we study the optimal dimension of $\boldsymbol{c}_{\phi_i}, \ell$. An alternative choice is to element-wisely add the unique identifier $\boldsymbol{c}_{\phi_i}$ and feature-value specific embeddings $\boldsymbol{w}_{\phi_{i j}}$ rather than concatenating them. In that case, both the dimension of $\boldsymbol{c}_{\phi_i}$ and $\boldsymbol{w}_{\phi_{i j}}$ are equal to the dimension of embedding $d$. The goal of having column embedding is to

## Comparsion FT-Transformer and TabTransformer

![[ft-tab-transformer.png]]
(Guess it's taken from some yandex slide. found here https://preview.redd.it/mk28f629uxw91.png?width=1916&format=png&auto=webp&s=9a801d48189cf7fd9d4039e107e236aaa93f6a6f)

## Notes Sebastian Raschka
-   Several open-source implementations are available on GitHub, however, I could not find the official implementation, so the results from this paper must be taken with a grain of salt.
-   The paper proposes a transformer-based architecture based on self-attention that can be applied to tabular data.
-   In addition to the purely supervised regime, the authors propose a semi-supervised approach leveraging unsupervised pre-training.
-   Looking at the average AUC across 15 datasets, the proposed TabTransformer (82.8) is on par with gradient-boosted trees (82.9).


## Notes from the keras blog
https://keras.io/examples/structured_data/tabtransformer/#experiment-2-tabtransformer

1.  All the categorical features are encoded as embeddings, using the same `embedding_dims`. This means that each value in each categorical feature will have its own embedding vector.
2.  A column embedding, one embedding vector for each categorical feature, is added (point-wise) to the categorical feature embedding.
3.  The embedded categorical features are fed into a stack of Transformer blocks. Each Transformer block consists of a multi-head self-attention layer followed by a feed-forward layer.
4.  The outputs of the final Transformer layer, which are the _contextual embeddings_ of the categorical features, are concatenated with the input numerical features, and fed into a final MLP block.
5.  A `softmax` classifer is applied at the end of the model.

The [paper](https://arxiv.org/abs/2012.06678) discusses both addition and concatenation of the column embedding in the _Appendix: Experiment and Model Details_ section. The architecture of TabTransformer is shown below, as presented in the paper.



## Notes from Talk by Zohar Karnin
(see here: https://www.youtube.com/watch?v=-ZdHhyQsvRc)

- networks can only interpret numbers.
	- Continuous: Easy for numerical input
	- Categorical: convert to embedding

- issues with categorical data
	- rare categories -> similar to infrequent words
	- similar categories; have a head-start by not initialising randomly -> semantic meaning of words
	- use transfer learning 
	- handle context

- pre-training is boroughed from nlp. Two approaches:
	- MLM: convert some categories to missing embedding, reconstruct the category
	- Replaced token detection: Replace category with random replacement, detect replaced
- not necessarily a mlp followed by transformer. Could be any network
- in a semi-supervised comparsion they also compared gbrts on pseudo labelled data

## Annotations
# Annotations  
(23/01/2023, 15:01:26)

“The TabTransformer is built upon self-attention based Transformers. The Transformer layers transform the embeddings of categorical features into robust contextual embeddings to achieve higher prediction accuracy.” ([Huang et al., 2020, p. 1](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=1&annotation=VXCRZEIK))

“Through extensive experiments on fifteen publicly available datasets, we show that the TabTransformer outperforms the state-of-theart deep learning methods for tabular data by at least 1.0% on mean AUC, and matches the performance of tree-based ensemble models.” ([Huang et al., 2020, p. 1](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=1&annotation=9VB46CGG))

“Furthermore, we demonstrate that the contextual embeddings learnt from TabTransformer are highly robust against both missing and noisy data features, and provide better interpretability.” ([Huang et al., 2020, p. 1](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=1&annotation=LN4F5NXL))

“The state-of-the-art for modelling tabular data is treebased ensemble methods such as the gradient boosted decision trees (GBDT)” ([Huang et al., 2020, p. 1](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=1&annotation=7E8H66FJ))

“The tree-based ensemble models can achieve competitive prediction accuracy, are fast to train and easy to interpret. These benefits make them highly favourable among machine learning practitioners. However, the tree-based models have several limitations in comparison to deep learning models. (a) They are not suitable for continual training from streaming data, and do not allow efficient end-to-end learning of image/text encoders in presence of multi-modality along with tabular data. (b) In their basic form they are not suitable for state-of-the-art” ([Huang et al., 2020, p. 1](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=1&annotation=WS4TEXDN))

“semi-supervised learning methods.” ([Huang et al., 2020, p. 1](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=1&annotation=BARDHRGP))

“Although these deep learning models achieve comparable prediction accuracy, they do not address all the limitations of GBDT and MLP. Furthermore, their comparisons are done in a limited setting of a handful of datasets. In particular, in Section 3.3 we show that when compared to standard GBDT on a large collection of datasets, GBDT perform significantly better than these recent models.” ([Huang et al., 2020, p. 1](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=1&annotation=LF8LB83Q))

“The TabTransformer is built upon Transformers (Vaswani et al. 2017) to learn efficient contextual embeddings of categorical features.” ([Huang et al., 2020, p. 1](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=1&annotation=ACLSR9L5))

“In particular, TabTransformer applies a sequence of multi-head attention-based Transformer layers on parametric embeddings to transform them into contextual embeddings,” ([Huang et al., 2020, p. 2](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=2&annotation=7ESD2X6Q))

“We find that highly correlated features (including feature pairs in the same column and cross column) result in embedding vectors that are close together in Euclidean distance, whereas no such pattern exists in contextfree embeddings learnt in a baseline MLP model.” ([Huang et al., 2020, p. 2](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=2&annotation=4VUZZ6NC))

“We also study the robustness of the TabTransformer against random missing and noisy data. The contextual embeddings make them highly robust in comparison to MLPs.” ([Huang et al., 2020, p. 2](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=2&annotation=R43KGY3T))

“One of the key benefits of our proposed method for semi-supervised learning is the two independent training phases: a costly pre-training phase on unlabeled data and a lightweight fine-tuning phase on labelled data.” ([Huang et al., 2020, p. 2](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=2&annotation=HHYGXHGS))

“that require a single training job including both the labelled and unlabeled data.” ([Huang et al., 2020, p. 2](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=2&annotation=TJ746Y4L))

“The TabTransformer architecture comprises a column embedding layer, a stack of N Transformer layers, and a multilayer perceptron. Each Transformer layer (Vaswani et al. 2017) consists of a multi-head self-attention layer followed by a position-wise feed-forward layer.” ([Huang et al., 2020, p. 2](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=2&annotation=NJ4IIWB4))

“Let (x, y) denote a feature-target pair, where x ≡ {xcat, xcont}. The xcat denotes all the categorical features and xcont ∈ Rc denotes all of the c continuous features. Let xcat ≡ {x1, x2, · · · , xm} with each xi being a categorical feature, for i ∈ {1, · · · , m}. We embed each of the xi categorical features into a parametric embedding of dimension d using Column embedding, which is explained below in detail. Let eφi (xi) ∈ Rd for i ∈ {1, · · · , m} be the embedding of the xi feature, and Eφ(xcat) = {eφ1 (x1), · · · , eφm (xm)} be the set of embeddings for all the categorical features. Next, these parametric embeddings Eφ(xcat) are inputted to the first Transformer layer. The output of the” ([Huang et al., 2020, p. 2](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=2&annotation=XDSY3C4K))

“first Transformer layer is inputted to the second layer Transformer, and so forth. Each parametric embedding is transformed into contextual embedding when outputted from the top layer Transformer, through successive aggregation of context from other embeddings. We denote the sequence of Transformer layers as a function fθ. The function fθ operates on parametric embeddings {eφ1 (x1), · · · , eφm (xm)} and returns the corresponding contextual embeddings {h1, · · · , hm} where hi ∈ Rd for i ∈ {1, · · · , m}. The contextual embeddings {h1, · · · , hm} are concatenated along with the continuous features xcont to form a vector of dimension (d × m + c). This vector is inputted to an MLP, denoted by gψ, to predict the target y. Let H be the cross-entropy for classification tasks and mean square error for regression tasks. We minimise the following loss function L(x, y) to learn all the TabTransformer parameters in an end-to-end learning by the first-order gradient methods. The TabTransformer parameters include φ for column embedding, θ for Transformer layers, and ψ for the top MLP layer. L(x, y) ≡ H(gψ(fθ(Eφ(xcat)), xcont), y) . (1) Below, we explain the Transformer layers and column embedding.” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=6SV6X2UE))

“-to-end learning by the first-order gradient methods. The TabTransformer parameters include φ for column embedding, θ for Transformer layers, and ψ for the top MLP layer. L(x, y) ≡ H(gψ(fθ(Eφ(xcat)), xcont), y) . (1) Below, we” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=3P5FEELB))

“The first layer expands the embedding to four times its size and the second layer projects it back to its original size.” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=P6GTHMG9))

“Since, in tabular data, there is no ordering of the features, we do not use positional encodings.” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=CEK56F22))

“We explore two different types of pre-training procedures, the masked language modelling (MLM) (Devlin et al. 2019) and the replaced token detection (RTD) (Clark et al. 2020).” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=7SXSI5CA))

“Given an input xcat = {x1, x2, ..., xm}, MLM randomly selects k% features from index 1 to m and masks them as missing. The Transformer layers along with the column embeddings are trained by minimising cross-entropy loss of a multi-class classifier that tries to predict the original features of the masked features, from the contextual embedding outputted from the top-layer Transformer.” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=Q95BK24A))

“Instead of masking features, RTD replaces the original feature by a random value of that feature. Here, the loss is minimised for a binary classifier that tries to predict whether or not the feature has been replaced. The RTD procedure as proposed in (Clark et al. 2020) uses auxiliary generator for sampling a subset of features that a feature should be replaced with. The reason they used an auxiliary encoder network as the generator is that there are tens of thousands of tokens in language data and a uniformly random token is too easy to detect. In contrast, (a) the number of classes within each categorical feature is typically limited; (b) a different binary classifier is defined for each column rather than a shared one, as each column has its own embedding lookup table.” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=WQQXQE5K))

“We evaluate TabTransformer and baseline models on 15 publicly available binary classification datasets” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=YPLBAW4J))

“Each dataset is divided into five cross-validation splits.” ([Huang et al., 2020, p. 3](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=3&annotation=CF6NDX9V))

“For the TabTransformer, the hidden (embedding) dimension, the number of layers and the number of attention heads are fixed to 32, 6, and 8 respectively. The MLP layer sizes are set to {4 × l, 2 × l}, where l is the size of its input. For hyperparameter optimisation (HPO), each model is given 20 HPO rounds for each cross-validation split.” ([Huang et al., 2020, p. 4](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=4&annotation=EZP4RKR9))

“Note, the pre-training is only applied in semi-supervised scenario. We do not find much benefit in using it when the entire data is labelled. Its benefit is evident when there is a large number of unlabeled examples and a few labelled examples.” ([Huang et al., 2020, p. 4](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=4&annotation=VN52BT3H))

“Next, we take contextual embeddings from different layers of the Transformer and compute a t-SNE plot (Maaten and Hinton 2008) to visualise their similarity in function space. More precisely, for each dataset we take its test data, pass their categorical features into a trained TabTransformer, and extract all contextual embeddings (across all columns) from a certain layer of the Transformer.” ([Huang et al., 2020, p. 4](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=4&annotation=6JJL5DJF))

“We can see that semantically similar classes are close Table 1: Comparison between TabTransfomers and the baseline MLP. The evaluation metric is AUC in percentage. Dataset Baseline MLP TabTransformer Gain (%) albert 74.0 75.7 1.7 1995 income 90.5 90.6 0.1 dota2games 63.1 63.3 0.2 hcdr main 74.3 75.1 0.8 adult 72.5 73.7 1.2 bank marketing 92.9 93.4 0.5 blastchar 83.9 83.5 -0.4 insurance co 69.7 74.4 4.7 jasmine 85.1 85.3 0.2 online shoppers 91.9 92.7 0.8 philippine 82.1 83.4 1.3 qsar bio 91.0 91.8 0.8 seismicbumps 73.5 75.1 1.6 shrutime 84.6 85.6 1.0 spambase 98.4 98.5 0.1 with each other and form clusters in the embedding space. Each cluster is annotated by a set of labels.” ([Huang et al., 2020, p. 4](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=4&annotation=598TXXKA))

“n addition to prove the effectiveness of Transformer layers, on the test data we take all of the contextual embeddings from each Transformer layer of a trained TabTransformer, use the embeddings from each layer along with the continuous variables as features, and separately fit a linear model with target y. Since all of the experimental datasets are for binary classification, the linear model is logistic regression. The motivation for this evaluation is defining the success of a simple linear model as a measure of quality for the learnt embeddings.” ([Huang et al., 2020, p. 4](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=4&annotation=TSMZQ2GS))

“We further demonstrate the robustness of TabTransformer on the noisy data and data with missing values, against the baseline MLP. We consider these two scenarios only on categorical features to specifically prove the robustness of contextual embeddings from the Transformer layers.” ([Huang et al., 2020, p. 5](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=5&annotation=M3669EMG))

“On the test examples, we firstly contaminate the data by replacing a certain number of values by randomly generated ones from the corresponding columns (features)” ([Huang et al., 2020, p. 5](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=5&annotation=JBNAVQGK))

“As the noisy rate increases, TabTransformer performs better in prediction accuracy and thus is more robust than MLP.” ([Huang et al., 2020, p. 5](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=5&annotation=N2IZT5BP))

“We conjecture that the robustness comes from the contextual property of the embeddings. Despite a feature being noisy, it draws information from the correct features allowing for a certain amount of correction” ([Huang et al., 2020, p. 5](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=5&annotation=U4YM67RX))

“Similarly, on the test data we artificially select a number of values to be missing and send the data with missing values to a trained TabTransformer to compute the prediction score.” ([Huang et al., 2020, p. 5](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=5&annotation=ENKX5A8P))

“There are two options to handle the embeddings of missing values: (1) Use the average learnt embeddings over all classes in the corresponding column; (2) the embedding for the class of missing value, the additional embedding for each column mentioned in Section 2. Since the benchmark datasets do not contain enough missing values to effectively train the embedding in option (2), we use the average embedding in (1) for imputation.” ([Huang et al., 2020, p. 5](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=5&annotation=X9BGH6LU))

“Specifically, we compare our pretrained and then fine-tuned TabTransformer-RTD/MLM against following semi-supervised models: (a) Entropy Regularisation (ER) (Grandvalet and Bengio 2006) combined with MLP and TabTransformer (b) Pseudo Labelling (PL) (Lee 2013) combined with MLP, TabTransformer, and GBDT (Jain 2017) (c) MLP (DAE): an unsupervised pre-training method designed for deep models on tabular data: the swap noise Denoising AutoEncoder (Jahrer 2018).” ([Huang et al., 2020, p. 6](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=6&annotation=5XJ4Z9FF))

“When the number of unlabeled data is large, Table 3 shows that our TabTransformer-RTD and TabTransformer-MLM significantly outperform all the other competitors. Particularly, TabTransformer-RTD/MLM improves over all the other competitors by at least 1.2%, 2.0% and 2.1% on mean AUC for the scenario of 50, 200, and 500 labelled data points respectively.” ([Huang et al., 2020, p. 6](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=6&annotation=6BKD38VS))

“Furthermore, we observe that when the number of unlabeled data is small as shown in Table 4, TabTransformerRTD performs better than TabTransformer-MLM, thanks to its easier pre-training task (a binary classification) than that of MLM (a multi-class classification). This is consistent with the finding of the ELECTRA paper (Clark et al. 2020).” ([Huang et al., 2020, p. 7](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=7&annotation=HU3JVTCV))

“Both evaluation results, Table 3 and Table 4, show that our TabTransformer-RTD and Transformers-MLM models are promising in extracting useful information from unlabeled data to help supervised training, and are particularly useful when the size of unlabeled data is large.” ([Huang et al., 2020, p. 7](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=7&annotation=D58VVW2M))

“The Pseudo labelling uses the current network to infer pseudo-labels of unlabeled examples, by choosing the most confident class. These pseudo-labels are treated like human-provided labels in the cross entropy loss” ([Huang et al., 2020, p. 7](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=7&annotation=MY3JLV44))

“The goal of having column embedding is to enable the model to distinguish the classes in one column from those in the other columns.” ([Huang et al., 2020, p. 10](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=10&annotation=AB5WAUWI))

“aining and testing conditions. As all the datasets are for binary classification, the cross entropy loss was used for both supervised and semisupervised training (for pre-training, the problem is binary classification in RTD and multi-class classification in MLM). For all deep models, the AdamW optimizer (Loshchilov and Hutter 2017) was used to update the model parameters, and a constant learning rate was applied throughout each training job. All models used early stopping based on the performance on the validation set and the early stopping patience (the number of epochs) is set as 15.” ([Huang et al., 2020, p. 10](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=10&annotation=GHEPLKP2))

“Table 5: Performance of TabTransformer with no column embedding, concatenation column embedding, and addition column embedding” ([Huang et al., 2020, p. 11](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=11&annotation=JLGAZJPH))

“For TabTransformer, the hidden (embedding) dimension, the number of layers and the number of attention heads in the Transformer were fixed to 32, 6, and 8 respectively during the experiments. The MLP layer sizes were fixed to {4 × l, 2 × l}, where l was the size of its input. However, these parameters were optimally selected based on 50 rounds of HPO run on 5 datasets. The search spaces were the number of attention heads {2, 4, 8}, the hidden dimension {32, 64, 128, 256}, and the number of layers {1, 2, 3, 6, 12}. The search spaces of the first and second hidden layer in MLP are exactly the same as those in MLP model setting. The dimension of cφi , \` was chosen as d/8 based on the ablation study in Appendix A.” ([Huang et al., 2020, p. 12](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=12&annotation=PH6SVV3K))

“or categorical variables, the processing options include whether to one-hot encode versus learn a parametric embedding, what embedding dimension to use, and how to apply dropout regularisation (whether to drop vector elements or whole embeddings). In our experiments we found that learnt embeddings nearly always improved performance as long as the cardinality of the categorical variable is significantly less than the number of data points, otherwise the feature is merely a means for the model to overfit” ([Huang et al., 2020, p. 13](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=13&annotation=HKWS6JF4))

“For scalar variables, the processing options include how to re-scale the variable (via quantiles, normalisation, or log scaling) or whether to quantize the feature and treat it like a categorical variable. While we have not explored this idea fully, the best strategy is likely to use all the different types of encoding in parallel, turning each scalar feature into three re-scaled features and one categorical feature. Unlike learning embeddings for high-cardinality categorical features, adding potentially-redundant encodings for scalar variables should not lead to overfitting, but can make the difference between a feature being useful or not.” ([Huang et al., 2020, p. 13](zotero://select/library/items/MH4GW34I)) ([pdf](zotero://open-pdf/library/items/QYWHEUYE?page=13&annotation=VB99PZDX))