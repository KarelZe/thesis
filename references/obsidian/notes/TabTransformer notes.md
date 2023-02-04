
Related:
- [[@huangTabTransformerTabularData2020]] propose TabTransformer
- [[@vaswaniAttentionAllYou2017]] propose the Transformer architecture
- [[@cholakovGatedTabTransformerEnhancedDeep2022]] Rubish paper that extends the TabTransformer
- [[@gorishniyRevisitingDeepLearning2021]] propose FTTransformer, which is similar to the



Notation adapted from [[@prokhorenkovaCatBoostUnbiasedBoosting2018]], [[@huangTabTransformerTabularData2020]]) and [[@phuongFormalAlgorithmsTransformers2022]]
Classification (ETransformer). Given a vocabulary $V$ and a set of classes $\left[N_{\mathrm{C}}\right]$, let $\left(x_n, c_n\right) \in$ $V^* \times\left[N_{\mathrm{C}}\right]$ for $n \in\left[N_{\text {data }}\right]$ be an i.i.d. dataset of sequence-class pairs sampled from $P(x, c)$. The goal in classification is to learn an estimate of the conditional distribution $P(c \mid x)$.

Notation. Let $V$ denote a finite set, called a $v o-$ cabulary, often identified with $\left[N_{\mathrm{V}}\right]:=\left\{1, \ldots, N_{\mathrm{V}}\right\}$

where $\boldsymbol{x} \equiv$ $\left\{\boldsymbol{x}_{\text {cat }}, \boldsymbol{x}_{\text {cont }}\right\}$.
The analogon for a sequence, i. e.  a row in the tabular dataset. 

Assume we observe a dataset of examples $\mathcal{D}=\left\{\left(\mathbf{x}_k, y_k\right)\right\}_{k=1 . . n}$, where $\mathbf{x}_k=\left(x_k^1, \ldots, x_k^m\right)$ is a random vector of $m$ features and $y_k \in \mathbb{R}$ is a target, which can be either binary or a numerical response. (from catboost paper [[@prokhorenkovaCatBoostUnbiasedBoosting2018]])
Let $(\boldsymbol{x}, y)$ denote a feature-target pair, where $\boldsymbol{x} \equiv$ $\left\{\boldsymbol{x}_{\text {cat }}, \boldsymbol{x}_{\text {cont }}\right\}$. The $\boldsymbol{x}_{\text {cat }}$ denotes all the categorical features and $x_{\text {cont }} \in \mathbb{R}^c$ denotes all of the $c$ continuous features. Let $\boldsymbol{x}_{\text {cat }} \equiv\left\{x_1, x_2, \cdots, x_m\right\}$ with each $x_i$ being a categorical feature, for $i \in\{1, \cdots, m\}$. (from [[@huangTabTransformerTabularData2020]] )