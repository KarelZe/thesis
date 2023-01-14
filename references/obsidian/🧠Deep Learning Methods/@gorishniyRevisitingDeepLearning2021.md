
title: Revisiting Deep Learning Models for Tabular Data
authors: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko
year: 2020
*tags:* #deep-learning #gradient_boosting #semi-supervised #transformer #tabtransformer #sota
*status:* #📦 
*related:* 
- [[@borisovDeepNeuralNetworks2022]]
- [[@huangTabTransformerTabularData2020]]
- [[@arikTabNetAttentiveInterpretable2020]]
*code:* [https://github.com/Yura52/rtdl](https://github.com/Yura52/rtdl) (package + baseline)

## Notes Sebastian Raschka
-   In this paper, the researchers discuss the issue of improper baselines in the deep learning for tabular data literature.
-   The main contributions of this paper are centered around two strong baselines: one is a ResNet-like architecture, and the other is a transformer-based architecture called FT-Transformer (Feature Tokenizer + Transformer).
-   Across all 11 datasets considered in this study, the FT-Transformer outperforms other deep tabular methods in 6 cases and has the best overall rank. The most competitive deep tabular method is NODE, which outperforms other methods in 4 out of 11 cases.
-   In comparison with gradient boosted trees such as XGBoost and CatBoost, the FT-Transformer outperforms the former in 7 out of 11 cases; the authors conclude there is no universally superior method.

## Notes📍
- Authors propose two strong deep learning based baselines. One is a ResNet-like architecture (not covered) and another is a transformer-based architecture (similar to [[@huangTabTransformerTabularData2020]]). Their choice is motivated by the fact that both ResNet and Transformers are battle tested architectures from other fields. 
- From their comparsion between gradient boosting and deep learning approaches they conclude that there is no universal, superior approach.
- The use of deep learning for tabular data is desireable for two reasons: First, they could yield higher performance and second, neural nets could be used in multi-modal pipelines, that use both tabular data and other data sources, for which neural nets already dominate.
- **Need for baselines:** In absence of standardized baselines for tabular data. The authors urge for simple and reliable solutions, that are competive across different tasks. They conclude that a MLP is *the main* simple baseline, however it is not always challenging enough. 
- Setteling on the FT-Transformer and not their ResNet-like architecture can be motivated with their finding that FT-Transformer "... performs well on a wider rang eof tasks than the more conventinal ResNet and other DL models".
- For gradient boosting used as a reference here are several established libraries such as catboost, XGBoost and LightGBM, (that differ in e. g., the growing policy of trees, handling missing values or the calculation of gradients. (see papers also see [[@josseConsistencySupervisedLearning2020]]))  Their performance however, doesn't differ much. (found in [[@gorishniyRevisitingDeepLearning2021]] and cited [[@prokhorenkovaCatBoostUnbiasedBoosting2018]])

## Notation
>In this work, we consider supervised learning problems. $D=\left\{\left(x_i, y_i\right)\right\}_{i=1}^n$ denotes a dataset, where $x_i=\left(x_i^{(n u m)}, x_i^{(\text {cat })}\right) \in \mathbb{X}$ represents numerical $x_{i j}^{(n u m)}$ and categorical $x_{i j}^{(\text {cat })}$ features of an object and $y_i \in \mathbb{Y}$ denotes the corresponding object label. The total number of features is denoted as $k$. The dataset is split into three disjoint subsets: $D=D_{\text {train }} \cup D_{\text {val }} \cup D_{\text {test }}$, where $D_{\text {train }}$ is used for training, $D_{\text {val }}$ is used for early stopping and hyperparameter tuning, and $D_{\text {test }}$ is used for the final evaluation. We consider three types of tasks: binary classification $\mathbb{Y}=\{0,1\}$, multiclass classification $\mathbb{Y}=\{1, \ldots, C\}$ and regression $\mathbb{Y}=\mathbb{R}$.

I could adapt their notation for categorical and continous data as wall as their notation for the true label. Might refer continous variables as 'cont' instead of 'num' as it is more obvious and also used in the models.

## FT-Transformer 🦾
- Stands for feature tokenizer + transformer
- The model trasnforms both categorical and continous features to embeddings and applies as tack of transformer layers to the embeddings.  Thus every row in dataset becomes its own embedding and the transformer operates on the feature level of one object. The `[CLS]` token (final token in embedding) is used for prediction.  The feature tokenizer transforms features to embeddings, which are then processed by the transformer.
- As both categorical and continous features features are embedded, the architecture deviates from similar architectures like [[@huangTabTransformerTabularData2020]]. 
![[ft-transformer-architecture.png]]

![[comparison-ft-tab-transformer.png]]

### Feature Tokenizer
> The Feature Tokenizer module (see Figure 2) transforms the input features $x$ to embeddings $T \in \mathbb{R}^{k \times d}$. The embedding for a given feature $x_j$ is computed as follows:
$$
T_j=b_j+f_j\left(x_j\right) \in \mathbb{R}^d \quad f_j: \mathbb{X}_j \rightarrow \mathbb{R}^d .
$$
where $b_j$ is the $j$-th feature bias, $f_j^{(n u m)}$ is implemented as the element-wise multiplication with the vector $W_j^{(n u m)} \in \mathbb{R}^d$ and $f_j^{(c a t)}$ is implemented as the lookup table $W_j^{(c a t)} \in \mathbb{R}^{S_j \times d}$ for categorical features. Overall:
$$
\begin{array}{ll}
T_j^{(\text {num })}=b_j^{(\text {num })}+x_j^{(n u m)} \cdot W_j^{(n u m)} & \in \mathbb{R}^d, \\
T_j^{(c a t)}=b_j^{(\text {cat })}+e_j^T W_j^{(\text {cat })} & \in \mathbb{R}^d, \\
T=\operatorname{stack}\left[T_1^{(\text {num })}, \ldots, T_{\left.k^{(n u m)}\right)}^{(n u m)}, T_1^{(\text {cat })}, \ldots, T_{\left.k^{(c a t)}\right)}^{(\text {cat })}\right] & \in \mathbb{R}^{k \times d} .
\end{array}
$$
where $e_j^T$ is a one-hot vector for the corresponding categorical feature.

### Transformer
> At this stage, the embedding of the CLS token (or "classification token", or "output token", is appended to $T$ and $L$ Transformer layers $F_1, \ldots, F_L$, are applied:
$$
T_0=\operatorname{stack}[[\mathrm{CLS}], T] \quad T_i=F_i\left(T_{i-1}\right) .
$$
The final representation of the cls token is then used for prediction, using:
$$
\hat{y}=\operatorname{Linear}\left(\operatorname{ReLU}\left(\text { LayerNorm }\left(T_L^{[\mathrm{CLS}]}\right)\right)\right)
$$

Compared to the original architecture pre-Norm is used for easier optimization. Also the first normalization layer is removed to improve performance. 

## Complexity Considerations
- FT-Transformer is resource intensive with respect to hardware and time, which can render them illsuited for datasets with a large number of features. The computational complexity $\mathcal{O}^2$ is mainly driven by the quadratic complexity of vanilla multi-headed self attention. 
- There are however approaches to approximate multi-headed self-attention (see [[@tayEfficientTransformersSurvey2022]]). 

## Comparsion with TabTransformer (own thoughts)
* longer embeddings as numerical features are also embedded and probably higher computational complexity / no of parameters (?)
- Should be able to maintain correlations between continous and categorical data as both inputs are not processed separately / fed through different pieces of neural networks.
	
## Hyperparameter tuning
- For tuning they rely on Optuna, which uses the tree-structured parzen estimator algorithm. Intrestingly they set both a budget in terms of iterations and time.
- Intrestingly they perform models with **default** and **tuned hyperparameters.** Good sanity check to see if optimization went wrong and gives more insights to write about.

## Results
- FT-Transformer with default parameters mostly outperforms the ensembles of GBDT. With tuned hyperparameters however the DL models do no longer universally outperform GBDT. They conclude that the datasets are more dl-friendly as neural net based approaches achieve a better performance. 
- Hence they conclude that there is no universal solution among dl models and gbdts.
- As FT-Transformer is able to maintain a competitive performance across all tasks, they conclude that FT-Transformer could be a universal solution for tabular datasets.

## Ablation study
- They tune and evlauate the FT-Transformer without feature biases and with feature bases and average the results over 15 runs.

## Feature Importances
- They obtain feature importances using average activation maps from the transformer's forward pass. The results are then averaged from multiple samples to obtain a distribution. They also compare against other appraoches like feature permutation [[@breimanRandomForests2001]] and calculate the rank correlation.
> In this section, we evaluate attention maps as a source of information on feature importances for FT-Transformer for a given set of samples. For the $i$-th sample, we calculate the average attention map $p_i$ for the (CLS) token from Transformer's forward pass. Then, the obtained individual distributions are averaged into one distribution $p$ that represents the feature importances:
$$
p=\frac{1}{n_{\text {samples }}} \sum_i p_i \quad p_i=\frac{1}{n_{\text {heads }} \times L} \sum_{h, l} p_{\text {ihl }} .
$$
where $p_{i h l}$ is the $h$-th head's attention map for the (CLS) token from the forward pass of the $l$-th layer on the $i$-th sample. The main advantage of the described heuristic technique is its efficiency: it requires a single forward for one sample.

## Tabular format
Tabular format is characterized by: “In these problems, data points are represented as vectors of heterogeneous features, which is typical for industrial applications and ML competitions, where neural networks have a strong non-deep competitor in the form of GBDT (Chen and Guestrin, 2016; Ke et al., 2017; Prokhorenkova et al., 2018).” (Gorishniy et al., 2021, p. 1)

## Problems in research for tabular data
"The “shallow” state-of-the-art for problems with tabular data is currently ensembles of decision trees, such as GBDT (Gradient Boosting Decision Tree) (Friedman, 2001), which are typically the top-choice in various ML competitions."
“Along with potentially higher performance, using deep learning for tabular data is appealing as it would allow constructing multi-modal pipelines for problems, where only one part of the input is tabular, and other parts include images, audio and other DL-friendly data. Such pipelines can then be trained end-to-end by gradient optimization for all modalities.” (Gorishniy et al., 2021, p. 1)

“Additionally, despite the large number of novel architectures, the field still lacks simple and reliable solutions that allow achieving competitive performance with moderate effort and provide stable performance across many tasks.” (Gorishniy et al., 2021, p. 1)

## Common architectures

“Attention-based models. Due to the ubiquitous success of attention-based architectures for different domains (Dosovitskiy et al., 2021; Vaswani et al., 2017), several authors propose to employ attentionlike modules for tabular DL as well (Arik and Pfister, 2020; Huang et al., 2020; Song et al., 2019)” (Gorishniy et al., 2021, p. 2)


“our simple adaptation of the Transformer architecture (Vaswani et al., 2017) for tabular data.” (Gorishniy et al., 2021, p. 2)

## FT-Transformer

“Second, FT-Transformer demonstrates the best performance on most tasks and becomes a new powerful solution for the field. Interestingly, FT-Transformer turns out to be a more universal architecture for tabular data:” (Gorishniy et al., 2021, p. 2)

“Finally, we compare the best DL models to GBDT and conclude that there is still no universally superior solution.” (Gorishniy et al., 2021, p. 2)





## Notation for supervised learning problem

“Notation. In this work, we consider supervised learning problems. D={(xi, yi)}in=1 denotes a dataset, where xi=(x(num) i , x(cat) i ) ∈ X represents numerical x(num) ij and categorical x(cat) ij features of an object and yi ∈ Y denotes the corresponding object label. The total number of features is denoted as k. The dataset is split into three disjoint subsets: D = Dtrain ∪ Dval ∪ Dtest, where Dtrain is used for training, Dval is used for early stopping and hyperparameter tuning, and Dtest is used for the final evaluation. We consider three types of tasks: binary classification Y = {0, 1}, multiclass classification Y = {1, . . . , C} and regression Y = R.” (Gorishniy et al., 2021, p. 3)



## Notes from W&B Paper Reading Group
(See here: https://www.youtube.com/watch?v=59uGzJaVzYc)

- there is a lack of benchmarks
- deep learning models are interesting for multi-modal use cases
- feature tokenizer is just a look-up table as well
- distillation, learning rate warmup, learning rate decay is not used in paper,  but could improve training times and maybe accuracy.
- there is a paper that studies that studies ensembeling for deep learning (Fort et al, 2020)
- there is no universal solution of gbdt and deep learning models
- deep learning is less interpretable