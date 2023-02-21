At times we fall back to the Transformer for machine translations, to develop a deeper understanding of the architecture and its components.



# Feature Sets

| Feature               | Feature Category             | Why?                                                                                                                        | FS 1 (Classical) | FS 2 (F1 + Grauer) | FS 4 (F3 + Others) | Transform   |
|-----------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------|--------------------|--------------------|-------------|
| TRADE_PRICE           | tick rule                    | See [[@leeInferringTradeDirection1991]]                                                                                     | x                | x                  | x                  | log         |
| price_ex_lag          | tick rule                    | See above.                                                                                                                  | x                | x                  | x                  | log         |
| price_all_lag         | tick rule                    | See above.                                                                                                                  | x                | x                  | x                  | log         |
| chg_ex_lag            | tick rule                    | See above.                                                                                                                  | x                | x                  | x                  | standardize |
| chg_all_lag           | tick rule                    | See above.                                                                                                                  | x                | x                  | x                  | standardize |
| price_ex_lead         | reverse tick rule            | See above.                                                                                                                  | x                | x                  | x                  | log         |
| price_all_lead        | reverse tick rule            | See above.                                                                                                                  | x                | x                  | x                  | log         |
| chg_ex_lead           | reverse tick rule            | See above.                                                                                                                  | x                | x                  | x                  | standardize |
| chg_all_lead          | reverse tick rule            | See above.                                                                                                                  | x                | x                  | x                  | standardize |
| BEST_BID              | quote rule                   | See above.                                                                                                                  | x                | x                  | x                  | log         |
| bid_ex                | quote rule                   | See above.                                                                                                                  | x                | x                  | x                  | log         |
| BEST_ASK              | quote rule                   | See above.                                                                                                                  | x                | x                  | x                  | log         |
| mid_ex                | mid quote 🆕                  | See above.                                                                                                                  |                  |                    |                    | log         |
| mid_best              | mid quote 🆕                  | See above.                                                                                                                  |                  |                    |                    | log         |
| ask_ex                | quote rule                   | See [[@leeInferringTradeDirection1991]]                                                                                     | x                | x                  | x                  | log         |
| bid_ask_ratio_ex      | Ratio of ask and bid 🆕       | ?                                                                                                                           |                  | x                  | x                  | standardize |
| spread_ex             | Absolute spread 🆕            | ?                                                                                                                           |                  |                    |                    | standardize |
| spread_best           | Absolute spread 🆕            | ?                                                                                                                           |                  |                    |                    | standardize |
| price_rel_nbb         | Tradeprice rel to nbb 🆕      | Relates trade exchange with nation-wide best.                                                                               |                  | x                  | x                  | standardize |
| price_rel_nbo         | Tradeprice rel to nbo 🆕      | See above.                                                                                                                  |                  | x                  | x                  | standardize |
| prox_ex               | EMO / CLNV                   | Most important predictor in [[@ellisAccuracyTradeClassification2000]] and [[@chakrabartyTradeClassificationAlgorithms2012]] | x                | x                  | x                  | standardize |
| prox_best             | EMO / CLNV                   | See above.                                                                                                                  | x                | x                  | x                  | standardize |
| bid_ask_size_ratio_ex | Depth rule                   | See [[@grauerOptionTradeClassification2022]]                                                                                |                  | x                  | x                  | standardize |
| bid_size_ex           | Depth rule / Trade size rule | See above.                                                                                                                  |                  | x                  | x                  | standardize |
| ask_size_ex           | Depth rule / Trade size rule | See above.                                                                                                                  |                  | x                  | x                  | standardize |
| rel_bid_size_ex       | Trade size rule              | See above.                                                                                                                  |                  | x                  | x                  | standardize |
| rel_ask_size_ex       | Trade size rule              | See above.                                                                                                                  |                  | x                  | x                  | standardize |
| TRADE_SIZE            | Trade size rule              | See above.                                                                                                                  |                  | x                  | x                  | standardize |
| STR_PRC               | option                       | ?                                                                                                                           |                  |                    | x                  | log         |
| day_vol               | option                       | ?                                                                                                                           |                  |                    | x                  | log         |
| bin_root              | option 🦺(many `UNKWN`)       | ?                                                                                                                           |                  |                    | x                  | binarize    |
| time_to_maturity      | option                       | ?                                                                                                                           |                  |                    | x                  | standardize |
| moneyness             | option                       | ?                                                                                                                           |                  |                    | x                  | standardize |
| bin_option_type       | option                       | ?                                                                                                                           |                  |                    | x                  | binarize    |
| bin_issue_type        | option                       | See [[@ronenMachineLearningTrade2022]]. Learn temporal patterns. Data is ordered by time.                                   |                  |                    | x                  | binarize    |
| date_month_sin        | date                         | See above.                                                                                                                  |                  |                    | x                  | pos enc     |
| date_month_cos        | date                         | See above.                                                                                                                  |                  |                    | x                  | pos enc     |
| date_day_sin          | date                         | See above.                                                                                                                  |                  |                    | x                  | pos enc     |
| date_day_cos          | date                         | See above.                                                                                                                  |                  |                    | x                  | pos enc     |
| date_weekday_sin      | date                         | See above.                                                                                                                  |                  |                    | x                  | pos enc     |
| date_weekday_cos      | date                         | See above.                                                                                                                  |                  |                    | x                  | pos enc     |
| date_time_sin         | date                         | See above.                                                                                                                  |                  |                    | x                  | pos enc     |
| date_time_cos         | date                         | See above.                                                                                                                  |                  |                    | x                  | pos enc     |
| date_year             | date 🦺(uniformative)         | See above.                                                                                                                  |                  |                    |                    | None        |

## LR algorithm
The algorithm is derived from an analysis of stock trades inside the quotes ([[@leeInferringTradeDirection1991]] 742). 

## Residual Connections


## Positional Embedding
Positional embeddings are not the only way to fix the location, however. Later works, like ([[@daiTransformerXLAttentiveLanguage2019]]4--5), remove the positional encoding in favour of a *relative position encoding*, which is only considered during computation.


## Point-wise FFN
and ([[@gevaTransformerFeedForwardLayers2021]]).

Later variants (see e. g., [[@devlinBERTPretrainingDeep2019]] or [[@radfordImprovingLanguageUnderstanding]]) commonly replace the $\operatorname{ReLU}$ with the *Gaussian Error Linear Units* $\operatorname{GELU}$ ([[@hendrycksGaussianErrorLinear2020]], p. 2) activation, which has empirically proven to improve the performance and convergence behaviour of Transformers ([[@narangTransformerModificationsTransfer2021]], p. 16; and [[@shazeerGLUVariantsImprove2020]] p. 4).

Like the [[🅰️Attention]] sub-layer, the feed-forward sub-layer is surrounded by residual connections ([[@heDeepResidualLearning2015]]) and followed by a layer-normalization ([[@baLayerNormalization2016]] (p. 4)) layer. 


### Normality

Test log-normality visually with qq-plots (https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy) or using statistical tests e. g.,  log-transform + normality test. https://stats.stackexchange.com/questions/134924/tests-for-lognormal-distribution

<mark style="background: #FFB8EBA6;">- min-max scaling and $z$ scaling preserve the distribution of the variables  (see [here.](https://stats.stackexchange.com/a/562204/351242)). Applying both cancels out each other (see proof [here.](https://stats.stackexchange.com/a/562204/351242)). </mark>

<mark style="background: #FF5582A6;">There are controversies(Note zero imputation can be problematic for neural nets, as shown in [[@yiWhyNotUse2020]] paper)</mark>
<mark style="background: #FF5582A6;">- For imputation look into [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]]
- [[@josseConsistencySupervisedLearning2020]] also compare different imputation methods and handling approaches of missing values in tree-based methods.
- for visualizations and approaches see [[@zhengFeatureEngineeringMachine]] and [[@butcherFeatureEngineeringSelection2020]]</mark>
<mark style="background: #FF5582A6;">- [[@yiWhyNotUse2020]] and [[@smiejaProcessingMissingData2018]] contain various references to papers to impute missing data in neural networks. 
- add no missing indicator to keep the number of parameters small.
</mark>
<mark style="background: #BBFABBA6;">- [[@lemorvanWhatGoodImputation2021]] for theoretical work on imputation.
- For patterns and analysis of imputed data see https://stefvanbuuren.name/fimd/ch-analysis.html</mark>



 %%we normalize all continous features into a range of $[-1,1]$ using formula [[#^5d5445]]:

$$
x^{\prime}=-1+\frac{2(x-\min (x))}{\max (x)-\min (x)} \tag{1}
$$
$$
X_{n o r m}=\frac{X-X_{\min }}{X_{\max }-X_{\min }}
$$

%%


- TODO: Why do we perform feature scaling at all?
- TODO: Try out robust scaler, as data contains outliers. Robust scaler uses the median which is robust to outliers and iqr for scaling. 
- TODO: Try out different IQR thresholds and report impact. Similarily done here: https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/
- We scale / normalize features to a $\left[-1,1\right]$  scale using statistics estimated on the training set to avoid data leakage. This is also recommended in [[@huyenDesigningMachineLearning]]. Interestingly, she also writes that empirically the interval $\left[-1,1\right]$ works better than $\left[0,1\right]$. Also read about this on stackoverflow for neural networks, which has to do with gradient calculation.
- Scale to an arbitrary range $\left[a,b\right]$ using the formula from [[@huyenDesigningMachineLearning]]:
$$
x^{\prime}=a+\frac{(x-\min (x))(b-a)}{\max (x)-\min (x)}
$$
- Feature scaling theoretically shouldn't be relevant for gradient boosting due to the way gbms select split points / not based on distributions. Also in my tests it didn't make much of a difference for gbms but for transformers. (see https://github.com/KarelZe/thesis/blob/main/notebooks/3.0b-mb-comparsion-transformations.ipynb) 
- [[@ronenMachineLearningTrade2022]] performed no feature scaling.
- [[@borisovDeepNeuralNetworks2022]] standardize numerical features and apply ordinal encoding to categorical features, but pass to the model which ones are categorical features. 
- [[@gorishniyRevisitingDeepLearning2021]] (p. 6) use quantile transformation, which is similar to the robust scaler, see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-pyf) Note that [[@grinsztajnWhyTreebasedModels2022]] only applied quantile transformations to all features, thus not utilize special implementations for categorical variables.

# Random Forests

As stated previously, decision trees suffer from a number of drawbacks. One of them being prone to overfitting. To mitigate the high variance, several trees can be combined to form an *ensemble* of trees. The final prediction is then jointly estimated from all member within the ensemble.

Popular ensemble methods for trees include bagging and random forests [[@breimanRandomForests2001]]. Also, Boosting is another approach that learns a sequence base learners such as simplified decision trees. [[@hastietrevorElementsStatisticalLearning2009]] (p. 587) We present boosting as part of section [[🐈Gradient Boosting]].

Bagged trees and Random Forests have in common to learn $B$ independent trees. However, they differ in whether learning is performed on a random subset of data or whether splits consider only a portion of all features.

Let's consider the regression case only. With bagging each tree is trained on a random subset of data, drawn with replacement from the training set. Learning a predictor on a so-called bootstraped sample, still causes the single tree to overfit. Especially, if trees are deep. Pruning the trees, selecting the best performing ones and averaging their estimates to a bagged predictor, helps to improve the accuracy [[@breimanBaggingPredictors1996]].

Besides this, averaging the estimates of several trees, bagging maintaines the desirable low-bias property of a single tree, assuming trees are grown large enough to capture subtleties of the data, while also improving on the variance. [[@hastietrevorElementsStatisticalLearning2009]]

Yet, one issue bagging can not resolve is, that bagged trees are not independent [[@hastietrevorElementsStatisticalLearning2009]]. This is due to the limitation, that all trees select their best split attributes from the same set of features. If features dominate in the bootstrap samples, they will yield similiar splitting sequences and thus highly correlated trees.

The variant of Bagging named *Random Forests* addresses the high correlation among trees, by considering only a random subset of all features for splitting. Random forests for regression, as introduced by [[@breimanRandomForests2001]]  consist of $B$ trees, that are grown in parallel to form a forest. At each split only a random subset of $m$ features is considered for splitting. Typically, $m$ is chosen to be the $\sqrt{p}$ of all $p$ input variables [[@hastietrevorElementsStatisticalLearning2009]].

The random forest predictor is then estimated as the average overall the set of $\left\{T\left(x ; \Theta_{b}\right)\right\}_{1}^{B}$  trees:
$$
\hat{f}_{\mathrm{rf}}^{B}(x)=\frac{1}{B} \sum_{b=1}^{B} T\left(x ; \Theta_{b}\right),
$$

with $\Theta_{b}$ being a parameter vector of the  $b$-th tree [[@hastietrevorElementsStatisticalLearning2009]].

As the variables considered for splitting differ from one split and one tree to another, the trees are less similar and hence correlated. Random Forests achieve a comparable accuracy to Boosting or even outperform them. As trees do not depend on previously built trees, they can be trained in parallel. These advantages come at the cost of of lower interpretability compared to decision trees. [[@breimanRandomForests2001]]

In the next section we discuss Boosting approaches, that grow trees in an adaptive manner.




A FFN tries to approximate an arbitrary function $f^{*}$. To do so, it defines a mapping $\boldsymbol{y}=f(\boldsymbol{x} ; \boldsymbol{\theta})$ from some input $\boldsymbol{x}$ to some output $\boldsymbol{y}$ and learns the parameters $\boldsymbol{\theta}$, that approximate the true output best.

Structurally, a FFN consists of an input layer, one or more hidden layer and output layer. Thereby, each layer is made up of neurons and relies on input from the previous layer. In the most trivial case, the network consists of only a single hidden layer and the output layer. Formally, the output is calculated as shown in equation (...). $\mathbf{X} \in \mathbb{R}^{n \times d}$ denotes the input consisting of $d$ features and $n$ samples, $\mathbf{H} \in \mathbb{R}^{n \times h}$  the output of the hidden layer with $h$ hidden units and $\mathbf{O} \in \mathbb{R}^{n \times q}$ the final output. The  weights and bias for the hidden layer and output layer are denoted by $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ and biases $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$ and output-layer weights $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ and biases $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q} .$

$$
\begin{aligned} \mathbf{H} &=\sigma\left(\mathbf{X} \mathbf{W}^{(1)}+\mathbf{b}^{(1)}\right) \\ \mathbf{O} &=\mathbf{H} \mathbf{W}^{(2)}+\mathbf{b}^{(2)} \end{aligned}
$$
As seen above, an affine transformation is applied to the input, followed activation function $\sigma(\cdot)$, that decides whether a neuron in the hidden layer is activated. The final prediction is then obtained  after another affine transformation the output layer. Here, the parameter set consists of $\boldsymbol{\theta} = \left \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)},\mathbf{W}^{(2)}, \mathbf{b}^{(2)} \right\}$.

To learn the function approximation, FFNs are trained using backpropagation by adjusting the parameters $\boldsymbol{\theta}$ of each layer to minimize a loss function $\mathcal{L}(\cdot)$. As backpropagation requires the calculation of the gradient, both the activation and loss functions have to be differentiable.

ReLU is a common choice. It's non-linear and defined as the element-wise maximum between the input $\boldsymbol{x}$ and $0$:

$$
\operatorname{ReLU}(\boldsymbol{x})=\max (\boldsymbol{x}, 0).
$$

The usage of ReLU as activation function is desirable for a number of reasons. First, it can be computated efficiently as no exponential function is required. Secondly, it solves the vanishing gradient problem present in other activation functions [[@glorotDeepSparseRectifier2011]].

Networks with a single hidden layer can approximate any arbitrary function given enough data and network capacity [[@hornikMultilayerFeedforwardNetworks1989]].  
In practice, similiar effects can be achieved by stacking several hidden layers and thereby deepening the network, while being more compact [[@zhangDiveDeepLearning2021]].

Deep neural nets combine several hidden layers by feeding the previous hidden layer's output into the subsequent hidden layer. Assuming a $\operatorname{ReLU}(\cdot)$ activation function, the stacking for a network with two hidden layers can be formalized as: $\boldsymbol{H}^{(1)}=\operatorname{ReLU}_{1}\left(\boldsymbol{X W}^{(1)}+\boldsymbol{b}^{(1)}\right)$ and $\boldsymbol{H}^{(2)}=\operatorname{ReLU}_{2}\left(\mathbf{H}^{(1)} \mathbf{W}^{(2)}+\mathbf{b}^{(2)}\right)$.

Feed forward networks are restricted to information flowing through the network in a forward manner. To also incorporate feedback from the output, we introduce Recursive Neural Nets as part of section (...).