In the following chapter, we motivate feature engineering, present our feature sets and discuss strategies for transforming features into a form that accelerates and advances the training of our models.

## Goal of feature engineering
Classical algorithms infer the initiator of the trade from the *raw* price and quote data. We employ feature engineering to pre-process input data and enhance the convergence and performance of our machine learning models. Gradient-boosted trees and neural networks, though flexible estimators, have limitations in synthesizing new features from existing ones, as demonstrated in empirical work on synthetic data by ([[@heatonEmpiricalAnalysisFeature2016]]5--6). Specifically, ratios, standard deviations, and differences can be difficult for these models to learn and must therefore be engineered beforehand. 

## Feature set definition
To establish a common ground, we derive three sets of features from raw data. The feature sets are are largely inspired by inherent features used within classical trade classification rules. Features are derived from quote and price data, except for feature sets 3 and 4, which include additional option characteristics and date features. 

Our first feature set uses price and quote data required by classical algorithms such the (reversed) Lee-and-Ready algorithm. We aid the models by estimating the change in trade price between the previous and successive distinguishable trades. This is similar to the criterion used in the (reverse) tick rule, but in an non-discretized fashion. 

Our second feature set extends the first feature set by the trade size and size of the quotes, required to estimate hybrid rules involving the depth rule and trade size rule.

Arguably, our models have simultaneous access to the previous and successive trade price and quotes for both the exchange and the NBBO, which is an advantage over base rules. As we benchmark against various, stacked hybrid classification rules, the data requirements are comparable. We emphasize this aspect, as previous 
 

Also, as data requirements 

Our largest feature set also incorporates option characteristics, including the strike price, the time to maturity, the moneyness, the option type and issue type as well as the underlying and traded volume of the option series. We hypthesize, that by

All feature set, the their definition and origin is documented in Appendix [[🍬appendix#^7c0162]].

Feature set {1} contains the price and time indicators necessary to compute the tick rule. Feature set {2} adds volume and lagged volume to {1}. These basic feature sets are composed of transaction level data included on every trade ticket (and their lags). ——————–Insert Table 4 here——————Feature sets {3} through {6} include {2} and other available TRACE transaction level data, FISD variables, and calculated metrics based on the TRACE data. Specifically, {3} includes the tick rule classifier, {4} incorporates {3} and a variety of running and rolling statistics, and {5} is a subset of {4} in which features unique to the TRACE dataset are removed.

## Problem of missing values and categoricals
The required pre-processing is minimal for tree-based learners. As one of few predictive models, trees can be extended to handle $\mathtt{[NaN]}$ values. Either by discarding missing values in the splitting procedure  (e. g.,[[@breimanClassificationRegressionTrees2017]] (p. 150 ff.)) or by incorporating missing values into the splitting criterion (e. g., [[@twalaGoodMethodsCoping2008]]) (p. 951). Recent literature for gradient boosting suggests, that handling missing data inside the algorithm slightly improves the accuracy over fitting trees on imputed data ([[@josseConsistencySupervisedLearning2020]] (p. 24) or [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]] (p. 6)). Also, some tree-based learners can handle categorical data without prior pre-processing, as shown in our chapter on ordered boosting ([[🐈Gradient Boosting]]).

However, neural networks can not inherently handle missing values, as a $\mathtt{[NaN]}$ value can not be propagated through the network. As such, missing values must be addressed beforehand. Similarily, categorical features, like the issue type, require an encoding, as no gradient can be calculated on categories.



## Solution to missing values and categoricals
In order to prepare a common datasets for *all* our machine learning models, we need to impute, scale and encode the data. Like in the chapter [[👨‍🍳Pre-Processing]] our feature scaling aims to be minimal intrusive, while facilitating efficient training for all our machine learning models. Following a common track in literature, we train our predictive model on imputed data.  We select an imputation with constants for being a single-pass strategy that minimizes data leakage and allows tree-based learners and neural networks to separate imputed values from observed ones. While imputation with constants is simplistic, it is on-par with more complex approaches (cp. [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]] p. 4). We choose a constant of $-1$, thus different from zero, so that the models can easily differentiate imputed from meaningful values and we avoid adversarial performance effects in neural networks from dropping input nodes (cp. [[@yiWhyNotUse2020]] p. 1 and [[@smiejaProcessingMissingData2018]]).[^5] No missing indicators are provided to keep the number of parameters in our models small.

As introduced in the chapters [[🐈Gradient Boosting]] and [[🤖Transformer]] both architectures have found to be robust to missing values. In conjunction with the low degree of missing values (compare chapter [[🚏Exploratory Data Analysis]]), we therefore expect the impact from missing values to be minor. To address concerns, that the imputation or scaling negatively impacts the performance of gradient boosted trees, we perform an ablation study in chapter [[🎋Ablation study]], and retrain our models on the unscaled and unimputed data set.

## Problem of feature scales
As observed in the [[🚏Exploratory Data Analysis]] data is not just missing but may also be skewed. Tree-based models can handle arbitrary feature scales, as the splitting process is based on the purity of the split but not on the scale of the splitting value.  

As we established in chapter ....

It has been well established that neural networks are long known to train faster on whitened data with zero mean, unit variance and uncorrelated inputs (cp. [[@lecunEfficientBackProp2012 1]]; p. 8). This is because a mean close to zero helps prevent  bias the direction of the weight update and scaling to unit variance helps balance the rate at which parameters are updated In order to maintain comparability with the traditional rules, inputs are not decorrelated. (reread in lecun paper or [here.](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/))

## Solution of feature scales

Continuous and categorical variable require different treatment, as derived below. Price and size-related features exhibit a positive skewness, as brought up in chapter [[🚏Exploratory Data Analysis]]. To avoid negative impacts during training (tails of distributions dominate calculations (see e. g. , [[@kuhnFeatureEngineeringSelection2020]] or https://deepai.org/machine-learning-glossary-and-terms/skewness), we reduce skewness with power transformations. We determine the transformation using the Box-Cox procedure ([[@boxAnalysisTransformations2022]]214), given by:
$$
\tilde{x}= \begin{cases}\frac{x^\lambda-1}{\lambda}, & \lambda \neq 0 \\ \log (x),& \lambda=0\end{cases}.\tag{1}
$$
Here, $\lambda$ is the power parameter and determines the specific power function. It is estimated by optimizing for the Gaussian likelihood on the training set. As shown in Equation $(1)$, a value of $\lambda=0$ corresponds to a log-transform, while $\lambda=1$ leaves the feature unaltered. As the test is only defined on positive $x$, we follow common practice by adding a constant of $1$ if needed.



When applying the test in feature engineering, it is important to note two major conceptual differences from the [[@boxAnalysisTransformations2022]] paper as pointed out by [[@kuhnFeatureEngineeringSelection2020]]. Here, the transform is used an unsupervised manner, as the transformation's outcome is not directly used in the model. Also, the transform is applied to all features, rather than the model's residuals.

Our estimates for $\lambda$ are documented in the Appendix [[🍬appendix]]. Based on the results of the box cox test, we apply a common $x^{\prime}=\log(x)$ transform with the effect of compressing large values and expanding smaller ones. (footnote More specifically, $x^{\prime}= \log(x+1)$ is used to prevent taking the logarithm of zero and improving numerical stability in floating point calculations[^1].) Due to the monotonous nature of the logarithm (power transform in general; see https://en.wikipedia.org/wiki/Power_transform), the splits of tree-based learners remain unaffected with only minor differences due to quantization <mark style="background: #ADCCFFA6;">(see quantization / histogram building in gradient boosting https://neurips.cc/media/neurips-2022/Slides/53370.pdf)</mark>. The log transform comes at the cost of an decreased interpretability (cp. [[@fengLogtransformationItsImplications2014]]).


To address the problem of convergence of our transformer-based architectures, we normalize the data set using $z$-score normalization given by formula [[#^5d5445]]:
$$
x^{\prime}=\frac{x-\mu}{\sigma}\tag{1}
$$
with mean
$$
\mu=\frac{1}{N} \sum_{i=1}^N\left(x_i\right),
$$
and standard deviation
$$
\sigma=\sqrt{\frac{1}{N} \sum_{i=1}^N\left(x_i-\mu\right)^2}.
$$
^5d5445
Following good measures, all statistics are estimated on the training set only.

Normalization has the advantage of preserving the data distribution, as shown by [[@kuhnFeatureEngineeringSelection2020]], which is an important property when comparing[[🏅Feature importance results]] based models against their classical counterparts in chapter [[🏅Feature importance measure]] . [^4]

As for the categorical variables a transformation is required. We perform a label encoding by randomly mapping every unique value onto an integer key. As an example, the option type in the set $\{\text{'C'},\text{'P'}\}$ would be randomly mapped onto $\{1,0\}$. This basic transformation allows to defer handling of categorical data to the model ([[@hancockSurveyCategoricalData2020]]10). Also, it minimizes target leakage. Classes not seen during are mapped to the key of an $\mathtt{[UNK]}$ token, as motivated in cref-[[💤Embeddings For Tabular Data]]. 

(Graphics left distribution of underlyings / right 10 most frequent underlyings)
Due to the high cardinality of the root, the 

One aspect that remains open, is the high cardinality of categorical features with as many as (add number of classes of root) classes. We postpone strategies to model-specific treatments in chapter [[💡Training of models (supervised)]]. The chapter also provides further insights on handling categories observed during training.

Known disadvantages of label encoding, as raised in ([[@hancockSurveyCategoricalData2020]]12), such as the unequal contributions of larger keys to the loss in neural networks or the artificially implied order, do not apply here, as the conversion is followed by sophisticated treatments within the models. We refer to cref-[[🤖TabTransformer]] and [[🤖FTTransformer]] for in-depth coverage in Transfomers and cref-[[🐈Gradient Boosting]] for ordered boosting.

A comprehensive overview of all feature transformations is given in Appendix [[🍬appendix#^8e998b]].

[^1]: See e. g., https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
[^2]: See chapter on ordered boosting, [[🤖TabTransformer]], or the [[🤖FTTransformer]] .
[^3]: Notice the similarities to the positional encoding used in [[@vaswaniAttentionAllYou2017]].
[^4]: Optionally, add proof in the appendix.
[^5]: Subsequent scaling may also affect the imputation constant.
