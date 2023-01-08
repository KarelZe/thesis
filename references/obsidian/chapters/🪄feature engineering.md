In the following chapter, we motivate feature engineering, present our feature sets and discuss strategies for transforming features into a form that accelerates and advances the training of our models.

## Goal of feature engineering
Classical algorithms ([[basic_rules]] or [[hybrid_rules]]) infer the initiator of the trade from the *raw* price and quote data. We employ feature engineering to pre-process input data and enhance the convergence and performance of our machine learning models. Gradient-boosted trees and neural networks, though flexible estimators, have limitations in synthesizing new features from existing ones, as demonstrated in empirical work on synthetic data by [@heatonEmpiricalAnalysisFeature2016] (p. 5-6). Specifically, ratios, standard deviations, and differences can be difficult for these models to learn and must therefore be engineered beforehand. To ensure fair comparison and improve the transferability of our results, we use simple, non-destructive transformations in our feature engineering. 

## Feature set definition
To establish a common ground, we derive three sets of features from raw data. The feature sets are are to a large extend inspired by inherent features used within classical trade classification rules. Features are derived from quote and price data, except for feature sets 3 and 4, which include additional option characteristics and date features. 

All feature set, the their definition and origin is documented in Appendix [[🍬appendix#^7c0162]].

## Problem of missing values and categoricals
The required pre-processing is minimal for tree-based learners. As one of few predictive models, trees can be extended to handle $\mathtt{[NaN]}$ values. Either by discarding missing values in the splitting procedure  (e. g.,[[@breimanClassificationRegressionTrees2017]] (p. 150 ff.)) or by incorporating missing values into the splitting criterion (e. g., [[@twalaGoodMethodsCoping2008]]) (p. 951). Recent literature for gradient boosting suggests, that handling missing data inside the algorithm slightly improves the accuracy over fitting trees on imputed data ([[@josseConsistencySupervisedLearning2020]] (p. 24) or [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]] (p. 6)). Also, some tree-based learners can handle categorical data without prior pre-processing, as shown in our chapter on ordered boosting ([[🐈gradient-boosting]]).

However, neural networks can not inherently handle missing values, as a $\mathtt{[NaN]}$ value can not be propagated through the network. As such, missing values must be addressed beforehand. Similarily, categorical features, like the issue type, require an encoding, as no gradient can be calculated on categories.

## Solution to missing values and categoricals
In order to prepare a common datasets for *all* our machine learning models, we need to impute, scale and encode the data. Like in the chapter [[preprocessing]] our feature scaling aims to be minimal intrusive, while facilitating efficient training for all our machine learning models. Following a common track in literature, we train our predictive model on imputed data.  We select an imputation with constants for being a single-pass strategy that minimizes data leakage and allows tree-based learners and neural networks to separate imputed values from observed ones. While imputation with constants is simplistic, it is on-par with more complex approaches (cp. [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]] p. 4). We choose a constant of $-1$, thus different from zero, so that the models can easily differentiate imputed from meaningful values and we avoid adversarial performance effects in neural networks from dropping input nodes (cp. [[@yiWhyNotUse2020]] p. 1 and [[@smiejaProcessingMissingData2018]]).[^5] No missing indicators are provided to keep the number of parameters in our models small.

Classical trade signing algorithms, such as the tick test, are also impacted by missing values. In theses cases, we defer to a random classification or a subsequent rule, if rules can not be computed. Details are provided in section [[training-of-supervised-models]].

As introduced in the chapters [[🐈gradient-boosting]] and [[🤖transformer]] both architectures have found to be robust to missing values. In conjunction with the low degree of missing values (compare chapter [[🚏exploratory data analysis]]), we therefore expect the impact from missing values to be minor. To address concerns, that the imputation or scaling negatively impacts the performance of gradient boosted trees, we perform an ablation study in chapter [[🎋ablation_study]], and retrain our models on the unscaled and unimputed data set.

## Problem of feature scales
As observed in the [[🚏exploratory data analysis]] data is not just missing but may also be skewed. Tree-based models can handle arbitrary feature scales, as the splitting process is based on the purity of the split but not on the scale of the splitting value.  

It has been well established that neural networks are long known to train faster on whitened data with zero mean, unit variance and uncorrelated inputs (cp. [[@lecunEfficientBackProp2012]]; p. 8). This is because a mean close to zero helps prevent  bias the direction of the weight update and scaling to unit variance helps balance the rate at which parameters are updated In order to maintain comparability with the traditional rules, inputs are not decorrelated. (reread in lecun paper or [here.](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/))

## Solution of feature scales

Continuous and categorical variable require different treatment, as derived below. Price and size-related features exhibit a positive skewness, as brought up in chapter [[🚏exploratory data analysis]]. To avoid negative impacts during training (tails of distributions dominate calculations (see e. g. , [[@kuhnFeatureEngineeringSelection2020]] or https://deepai.org/machine-learning-glossary-and-terms/skewness), we reduce skewness with power transformations. We determine the transformation using the Box-Cox procedure ([[@boxAnalysisTransformations2022]] p. 214), given by:
$$
\tilde{x}= \begin{cases}\frac{x^\lambda-1}{\lambda}, & \lambda \neq 0 \\ \log (x),& \lambda=0\end{cases}.\tag{1}
$$
Here, $\lambda$ is the power parameter and determines the specific power function. It is estimated by optimizing for the Gaussian likelihood on the training set. As shown in Equation $(1)$, a value of $\lambda=0$ corresponds to a log-transform, while $\lambda=1$ leaves the feature unaltered. As the test is only defined on positive $x$, we follow common practice by adding a constant of $1$ if needed. 

When applying the test in feature engineering, it is important to note two major conceptual differences from the [[@boxAnalysisTransformations2022]] paper as pointed out by [[@kuhnFeatureEngineeringSelection2020]]. Here, the transform is used an unsupervised manner, as the transformation's outcome is not directly used in the model. Also, the transform is applied to all features, rather than the model's residuals.

Our estimates for $\lambda$ are documented in the Appendix [[🍬appendix]]. Based on the results of the box cox test, we apply a common $x^{\prime}=\log(x)$ transform with the effect of compressing large values and expanding smaller ones. More specifically, $x^{\prime}= \log(x+1)$ is used to prevent taking the logarithm of zero and improving numerical stability in floating point calculations[^1]. Due to the monotonous nature of the logarithm (power transform in general; see https://en.wikipedia.org/wiki/Power_transform), the splits of tree-based learners remain unaffected with only minor differences due to quantization <mark style="background: #ADCCFFA6;">(see quantization / histogram building in gradient boosting https://neurips.cc/media/neurips-2022/Slides/53370.pdf)</mark>. The log transform comes at the cost of an decreased interpretability (cp. [[@fengLogtransformationItsImplications2014]]).

Our largest feature set als contains dates and times of the trade. In contrast to other continuous features, the features are inherently cyclic. We exploit this property for hours, days, and months and apply a fourier transform to convert the features into a smooth variable using formula [[#^773161]]:

$$
\begin{aligned}
x_{\sin} &= \sin\left(\frac{2\pi x}{\max(x)} \right), \text{and}\\
c_{\cos} &= \cos\left(\frac{2\pi x}{\max(x)} \right),
\end{aligned}
$$
^773161
where $x$ is the raw input and $x_{\sin}$ and $x_{\cos}$ are the cyclical features. This cyclic continuous encoding, has the effect of preserving temporal proximity, as shown in Figure [[#^278944]]. As visualized for dates, the month's ultimo and the next month's first are close to each other in the individual features and on the unit circle. [^3]

![[positional_encoding.png]]
(found here similarly: https://www.researchgate.net/figure/A-unit-circle-example-of-frequency-encoding-of-spatial-data-using-the-Fourier-series-a_fig2_313829438) ^278944

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
Following good measusre, all statistics are estimated on the training set only.

Normalization has the advantage of preserving the data distribution, as shown by [[@kuhnFeatureEngineeringSelection2020]], which is an important property when comparing our machine learning based models against their classical counterparts in chapter [[feature_importance]]. [^4]

As for the categorical variables a transformation is required. We perform a label encoding by randomly mapping every unique value onto an integer key. As an example, the option type in the set $\{\text{'C'},\text{'P'}\}$ would be randomly mapped onto $\{1,0\}$. This basic transformation allows to defer handling of categorical data to the model [[@hancockSurveyCategoricalData2020]] (p. 10). Also, it minimizes target leakage. Classes not seen during are mapped to the key of a $\mathtt{[UNK]}$ token.

Known disadvantages of label encoding, as raised in [[@hancockSurveyCategoricalData2020]] (p. 12), such as the unequal contributions of larger keys to the loss in neural networks or the artificially implied order, do not apply here, as the conversion is followed by sophisticated treatments within the models[^2].

One aspect that remains open, is the high cardinality of categorical features with as many as (add number of classes of root) classes. We postpone strategies to model-specific treatments in chapter [[training-of-supervised-models]]. The chapter also provides further insights on handling categories observed during training.

A comprehensive overview of all feature transformations is given in Appendix [[🍬appendix#^8e998b]].

[^1]: See e. g., https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
[^2]: See chapter on ordered boosting, [[extensions-to-tabtransformer]], or the [[fttransformer]].
[^3]: Notice the similarities to the positional encoding used in [[@vaswaniAttentionAllYou2017]].
[^4]: Optionally, add proof in the appendix.
[^5]: Subsequent scaling may also affect the imputation constant.
