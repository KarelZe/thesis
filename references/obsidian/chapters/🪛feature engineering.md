In the following chapter, we motivate feature engineering, present our feature sets and discuss strategies for transforming features into a form that accelerates and advances the training of our models.

## Goal of feature engineering
Classical algorithms ([[basic_rules]] or [[hybrid_rules]]) infer the initiator of the trade from price and quote data. ~~To establish a common ground, we derive three sets of features from raw data. ~~

Instead of learning our models on unprocessed data directly, we perform *feature engineering* to convert input data into a form that can be digested by the models and expectedly improves their convergence behaviour. Feature engineering is also contributing to the model's performance. ~~, as it helps to unravel relationships between the predictors and the response variable [[@kuhnFeatureEngineeringSelection2020]] (xi).~~ Despite gradient-boosted trees and neural networks being flexible estimators, the empirical work of [[@heatonEmpiricalAnalysisFeature2016]] (p. 5-6) reveals limitations with regard to their synthesizing capabilities of engineered features. In particular, ratios, standard deviations and differences for gradient boosting as well as differences for neural networks and must therefore be engineered beforehand. This underpins the necessity for feature engineering.

While feature engineering aids our model's performance we restrict ourself to simple, non-distructive transformations. This allows for a fair comparsion between the algorithms and improves the transferability of our results. Ultimately, all features are derived from quote and price data, used inherently in the classical algorithms, as shown in the next section. The only exception is feature set 3 (see [[🧃Feature Sets]]), that adds additional option characteristics and date features. 

## Feature set definition

The featue sets are are to a large extend inspired by the decision rules used in ...

<mark style="background: #ABF7F7A6;">TODO: Add to each feature, where it has been used.
TODO: Point out some interesting features here in the text.
TODO: We are less concerned about providing redundant data to the model, as trees do not base their splitting process on correlation. Also, with neural nets being a universal function approximator.
TODO: explain why we don't discretize / binarize features, other than classical models. See [[@kuhnFeatureEngineeringSelection2020]]
TODO: Explain why we don't include the results of the classical rules themselves, due to redundant low-precision encoding. May give an example explaining for decision trees.
TODO: Adress the usefullness of the engineered features. Sketch the results of the *adversarial validation*. Which features are very different in the training set and the validation set? Which features are most important in adversarial validation?
TODO: Plot distributions of features from training and validation set. Could also test using https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test test if samples are drawn from the same distribution.</mark>

All feature set, the their definition and origin is documented in Appendix [[🍬appendix#^7c0162]].

## Problem of missing values and categoricals
The required pre-processing is minimal for tree-based learners. Missing values can be handled by sending down `[NaN]` values at one side of the tree. Recent literature indicates that handling missing data inside the algorithm slightly improves over <mark style="background: #ABF7F7A6;">... simpler approaches over .</mark> <mark style="background: #FF5582A6;">See [[@breimanClassificationRegressionTrees2017]] on surrogate splits</mark>. Also, some tree-based learners can handle categorical data without prior pre-processing, as shown in our chapter on ordered boosting ([[🐈gradient-boosting]]).

Neural networks can not inherently handle missing values, as a $\mathtt{[NaN]}$ value can not be propagated through the network. As such, missing values must be addressed beforehand. Similarily, categorical features, like the issue type, require an encoding, as no gradient can be calculated on categories.

## Solution to missing values and categoricals
In order to prepare a common datasets for *all* our models, we need to impute, scale and encode the data. Like in the chapter [[preprocessing]] our feature scaling aims to be minimal intrusive, while facilitating efficient training for all our machine learning models.

Missing values are imputed with zeros. This simple, one-pass  strategy ~~minimizes the bias from imputation~~, avoids data leakage, and allows tree-based learners and neural networks to separate imputed values from observed ones. While the imputation with constants is simple, it is on-par with more complex approaches as <mark style="background: #FF5582A6;">(...)</mark> while minimizing the bias from imputatation. <mark style="background: #FF5582A6;">There are controversies(Note zero imputation can be problematic for neural nets, as shown in [[@yiWhyNotUse2020]] paper)</mark>
<mark style="background: #FF5582A6;">- For imputation look into [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]]
- [[@josseConsistencySupervisedLearning2020]] also compare different imputation methods and handling approaches of missing values in tree-based methods.
- for visualizations and approaches see [[@zhengFeatureEngineeringMachine]] and [[@butcherFeatureEngineeringSelection2020]]</mark>
<mark style="background: #FF5582A6;">- [[@yiWhyNotUse2020]] and [[@smiejaProcessingMissingData2018]] contain various references to papers to impute missing data in neural networks. 
- add no missing indicator to keep the number of parameters small.
</mark>
<mark style="background: #BBFABBA6;">- [[@lemorvanWhatGoodImputation2021]] for theoretical work on imputation.
- For patterns and analysis of imputed data see https://stefvanbuuren.name/fimd/ch-analysis.html</mark>
- We impute prior to scaling, as the imputation can affect the statistics
- normalization scales, but does not change the distribution. See. e. g., [[@kuhnFeatureEngineeringSelection2020]]

As introduced in the chapters [[🐈gradient-boosting]] and [[🤖transformer]] both architectures have found to be robust to missing values. 

In conjunction with the low degree of missing values (compare chapter [[🌴exploratory_data_analysis]]), we therefore expect the impact from missing values to be minor. To address concerns, that the imputation or scaling negatively impacts the performance of gradient boosted trees, we perform an ablation study in chapter [[🎋ablation_study]], and retrain our models on the unscaled and unimputed data set.

## Problem of feature scales

Tree-based models can handle arbitrary feature scales, as the splitting process is based on the purity of the split but not on the scale of the splitting value. 

<mark style="background: #FFB86CA6;">
Also, neural networks are known to train faster, .... normalized / standardized data . problems with convergence etc. Motivation for scaling features to $[-1,1]$ range or zero mean. https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used -></mark> Also see [[@kuhnFeatureEngineeringSelection2020]]

## Solution of feature scales

Continous and categorical variable require different treatment, as derived below. 

Price and size-related features exhibit a positive skewness, as brought up in chapter [[🌴exploratory_data_analysis]].

<mark style="background: #D2B3FFA6;">A Box-Cox transformation (Box and Cox, 1964) was used to estimate this transformation. The Box-Cox procedure, originally intended as a transformation of a model's outcome, uses maximum likelihood estimation to estimate a transformation parameter $\lambda$ in the equation
$$
x^*= \begin{cases}\frac{x^\lambda-1}{\lambda \tilde{x}^{\lambda-1}}, & \lambda \neq 0 \\ \tilde{x} \log x, & \lambda=0\end{cases}
$$
where $\tilde{x}$ is the geometric mean of the predictor data. In this procedure, $\lambda$ is estimated from the data. Because the parameter of interest is in the exponent, this type of transformation is called a power transformation. Some values of $\lambda$ map to common transformations, such as $\lambda=1$ (no transformation), $\lambda=0(\log ), \lambda=0.5$ (square root), and $\lambda=-1$ (inverse). As you can see, the Box-Cox transformation is quite flexible in its ability to address many different data distributions. For the data in</mark> [[@kuhnFeatureEngineeringSelection2020]]

Could map to the observation that trade prices are log-normally distributed. https://financetrain.com/why-lognormal-distribution-is-used-to-describe-stock-prices

We adhere to a notation found in [[@kuhnFeatureEngineeringSelection2020]]

Test log-normality visually with qq-plots (https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy) or using statistical tests e. g.,  log-transform + normality test. https://stats.stackexchange.com/questions/134924/tests-for-lognormal-distribution

<mark style="background: #FFB8EBA6;">Based on the (Box Cox test?),-> dates </mark> [[@boxAnalysisTransformations2022]] we apply a common $x^{\prime}=\log(x)$ transform to mitigate the skewness with the result of compressing large values and expanding smaller ones. More specifically, $x^{\prime}= \log(x+1)$ is used to prevent taking the logarithm of zero and improving numerical stability in floating point calculations[^1]. Due to the montonous nature of the logarithm, the splits of tree-based learners remain unaffected from this transformation.

<mark style="background: #FFB86CA6;">log-transform can hamper interpretability [[@fengLogtransformationItsImplications2014]]</mark>

Our largest feature set als contains dates and times of the trade. In contrast to other continous features, the features are inherently cyclic. We exploit this property for hours, days, and months and apply a fourier transform to convert the features into a smooth variable using formula [[#^773161]].

$$
\begin{aligned}
x' &= \sin\left(\frac{2\pi x}{86400} \right)\\
x'' &= \cos\left(\frac{2\pi x}{86400} \right)
\end{aligned}
$$
<mark style="background: #BBFABBA6;">TODO: make formula generic to for date, month, hour etc.  Put into a broader scope. -> See Kuhn?</mark>
^773161

This cyclic continous enconding, has the effect of preserving temporal proximity, as shown in Figure [[#^278944]]. As visualized for dates, the month's ultimo and the next month's first are close to each other in the individual features and on the unit circle. [^3]

![[positional_encoding.png]]
(found here similarily: https://www.researchgate.net/figure/A-unit-circle-example-of-frequency-encoding-of-spatial-data-using-the-Fourier-series-a_fig2_313829438) ^278944

To address the problem of convergence of our transformer-based architectures, we normalize the data set using $z$-score normalization given by formula [[#^5d5445]]: ~~we normalize all continous features into a range of $[-1,1]$ using formula [[#^5d5445]]:

$$
x^{\prime}=-1+\frac{2(x-\min (x))}{\max (x)-\min (x)} \tag{1}
$$
$$
X_{n o r m}=\frac{X-X_{\min }}{X_{\max }-X_{\min }}
$$

Standardization:
$$
z=\frac{x-\mu}{\sigma}\tag{1}
$$
with mean
$$
\mu=\frac{1}{N} \sum_{i=1}^N\left(x_i\right)
$$
and standard deviation
$$
\sigma=\sqrt{\frac{1}{N} \sum_{i=1}^N\left(x_i-\mu\right)^2}.
$$
^5d5445

<mark style="background: #D2B3FFA6;">Most algorithms based on gradient descent require data to be scaled.The presence of feature value X in the formula will affect the step size of the gradient descent. The difference in ranges of features will cause different step sizes for each feature. Having features on a similar scale can help the gradient descent to converge. Why is this not true for gradient boosting? https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/</mark>

<mark style="background: #FFB8EBA6;">- min-max scaling and $z$ scaling preserve the distribution of the variables  (see [here.](https://stats.stackexchange.com/a/562204/351242)). Applying both cancels out each other (see proof [here.](https://stats.stackexchange.com/a/562204/351242)). </mark>

Normalization has the advantage of preserving the data distribution, which is an important property when comparing our machine learning based models against their classical counterparts in chapter [[feature_importance]]. 

As for the categorical variables a transformation is required. We perform a label encoding by randomly mapping every unique value onto an integer key. As an example, the option type in the set $\{\text{'C'},\text{'P'}\}$ would be randomly mapped onto $\{1,0\}$. This basic transformation allows to defer handling of categorical data to the model [[@hancockSurveyCategoricalData2020]] (p. 10). Also, it minimizes target leakage. Classes not seen during are mapped to the key of a $\mathtt{[UNK]}$ token.

Known disadvantages of label encoding, as raised in [[@hancockSurveyCategoricalData2020]] (p. 12), such as the unequal contributions of larger keys to the loss in neural networks or the artificially implied order, do not apply here, as the conversion is followed by sophisticated treatments within the models[^2].

One aspect that remains open, is the high cardinality of categorical features with as many as (add number of classes of root) classes. We postpone strategies to model-specific treatments in chapter [[training-of-supervised-models]]. The chapter also provides further insights on handling categories observed during training.

An overview of all feature transformations is added to Appendix [[🍬appendix#^8e998b]].

[^1]: See e. g., https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
[^2]: See chapter on ordered boosting, tabtransformer, or the fttransformer.
[^3]: Notice the similarities to the positional encoding used in [[@vaswaniAttentionAllYou2017]].