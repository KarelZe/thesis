## Motivation of feature engineering

<mark style="background: #ABF7F7A6;">- Some features are more difficult to learn for decision trees and neural nets. Provide aid. https://www.kaggle.com/code/jeffheaton/generate-feature-engineering-dataset/notebook
</mark>

## Feature set definition

<mark style="background: #ABF7F7A6;">- A natural question is, ... About the usefulness of the engineered featues.
-Two aspects should drive / guide the feature selection / creation:
- Use features, that are used in classical rules
- Apply transformers, that are best suited for the models.
- Think about using a frequency of trade feature or so. Also use order imbalances as features. Came up with this idea when reading [[@aitkenIntradayAnalysisProbability1995]]
- Some feature ideas like order imbalance could be adapted from [[@aitkenIntradayAnalysisProbability1995]].
</mark>

<mark style="background: #D2B3FFA6;">- [[@ronenMachineLearningTrade2022]] suggest to use models that can handle time series components. This would limit our choices. Thus we use feature engineering to induce a notion of time into our models.

- Explain why it is necessary to include lagged data as column -> most ml models for tabular data only read rowise. No notion of previous observations etc. Some approaches however exist like specialized attention mechanisms to develop a notion of proximity.</mark>

 All feature set, the their definition and origin is documented in Appendix [[🍬appendix#^7c0162]].

## Problem of missing values and categoricals
The required pre-processing is minimal for tree-based learners. Missing values can be handled by sending down `[NaN]` values at one side of the tree. Recent literature indicates that handling missing data inside the algorithm slightly improves over ... simpler approaches over . Also, some tree-based learners can handle categorical data without prior pre-processing, as shown in our chapter on ordered boosting ([[🐈gradient-boosting]]).

Neural networks can not inherently handle missing values, as a $\mathtt{[NaN]}$ value can not be propagated through the network. As such, missing values must be addressed beforehand. Similarily, categorical features, like the issue type, require an encoding, as no gradient can be calculated on categories.

## Solution to missing values and categoricals
In order to prepare a common datasets for *all* our models, we need to impute, scale and encode the data. Like in the chapter [[preprocessing]] our feature scaling aims to be minimal intrusive, while facilitating an efficient training for all our machine learning models.

Missing values are imputed with zeros. This simple, one-pass  strategy ~~minimizes the bias from imputation~~, avoids data leakage, and allows tree-based learners and neural networks to separate imputed values from observed ones. While the imputation with constants is simple, it is on-par with more complex approaches as <mark style="background: #FF5582A6;">(...)</mark> while minimizing the bias from imputatation. <mark style="background: #FF5582A6;">(Note zero imputation can be problematic for neural nets, as shown in [[@yiWhyNotUse2020]] paper)</mark>
<mark style="background: #FF5582A6;">- For imputation look into [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]]
- [[@josseConsistencySupervisedLearning2020]] also compare different imputation methods and handling approaches of missing values in tree-based methods.
- for visualizations and approaches see [[@zhengFeatureEngineeringMachine]] and [[@butcherFeatureEngineeringSelection2020]]</mark>
<mark style="background: #FF5582A6;">[[@yiWhyNotUse2020]] and [[@smiejaProcessingMissingData2018]] contain various references to papers to impute missing data in neural networks. 
</mark>
- [[@lemorvanWhatGoodImputation2021]] for theoretical work on imputation.
- For patterns and analysis of imputed data see https://stefvanbuuren.name/fimd/ch-analysis.html
As introduced in the chapters [[🐈gradient-boosting]] and [[🤖transformer]] both architectures have found to be robust to missing values. 

In conjunction with the low degree of missing values (compare chapter [[🌴exploratory_data_analysis]]), we therefore expect the impact from missing values to be minor. To address concerns, that the imputation or scaling negatively impacts the performance of gradient boosted trees, we perform an ablation study in chapter [[🎋ablation_study]], and retrain our models on the unscaled and unimputed data set.

## Problem of feature scales

Tree-based models can handle arbitrary feature scales, as the splitting process is based on the purity of the split but not on the scale of the splitting value. 

<mark style="background: #FFB86CA6;">
Also, neural networks are known to train faster, .... normalized / standardized data . problems with convergence etc. Motivation for scaling features to $[-1,1]$ range or zero mean. https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used</mark>

## Solution of feature scales

Continous and categorical variable require different treatment, as derived below. 

Price and size-related features exhibit a positive skewness, as brought up in chapter [[🌴exploratory_data_analysis]]. <mark style="background: #FFB8EBA6;">Based on the (Box Cox test?),</mark> we apply a common $x^{\prime}=\log(x)$ transform to mitigate the skewness with the result of compressing large values and expanding smaller ones. More specifically, $x^{\prime}= \log(x+1)$ is used to prevent taking the logarithm of zero and improving numerical stability in floating point calculations[^1]. Due to the montonous nature of the logarithm, the splits of tree-based learners remain unaffected from this transformation.

Our largest feature set als contains dates and times of the trade. In contrast to other continous features, the features are inherently cyclic. We exploit this fact for hours, dates and months and apply a fourier transform to convert the features into a smooth variable using formula [[#^773161]].

$$
\begin{aligned}
x' &= \sin\left(\frac{2\pi x}{86400} \right)\\
x'' &= \cos\left(\frac{2\pi x}{86400} \right)
\end{aligned}
$$
<mark style="background: #BBFABBA6;">TODO: make formula generic to for date, month, hour etc.  Put into a broader scope.</mark>
^773161

This cyclic continous enconding, has the effect of preserving temporal proximity, as shown in Figure [[#^278944]]. As visualized for dates, the month's ultimo and the next month's first are close to each other in the individual features and on the unit circle. [^3]

![[positional_encoding.png]]
(found here similarily: https://www.researchgate.net/figure/A-unit-circle-example-of-frequency-encoding-of-spatial-data-using-the-Fourier-series-a_fig2_313829438) ^278944

To address the problem of <mark style="background: #FF5582A6;">(...)</mark> of our transformer-based architectures, we normalize all continous features into a range of $[-1,1]$ using formula [[#^5d5445]]:

$$
x^{\prime}=-1+\frac{2(x-\min (x))}{\max (x)-\min (x)} \tag{1}
$$

^5d5445
<mark style="background: #FFB8EBA6;">- min-max scaling and $z$ scaling preserve the distribution of the variables  (see [here.](https://stats.stackexchange.com/a/562204/351242)). Applying both cancels out each other (see proof [here.](https://stats.stackexchange.com/a/562204/351242)). </mark>

Normalization has the advantage of preserving the data distribution, which is an important property when comparing our machine learning based models against their classical counterparts in chapter [[feature_importance]]. 

As for the categorical variables a transformation is required. We perform a label encoding by randomly mapping every unique value onto an integer key. As an example, the option type in the set $\{\text{'C'},\text{'P'}\}$ would be randomly mapped onto $\{1,0\}$. This basic transformation allows to defer handling of categorical data to the model [[@hancockSurveyCategoricalData2020]] (p. 10). Also, it minimizes target leakage. Classes not seen during are mapped to the key of a $\mathtt{[UNK]}$ token. <mark style="background: #D2B3FFA6;">(Notice the severe impact with regard to the feature root.)</mark>

Known disadvantages of label encoding, as raised in [[@hancockSurveyCategoricalData2020]] (p. 12), such as the unequal contributions of larger keys to the loss in neural networks or the artificially implied order, do not apply here, as the conversion is followed by sophisticated treatments within the models[^2].

One aspect that remains open, is the high cardinality of categorical features with as many as (add number of classes of root) classes. We postpone strategies to model-specific treatments in chapter [[training-of-supervised-models]]. The chapter also provides further insights on handling categories observed during training.

An overview of all feature transformations is added to Appendix [[🍬appendix#^8e998b]].

- Create larger feature sets to find the very best feature set.
- Which features are very different in the training set and the validation set?
- Which features are most important in adversarial validation?
- Plot distributions of features from training and validation set. Could also test using https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test test if samples are drawn from the same distribution.

- Try different encondings e. g., of the spread.

- Cite [[@rubinInferenceMissingData1976]] for different patterns in missing data.


- What are the drawbacks of feature engineering?

- How is the definition of feature sets be motivated?
- Why does positional encoding make sense?
- Differentiate between categorical and continous variables?


- Why does standardization not affect learning of gbms
- Motivation for use of $\log$ is turn lognormal distribution into normal distribution or to reduce variability coming from outliers. 
	- https://datascience.stackexchange.com/questions/40089/what-is-the-reason-behind-taking-log-transformation-of-few-continuous-variables
	- Test log-normality visually with qq-plots (https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy) or using statistical tests e. g.,  log-transform + normality test. https://stats.stackexchange.com/questions/134924/tests-for-lognormal-distribution
	- Verify that my observation that log transform works only prices but not so much for size features. Could map to the observation that trade prices are log-normally distributed. https://financetrain.com/why-lognormal-distribution-is-used-to-describe-stock-prices
	- For references to tests for log normality see [[@antoniouLognormalDistributionStock2004]]
	- handle variables with high cardinality
- How do to reduce the number of categorical variables?
- strict assumption as we have out-of-vocabulary tokens e. g., unseen symbols like "TSLA".  (see done differently here https://keras.io/examples/structured_data/tabtransformer/)
- Idea: Instead of assign an unknown token it could help assign to map the token to random vector. https://stackoverflow.com/questions/45495190/initializing-out-of-vocabulary-oov-tokens
- Idea: reduce the least frequent root symbols.
- Apply an idea similar to sentence piece. Here, the number of words in vocabulary is fixed https://github.com/google/sentencepiece. See repo for paper / algorithm.
- redundant features:
	- [[@huangSnapshotEnsemblesTrain2017]] argue, that for continous features both quantized, normalized and log scaled can be kept. The say, that this redundant encoding shouldn't lead to overfitting
- combine size features and price features into a ratio. e. g., "normalize" price with volume. Found this idea here [[@antoniouLognormalDistributionStock2004]]
- log-transform can hamper interpretability [[@fengLogtransformationItsImplications2014]]
- The right word for testing different settings e. g., scalings or imputation approaches is https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence) 
- In my dataset the previous or subsequent trade price is already added as feature and thus does not have to be searched recursively.
kenization support: https://github.com/google/sentencepiece


[^1]: See e. g., https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
[^2]: See chapter on ordered boosting, tabtransformer, or the fttransformer.
[^3]: Notice the similarities to the positional encoding used in [[@vaswaniAttentionAllYou2017]].