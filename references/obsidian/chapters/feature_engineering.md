- Some features are more difficult to learn for decision trees and neural nets. Provide aid. https://www.kaggle.com/code/jeffheaton/generate-feature-engineering-dataset/notebook
- Two aspects should drive the feature selection / creation:
	- Use features, that are used in classical rules
	- Apply transformers, that are best suited for the models.
	- Create larger feature sets to find the very best feature set.
- Which features are very different in the training set and the validation set?
- Which features are most important in adversarial validation?
- Plot distributions of features from training and validation set. Could also test using https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test test if samples are drawn from the same distribution.
- See https://neptune.ai/blog/tabular-data-binary-classification-tips-and-tricks-from-5-kaggle-competitions for more ideas
- Previously not done due to use of simple rules only. 
- Try different encondings e. g., of the spread.
- Which architectures require what preprocessing? Derive?
- Perform search?
- Standardization is necessary for algorithms that are sensitive to the scale of features, such as neural net. [[@banachewiczKaggleBookData2022]]
- Use simple method to impute missing data like mean. We seldomly require more sophisticated methods. [[@banachewiczKaggleBookData2022]]
- Look which of the models can handle missing data inherently. How would TabNet or TabTransformer do it?
- Can some of the features be economically motivated?
- Apply feature transformations that are economically motivated.
- It might be wise to limit the transformations to ones that are present in the classical rules. Would help with reasoning.
- Try out features that are inherently used in the depth rule or the trade rule. 
- For imputation look into [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]]
- [[@josseConsistencySupervisedLearning2020]] also compare different imputation methods and handling approaches of missing values in tree-based methods.
- for visualizations and approaches see [[@zhengFeatureEngineeringMachine]] and [[@butcherFeatureEngineeringSelection2020]]
- Positional encoding was achieved using $\sin()$ and $\cos()$ transformation.
- ![[sine_cosine_transform 1.png]]
- [[@ronenMachineLearningTrade2022]] suggest to use models that can handle time series components. This would limit our choices. Thus we use feature engineering to induce a notion of time into our models.

- Implementation pattern https://www.linkedin.com/posts/sarah-floris_python-pythonprogramming-cleancode-activity-6990302724584087552-6lzF?utm_source=share&utm_medium=member_android
- Think about using a frequency of trade feature or so. Also use order imbalances as features. Came up with this idea when reading [[@aitkenIntradayAnalysisProbability1995]]
- Some feature ideas like order imbalance could be adapted from [[@aitkenIntradayAnalysisProbability1995]].
- Positional encode trading time throughout the day.
- Explain why it is necessary to include lagged data as column -> most ml models for tabular data only read rowise. No notion of previous observations etc. Some approaches however exist like specialized attention mechanisms to develop a notion of proximity.
- min-max scaling and $z$ scaling preserve the distribution of the variables  (see [here.](https://stats.stackexchange.com/a/562204/351242)). Applying both cancels out each other (see proof [here.](https://stats.stackexchange.com/a/562204/351242)). 
- zero imputation might be a poor choice for neural networks. (see practical and theoretical explanation in [[@yiWhyNotUse2020]]).
- [[@yiWhyNotUse2020]] and [[@smiejaProcessingMissingData2018]] contain various references to papers to impute missing data in neural networks. 
- Cite [[@rubinInferenceMissingData1976]] for different patterns in missing data.
- [[@lemorvanWhatGoodImputation2021]] for theoretical work on imputation.
- For patterns and analysis of imputed data see https://stefvanbuuren.name/fimd/ch-analysis.html

- What are the drawbacks of feature engineering?
- Why standardize the data at all?
- How is the definition of feature sets be motivated?
- Why does positional encoding make sense?
- Differentiate between categorical and continous variables?
- How to handle high number of categorical variables in data set? How does this relate to gradient boosted trees and transformers?
	- What does it mean for the number of parameters in a transformer model to have one more category?
	- Use a linear projection: https://www.kaggle.com/code/limerobot/dsb2019-v77-tr-dt-aug0-5-3tta/notebook
	- https://en.wikipedia.org/wiki/Additive_smoothing
	- How is the training of the gradient boosted tree affected?
	- For explosion in parameters also see [[@tunstallNaturalLanguageProcessing2022]]. Could apply their reasoning (calculate no. of parameters) for my work. 
	- KISS. Dimensionality is probably not so high, that it can not be handled. It's much smaller than common corpi sizes. Mapping to 'UKNWN' character. -> Think how this can be done using the current `sklearn` implementation.
	- The problem of high number of categories is called a high cardinality problem of categoricals see e. g., [[@huangTabTransformerTabularData2020]]
- Why do we need standardized inputs for neural nets?
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
	- [[@huangSnapshotEnsemblesTrain2017]] argue, that for continous features both quantized, normalized and log scaled can be kept. The say, that this redundant encoding shouldn't lead to overfitting.
- for final feature set see [[🧃Feature Sets]]
- combine size features and price features into a ratio. e. g., "normalize" price with volume. Found this idea here [[@antoniouLognormalDistributionStock2004]]
- log-transform can hamper interpretability [[@fengLogtransformationItsImplications2014]]
- The right word for testing different settings e. g., scalings or imputation approaches is https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence) 
- In my dataset the previous or subsequent trade price is already added as feature and thus does not have to be searched recursively.
- Motivation for scaling features to $[-1,1]$ range or zero mean. https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used
- If needed tokenization support: https://github.com/google/sentencepiece
- Visualize behaviour over time e. g., appearing `ROOT`s and calculate statistics. How many of the clients / percentage are in the train set and how many are just in the test set?
![[uuid_over_time.png]]
(found at https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284)
- Add date features such as season

## Problem of missing values and categoricals
The required pre-processing is minimal for tree-based learners. Missing values can be handled by sending down `[NaN]` values at one side of the tree. Recent literature indicates that handling missing data inside the algorithm slightly improves over ... simpler approaches over . Also, some tree-based learners can handle categorical data without prior pre-processing, as shown in our chapter on ordered boosting ([[🐈gradient-boosting]]).

Neural networks can not inherently handle missing values, as a $\mathtt{[NaN]}$ value can not be propagated through the network. As such, missing values must be addressed beforehand. Similarily, categorical features, like the issue type, require an encoding <mark style="background: #FF5582A6;">(can not be input into the network)</mark>, as no gradient can be calculated on categories <mark style="background: #FF5582A6;">(such as $\mathtt{'C'}$ or $\mathtt{'P'}$)</mark>.

## Solution to missing values and categoricals
In order to prepare a common datasets for *all* our models, we need to impute, scale and encode the data. Like in the chapter [[preprocessing]] our feature scaling aims to be minimal intrusive, while facilitating an efficient training for all our machine learning models.

Missing values are imputed with zeros. This simple, one-pass  strategy ~~minimizes the bias from imputation~~, avoids data leakage, and allows tree-based learners and neural networks to separate imputed values from observed ones. While the imputation with constants is simple, it is on-par with more complex approaches as (...) while minimizing the bias from imputatation. 

As introduced in the chapters [[🐈gradient-boosting]] and [[🤖transformer]] both architectures are known to be robust to missing values. In conjunction with the low degree of missing values (compare chapter [[🌴exploratory_data_analysis]]), we therefore expect the impact from missing values to be minor. To address concerns, that the imputation or scaling negatively impacts the performance of gradient boosted trees, we perform an ablation study in chapter [[🎋ablation_study]], and retrain our models on the unscaled and unimputed data set.

## Problem of feature scales

Tree-based models can handle arbitrary feature scales, as the splitting process is based on the purity of the split but not on the scale of the splitting value. 

Also, neural networks are known to train faster, .... <mark style="background: #BBFABBA6;">normalized / standardized data</mark> . problems with convergence etc.

## Solution of feature scales

Continous and categorical variable require a differentiated approach. 

Price and size-related features exhibit a positive skewness, as brought up in chapter [[🌴exploratory_data_analysis]].  We apply a common $x^{\prime}=\log(x)$ transform to mitigate skewness with the effect of compressing large values and expanding smaller ones. More specifically, $x^{\prime}= \log(x+1)$ is used to prevent taking the logarithm of zero <mark style="background: #FFB86CA6;">and improve numerical stability in floating point calculations. (See e. g., https://numpy.org/doc/stable/reference/generated/numpy.log1p.html).</mark>
Due to the montonous nature of the $\log(\cdot)$, the distribution of the feature is preserved. 

Likewise, normalization as given in formula (...) maintains the distribution of the data, thereby altering only the range. 

If you want your feature to be in an arbitrary range $[a, b]$-empirically, I find the range $[-1,1]$ to work better than the range $[0,1]$-you can use the following formula:
$$
x^{\prime}=a+\frac{(x-\min (x))(b-a)}{\max (x)-\min (x)}
$$


<mark style="background: #ADCCFFA6;">he missing values were substituted with zeros for the linear regression and models based on pure neural networks since these methods cannot accept them otherwise. We apply the ordinal encoding to categorical values for all models. According to the work [54], the chosen encoding strategy shows comparable performance to more advanced methods.
</mark>

One problem that remains open, is the high cardinality of categorical features. We derive strategies in chapter [[training-of-supervised-models]]. The chapter also provides further insights on handling categories not seen during training.