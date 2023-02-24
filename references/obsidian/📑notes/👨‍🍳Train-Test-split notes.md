

## Option Seminar 🦬
The available data set is split into three disjoint sets. First, the training set is used to fit the model to the data. Secondly, the validation set is dedicated to tuning the 2We also considered other scaling schemes, including rank-wise, global, and mixed approaches. 3We use different imputation strategies, including removal of missing values, zero fill, linear interpolation, and imputation with the subset mean and median. We find zero-fill to work best. 4 EMPIRICAL STUDY 12 models hyperparameters, such as the ν of a GBM. Conforming to the best practice approach, the accuracy of the return forecasts is assessed on unseen data. As the data set comes in a multi-time-series form, a strict temporal ordering must be maintained when splitting the data so that models can capture the temporal properties of the data and leakage between sets is minimised. We compare two approaches that maintain this property with a static training scheme and a rolling scheme. For the static training scheme we follow Gu et al. (2021) and Chen et al. (2021) and split the data set into three sets. We use data until November 2010 for training and validate our models on data from December 2010 to October 2015, as shown in figure 1a. The remaining data is used for out-of-sample testing until June 2020 for the annual and May 2021 for the monthly data set. This yields a classical 59.67- 20.46-19.86 % split for the annual horizon. While computationally cheap, a static scheme fails to leverage recent information for the prediction from December 2010 onwards. A promising alternative is a rolling scheme, that incorporates recent data through retraining a model on updating sets, as employed by Freyberger et al. (2020), Gu et al. (2020) or Grammig et al. (2020). In a rolling scheme, fixed-size training and validation windows gradually shift forward in time, thereby dropping older observations and incorporating newer ones. The performance is evaluated on then 4 EMPIRICAL STUDY 13 unseen data starting in November 2016. We set the window length to one year for the training and the validation set and refit our models annually, as visualised in figure 1b totalling in up to ten re-trainings, which is a balanced choice between data recency and computational feasibility



## Ammos


**Why?**
- The split is required to get unbiased performance estimates of our models. It is not required for classical rules, as these rules have no parameters to estimates or hyperparameters to tune.
- “Typically, machine learning involves a lot of experimentation, though – for example, the tuning of the **internal knobs of a learning algorithm**, the so-called hyperparameters. Running a learning algorithm over a training dataset with different hyperparameter settings will result in different models. Since we are typically interested in selecting the best-performing model from this set, we need to find a way to estimate their respective performances in order to rank them against each other.” ([[@raschkaModelEvaluationModel2020]], p. 4)
- “We want to estimate the generalization performance, the predictive performance of our model on future (unseen) data.” (Raschka, 2020, p. 4)
- “We want to increase the predictive performance by tweaking the learning algorithm and selecting the best performing model from a given hypothesis space.” ([[@raschkaModelEvaluationModel2020]], p. 4)
- Training set is used to fit the model to the data
- Validation set is there for tuning the hyperparameters. ([[@hastietrevorElementsStatisticalLearning2009]] 222) write "to estimate prediction error for model selection"
- Test set for unbiased, out-of-sample performance estimates. ([[@hastietrevorElementsStatisticalLearning2009]] 222) write "estimate generalization error of the model"

**Our split:**
- We perform a *static* split into three disjoint sets. (aka holdout method)
- We use a 60-20-20 split and assign dates to be either in one set to simplify evaluation.  How does the rounding to the next day work? test set should be long enough to allow a meaningful comparsion against Grauer
- dates:
```python
train = df[df.QUOTE_DATETIME.between("2005-05-02 00:00:01", "2013-10-24 23:59:00")]
val = df[df.QUOTE_DATETIME.between("2013-10-25 00:00:01", "2015-11-05 23:59:00")]
test = df[df.QUOTE_DATETIME.between("2015-11-06 00:00:01", "2017-05-31 23:59:00")]
```

- Common splitting strategy should be dependent on the training sample size and signal-to-noise ratio in the data ([[@hastietrevorElementsStatisticalLearning2009]]222)
- A common split is e. g., 50-25-25. ([[@hastietrevorElementsStatisticalLearning2009]]222)
- Work of [[@grauerOptionTradeClassification2022]] showed that the classification performance deterioriates over time. Thus, most recent data poses the most rigorous test conditions due to the identical data basis. 
- The train set holds the most recent and thus most relevant observations, which will be most challenging to predict.
- We maintain the temporal ordering within the data and avoid data leakage: e. g., from unknown `ROOT`s, as only trailing observations are used. (see similar reasoning in [[@lopezdepradoAdvancesFinancialMachine2018]] for trading strategies).
- Observations in finance are often not iid. The test set is used multiple times during model development resulting in a testing and selection bias [[@lopezdepradoAdvancesFinancialMachine2018]]. Serial correlation might be less of an issue here.
- How can we incorporate uncertainty? How can we make sure that it was not just luck? -> confidence intervals (see [[@raschkaModelEvaluationModel2020]])
- Estimates might be lower bound -> [[@raschkaModelEvaluationModel2020]] refers to this as a pesimistic bias
- We split apart the test set early on as recommended by [[@lonesHowAvoidMachine2022]] (do not cite) The best thing you can do to prevent these issues is to partition off a subset of your data right at the start of your project, and only use this independent test set once to measure the generality of a single model at the end of the project (see Do save some data to evaluate your final model instance)

- Come back to notation in [[🍪Selection Of Supervised Approaches]]

- “Does this replace the test set (or, analogously, the assessment set)? No. Since the validation data are guiding the training process, they can’t be used for a fair assessment for how well the modeling process is working” ([[@kuhnFeatureEngineeringSelection2020]], p. 53)

- Adress leakage as part of eda. “Exploratory Data Analysis (EDA) can be a powerful tool for identifying leakage. EDA is the good practice of getting more intimate with the raw data, examining it through basic and interpretable visualization or statistical tools. Prejudice free and methodological, this kind of examination can expose leakage as patterns in the data that are surprising.” ([[@kaufmanLeakageDataMining2012]] p. 165)

- “On the very practical side, a good starting point for EDA is to look for any form of unexpected data properties. Common giveaways are found in identifiers, matching (or inconsistent matching) of identifiers (i.e., sample selection biases), surprises in distributions (spikes in densities of continuous values), and finally suspicious order in supposedly random data.” ([[@kaufmanLeakageDataMining2012]], p. 165)

- To verify the samples in the training and validation set are drawn from the same distribution, we perform adversarial validation.  

**How much data is enough?**
- more data is better, but what about the shift in data?
- Plot learning curves to estimate whether performance will increase with the number of samples. Use it to motivate semi-supervised learning.  [Plotting Learning Curves — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) and [Tutorial: Learning Curves for Machine Learning in Python for Data Science (dataquest.io)](https://www.dataquest.io/blog/learning-curves-machine-learning/)
![[learning-curves-samples.png]]


**Why no dynamic training split?**
- By the same reasoning, we expect a moving window to improve further. Model can learn on more recent data. By keeping the train-test split static it can be seen more like a lower bound. Reason about computational complexity.
- “3.4.4 | Rolling Origin Forecasting This procedure is specific to time-series data or any data set with a strong temporal component (Hyndman and Athanasopoulos, 2013). If there are seasonal or other chronic trends in the data, random splitting of data between the analysis and assessment sets may disrupt the model’s ability to estimate these patterns. In this scheme, the first analysis set consists of the first M training set points, assuming that the training set is ordered by time or other temporal component. The assessment set would consist of the next N training set samples. The secon esample keeps the data set sizes the same but increments the analysis set to use samples 2 through M + 1 while the assessment contains samples M + 2 to M + N + 1. The splitting scheme proceeds until there is no more data to produce the same data set sizes. Supposing that this results in B splits of the data, the same process is used for modeling and evaluation and, in the end, there are B estimates of performance generated from each of the assessment sets. A simple method for estimating performance is to again use simple averaging of these data. However, it should be understood that, since there is significant overlap in the rolling assessment sets, the B samples themselves constitute a time-series and might also display seasonal or other temporal effects. Figure 3.8 shows this type of resampling where 10 data points are used for analysis and the subsequent two training set samples are used for assessment. There are a number of variations of the procedure: • The analysis sets need not be the same size. They can cumulatively grow as the moving window proceeds along the training set. In other words, the first analysis set would contain M data points, the second would contain M + 1 and so on. This is the approach taken with the Chicago train data modeling and is described in Chapter 4. • The splitting procedure could skip iterations to produce fewer resamples. For example, in the Chicago data, there are daily measurements from 2001 to 2016. Incrementing by one day would produce an excessive value of B. For these data, 13 samples were skipped so that the splitting window moves in two-week blocks instead of by individual day. • If the training data are unevenly sampled, the same procedure can be used buFor example, the window could move over 12-hour periods for the analysis sets and 2-hour periods for the assessment sets. This resampling method differs from the previous ones in at least two ways. The splits are not random and the assessment data set is not the remainder of the training set data once the analysis set was removed.” ([[@kuhnFeatureEngineeringSelection2020]], p. 53)


**Why random / cv train-test splits are missleading**?
- [[@ronenMachineLearningTrade2022]] performed a 70-30 % random split. This can be problematic for obvious reasons.
- [[@ellisAccuracyTradeClassification2000]] performed a split, where two sets bracketed?
- Splitting time-correlated data randomly can bias the results and correlations are often non-obvious e. g., `ROOT`s, advent of etfs. She advocates to split data *by time* to avoid leakage ([[@huyenDesigningMachineLearning]]137)).
- Common assumption is that samples are “i.i.d. We assume that the training examples are i.i.d (independent and identically distributed), which means that all examples have been drawn from the same probability distribution and are statistically independent from each other. A scenario where training examples are not independent would be working with **temporal data** or time-series data.” ([[@raschkaModelEvaluationModel2020]], p. 5) -> samples are not iid here.
- “One other aspect of resampling is related to the concept of information leakage which is where the test set data are used (directly or indirectly) during the training process. This can lead to overly optimistic results that do not replicate on future data points and can occur in subtle ways.” ([[@kuhnFeatureEngineeringSelection2020]], 2020, p. 55)
- “As a final point on legitimacy, let us mention that once it has been clearly defined for a problem, the major challenge becomes preparing the data in such a way that ensures models built on this data would be leakage free. Alternatively, when we do not have full control over data collection or when they are simply given to us, a methodology for detecting when a large number of seemingly innocent pieces of information are in fact plagued with leakage is required. This shall be the focus of the following two sections.” ([[@kaufmanLeakageDataMining2012]], p. 162)

- “Leaking features are then covered by a simple condition for the absence of leakage: ∀x component of X , x ∈ legit{y}. (2) That is, any feature made available by the data preparation process is deemed legitimate by the precise formulation of the modeling problem at hand, element by element with respect to its matching target. The prevailing example for this type of leakage is what we call the no-time-machine requirement. In the context of predictive modeling, it is implicitly required that a legitimate model only build on features with information from a time earlier (or sometimes, no later) than that of the target. Formally, X and y, are defined over some time axis t (not necessarily physical time). Prediction is required by the client for a target element y at time t{y}. Each feature x (one of the components of X) associated with an observation is unobservable to the client until t{x} from then on it is observable. Let t{y} denote the ordered set resulting from element-wise application of the operator t{y} on the ordered set y. Similarly define the ordered set t{x }. We then have: t{x } < t{y}⇔x ∈ legit{y}. (3) Such a rule should be read as: A legitimate feature is an ordered set whose every element is observable to the client earlier than its W-associated target element. Note that the different definitions of the “timestamping” operator t for features and targets is crucial. A good example for its necessity is leakage in the financial world, which relates to the date when information becomes public, and thus observable to the client using a hypothetical financial model (assuming the client is not a rogue inside trader). Specifically, stock-price prediction models would be highly “successful” should they use quarterly data assuming they are available a day after the quarter ends, whereas in reality they are usually publicly known only about three weeks later. We therefore define leakage legitimacy in the predictive modeling case using the concept of observability time of the features and prediction time of the target. While the simple no-time-machine requirement is indeed the most common case, one could think of additional scenarios which are still covered by condition (2).” ([[@kaufmanLeakageDataMining2012]], p. 159)
- “or modeling problems where the usual “i.i.d. elements” assumption is valid, and when without loss of generality considering all information specific to the element being predicted as features rather than examples, condition (9) simply reduces to condition (2) since irrelevant observations can always be considered legitimate. In contrast, when dealing with problems exhibiting nonstationarity, otherwise known as concept drift Widmer and Kubat 1996, and more specifically the case when samples of the target are not mutually independent, condition (9) cannot be reduced to condition (2).” ([[@kaufmanLeakageDataMining2012]], p. 161)
- See argumentation in [[@lopezdepradoAdvancesFinancialMachine2018]] why CV is problematic. CV assumes the samples to be iid, in practice they are not.

**How can we test for serial correlations?**
- I think it's ok to argue by example.

**Train-test-split in trade classification:**
- *validation* Classical rules typically don't require a validation set, as the rules are free of hyperparameters.
- *test* authors use a true out-of-sample test set to test their hypothesis. April for training and May to June for testing. Also, they test their hypothesis on a second data set. ([[@chakrabartyTradeClassificationAlgorithms2007]]3809)

**Visualization:**
![[viz-training-schemes.png]]




