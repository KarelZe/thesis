
## Links🔗
https://www.yourdatateacher.com/2022/05/02/are-your-training-and-test-sets-comparable/
https://stats.stackexchange.com/questions/208517/kolmogorov-smirnov-test-vs-t-test
http://finance.martinsewell.com/stylized-facts/dependence/
https://gsarantitis.wordpress.com/2020/04/16/data-shift-in-machine-learning-what-is-it-and-how-to-detect-it/
https://datascience.stanford.edu/news/splitting-data-randomly-can-ruin-your-model
https://scikit-learn.org/stable/modules/cross_validation.html#timeseries-cv
https://arxiv.org/pdf/1905.11744.pdf

## Ammos 🗒️

Serial dependence occurs when the value of a datapoint at one time is statistically dependent on another datapoint in another time. However, this attribute of time series data violates one of the fundamental assumptions of many statistical analyses  that data is statistically independent. (https://www.influxdata.com/blog/autocorrelation-in-time-series-data/)

However, the random selection of test observations does not warrant independence from training observations when dependence structures exist in the data, i.e., when observations close to each other tend to have similar characteristics

In many cases, data is time-correlated, which means that the time the data is generated affects its label distribution and thus overestimate the reported results for their classifier.

### Kolmorgov-Smirnov Test

- Seems to be more appropriate than the $t$-test.

```
dataset = load_diabetes(as_frame=True) X,y = dataset['data'],dataset['target'] X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

feature_name = 'bp' df = pd.DataFrame({ feature_name:np.concatenate((X_train.loc[:,feature_name],X_test.loc[:,feature_name])), 'set':['training']*X_train.shape[0] + ['test']*X_test.shape[0] }) sns.ecdfplot(data=df,x=feature_name,hue='set')
```



As the Even in-sample a perfect classification a perfect classification may be illussinary to . .

We do not achieve a perfect training accuracy which attribute to a low-signal-to-nose-ratio

For the static training scheme we follow Gu et al. (2021) and Chen et al. (2021) and split the data set into three sets. We use data until November 2010 for training and validate our models on data from December 2010 to October 2015, as shown in figure 1a. The remaining data is used for out-of-sample testing until June 2020 for the annual and May 2021 for the monthly data set. This yields a classical 59.67- 20.46-19.86 % split for the annual horizon. While computationally cheap, a static scheme fails to leverage recent information for the prediction from December 2010 onwards. A promising alternative is a rolling scheme, that incorporates recent data through retraining a model on updating sets, as employed by Freyberger et al. (2020), Gu et al. (2020) or Grammig et al. (2020). In a rolling scheme, fixed-size training and validation windows gradually shift forward in time, thereby dropping older observations and incorporating newer ones. The performance is evaluated on then 4 EMPIRICAL STUDY 13 unseen data starting in November 2016. We set the window length to one year for the training and the validation set and refit our models annually, as visualised in figure 1b totalling in up to ten re-trainings, which is a balanced choice between data recency and computational feasibility.

<mark style="background: #ABF7F7A6;">You don't randomly split in time-series datasets because it doesn't respect the temporal order and causes _data-leakage_, e.g. unintentionally inferring the trend of future samples.</mark>

<mark style="background: #BBFABBA6;">In machine learning, train/test split splits the data randomly, as there’s no dependence from one observation to the other. That’s not the case with time series data. Here, you’ll want to use values at the rear of the dataset for testing and everything else for training.</mark>

one of the fundamental assumptions of mafny statistical analyses  that data is statistically independent. (https://www.influxdata.com/blog/autocorrelation-in-time-series-data/)

Convert between label and class: https://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf

**Why?**
- The split is required to get unbiased performance estimates of our models. It is not required for classical rules, as these rules have no parameters to estimates or hyperparameters to tune.
- “Typically, machine learning involves a lot of experimentation, though – for example, the tuning of the **internal knobs of a learning algorithm**, the so-called hyperparameters. Running a learning algorithm over a training dataset with different hyperparameter settings will result in different models. Since we are typically interested in selecting the best-performing model from this set, we need to find a way to estimate their respective performances in order to rank them against each other.” ([[@raschkaModelEvaluationModel2020]], p. 4)
- “We want to estimate the generalisation performance, the predictive performance of our model on future (unseen) data.” (Raschka, 2020, p. 4)
- “We want to increase the predictive performance by tweaking the learning algorithm and selecting the best performing model from a given hypothesis space.” ([[@raschkaModelEvaluationModel2020]], p. 4)
- Training set is used to fit the model to the data
- Validation set is there for tuning the hyperparameters. ([[@hastietrevorElementsStatisticalLearning2009]] 222) write "to estimate prediction error for model selection"
- Test set for unbiased, out-of-sample performance estimates. ([[@hastietrevorElementsStatisticalLearning2009]] 222) write "estimate generalisation error of the model"
- https://stackoverflow.com/questions/4503325/autocorrelation-of-a-multidimensional-array-in-numpy

```python
def xcorr(x):
    l = 2 ** int(np.log2(x.shape[1] * 2 - 1))
    fftx = fft(x, n = l, axis = 1)
    ret = ifft(fftx * np.conjugate(fftx), axis = 1)
    ret = fftshift(ret, axes=1)
    return ret
import numpy
from numpy.fft import fft, ifft

data = numpy.arange(5*4).reshape(5, 4)
padding = numpy.zeros((5, 3))
dataPadded = numpy.concatenate((data, padding), axis=1)
print dataPadded
```

Ideally, the test set should be kept in a “vault,” and be brought out only at the end of the data analysis. Suppose instead that we use the test-set repeatedly, choosing the model with smallest test-set error. Then the test set error of the final chosen model will underestimate the true test error, sometimes substantially. It is difficult to give a general rule on how to choose the number of observations in each of the three parts, as this depends on the signal-tonoise ratio in the data and the training sample size. A typical split might be 50% for training, and 25% each for validation and testing:

**Our split:**
- We perform a *static* split into three disjoint sets. (aka holdout method)
- We use a 60-20-20 split and assign dates to be either in one set to simplify evaluation.  How does the rounding to the next day work? test set should be long enough to allow a meaningful comparison against Grauer
- Common splitting strategy should be dependent on the training sample size and signal-to-noise ratio in the data ([[@hastietrevorElementsStatisticalLearning2009]]222)
- A common split is e. g., 50-25-25. ([[@hastietrevorElementsStatisticalLearning2009]]222)
- Work of [[@grauerOptionTradeClassification2022]] showed that the classification performance deterioriates over time. Thus, most recent data poses the most rigorous test conditions due to the identical data basis. 
- The train set holds the most recent and thus most relevant observations, which will be most challenging to predict.
- We maintain the temporal ordering within the data and avoid data leakage: e. g., from unknown `ROOT`s, as only trailing observations are used. (see similar reasoning in [[@lopezdepradoAdvancesFinancialMachine2018]] for trading strategies).
- Observations in finance are often not iid. The test set is used multiple times during model development resulting in a testing and selection bias [[@lopezdepradoAdvancesFinancialMachine2018]]. Serial correlation might be less of an issue here.
- How can we incorporate uncertainty? How can we make sure that it was not just luck? -> confidence intervals (see [[@raschkaModelEvaluationModel2020]])
- Estimates might be lower bound -> [[@raschkaModelEvaluationModel2020]] refers to this as a pesimistic bias
- We split apart the test set early on as recommended by [[@lonesHowAvoidMachine2022]] (do not cite) The best thing you can do to prevent these issues is to partition off a subset of your data right at the start of your project, and only use this independent test set once to measure the generality of a single model at the end of the project (see Do save some data to evaluate your final model instance)

- “Does this replace the test set (or, analogously, the assessment set)? No. Since the validation data are guiding the training process, they can’t be used for a fair assessment for how well the modelling process is working” ([[@kuhnFeatureEngineeringSelection2020]], p. 53)



- To verify the samples in the training and validation set are drawn from the same distribution, we perform adversarial validation.  

![[learning-curves-bias-variance.png]]

The new gap between the two learning curves suggests a substantial increase in variance. The low training MSEs corroborate this diagnosis of high variance. The large gap and the low training error also indicates an overfitting problem. Overfitting happens when the model performs well on the training set, but far poorer on the test (or validation) set. One more important observation we can make here is that _adding new training instances_ is very likely to lead to better models.

**How much data is enough?**

The new gap between the two learning curves suggests a substantial increase in variance. The low training MSEs corroborate this diagnosis of high variance. The large gap and the low training error also indicates an overfitting problem. Overfitting happens when the model performs well on the training set, but far poorer on the test (or validation) set. One more important observation we can make here is that _adding new training instances_ is very likely to lead to better models. The validation curve doesn’t plateau at the maximum training set size used. It still has potential to decrease and converge toward the training curve, similar to the convergence we see in the linear regression case. So far, we can conclude tha


- more data is better, but what about the shift in data?
- Plot learning curves to estimate whether performance will increase with the number of samples. Use it to motivate semi-supervised learning.  [Plotting Learning Curves — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) and [Tutorial: Learning Curves for Machine Learning in Python for Data Science (dataquest.io)](https://www.dataquest.io/blog/learning-curves-machine-learning/)
![[learning-curves-samples.png]]


**Why no dynamic training split?**
- By the same reasoning, we expect a moving window to improve further. Model can learn on more recent data. By keeping the train-test split static it can be seen more like a lower bound. Reason about computational complexity.


**Why random / cv train-test splits are missleading**?
- [[@ronenMachineLearningTrade2022]] performed a 70-30 % random split. This can be problematic for obvious reasons.
- [[@ellisAccuracyTradeClassification2000]] performed a split, where two sets bracketed?
- Splitting time-correlated data randomly can bias the results and correlations are often non-obvious e. g., `ROOT`s, advent of etfs. She advocates to split data *by time* to avoid leakage ([[@huyenDesigningMachineLearning]]137)).
- Common assumption is that samples are “i.i.d. We assume that the training examples are i.i.d (independent and identically distributed), which means that all examples have been drawn from the same probability distribution and are statistically independent from each other. A scenario where training examples are not independent would be working with **temporal data** or time-series data.” ([[@raschkaModelEvaluationModel2020]], p. 5) -> samples are not iid here.
- “One other aspect of resampling is related to the concept of information leakage which is where the test set data are used (directly or indirectly) during the training process. This can lead to overly optimistic results that do not replicate on future data points and can occur in subtle ways.” ([[@kuhnFeatureEngineeringSelection2020]], 2020, p. 55)
- “As a final point on legitimacy, let us mention that once it has been clearly defined for a problem, the major challenge becomes preparing the data in such a way that ensures models built on this data would be leakage free. Alternatively, when we do not have full control over data collection or when they are simply given to us, a methodology for detecting when a large number of seemingly innocent pieces of information are in fact plagued with leakage is required. This shall be the focus of the following two sections.” ([[@kaufmanLeakageDataMining2012]], p. 162)

- “Leaking features are then covered by a simple condition for the absence of leakage: ∀x component of X , x ∈ legit{y}. (2) That is, any feature made available by the data preparation process is deemed legitimate by the precise formulation of the modelling problem at hand, element by element with respect to its matching target. The prevailing example for this type of leakage is what we call the no-time-machine requirement. In the context of predictive modelling, it is implicitly required that a legitimate model only build on features with information from a time earlier (or sometimes, no later) than that of the target. Formally, X and y, are defined over some time axis t (not necessarily physical time). Prediction is required by the client for a target element y at time t{y}. Each feature x (one of the components of X) associated with an observation is unobservable to the client until t{x} from then on it is observable. Let t{y} denote the ordered set resulting from element-wise application of the operator t{y} on the ordered set y. Similarly define the ordered set t{x }. We then have: t{x } < t{y}⇔x ∈ legit{y}. (3) Such a rule should be read as: A legitimate feature is an ordered set whose every element is observable to the client earlier than its W-associated target element. Note that the different definitions of the “timestamping” operator t for features and targets is crucial. A good example for its necessity is leakage in the financial world, which relates to the date when information becomes public, and thus observable to the client using a hypothetical financial model (assuming the client is not a rogue inside trader). Specifically, stock-price prediction models would be highly “successful” should they use quarterly data assuming they are available a day after the quarter ends, whereas in reality they are usually publicly known only about three weeks later. We therefore define leakage legitimacy in the predictive modelling case using the concept of observability time of the features and prediction time of the target. While the simple no-time-machine requirement is indeed the most common case, one could think of additional scenarios which are still covered by condition (2).” ([[@kaufmanLeakageDataMining2012]], p. 159)
- “or modelling problems where the usual “i.i.d. elements” assumption is valid, and when without loss of generality considering all information specific to the element being predicted as features rather than examples, condition (9) simply reduces to condition (2) since irrelevant observations can always be considered legitimate. In contrast, when dealing with problems exhibiting nonstationarity, otherwise known as concept drift Widmer and Kubat 1996, and more specifically the case when samples of the target are not mutually independent, condition (9) cannot be reduced to condition (2).” ([[@kaufmanLeakageDataMining2012]], p. 161)
- See argumentation in [[@lopezdepradoAdvancesFinancialMachine2018]] why CV is problematic. CV assumes the samples to be iid, in practise they are not.

**How can we test for serial correlations?**
- I think it's ok to argue by example.



