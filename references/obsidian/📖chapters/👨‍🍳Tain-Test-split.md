eschews data leakage problems

Previous works 

https://gsarantitis.wordpress.com/2020/04/16/data-shift-in-machine-learning-what-is-it-and-how-to-detect-it/

The datasets (see cref-[[🌏Dataset]]) are split into three disjoint sets. The training set is used to fit the model to the data. The validation set dedicated to tuning the hyperparameters, and we reserve the test set for unbiased out-of-sample estimates.

Two main observations guide the train-test-split:
- samples are not i.i.d.
- drift in the data

Serial dependence occurs when the value of a datapoint at one time is statistically dependent on another datapoint in another time. However, this attribute of time series data violates one of the fundamental assumptions of many statistical analyses  that data is statistically independent. (https://www.influxdata.com/blog/autocorrelation-in-time-series-data/)

To verify the samples in the training and validation set are drawn from the same distribution, we perform adversarial validation. By this means, we eliminate features  

We use a classical train-test split to 


The performed split is notably different from previous works. 

**Train-test-split in trade classification:**
- *validation* Classical rules typically don't require a validation set, as the rules are free of hyperparameters.
- *test* authors use a true out-of-sample test set to test their hypothesis. April for training and May to June for testing. Also, they test their hypothesis on a second data set. ([[@chakrabartyTradeClassificationAlgorithms2007]]3809)
 [[@ellisAccuracyTradeClassification2000]] use true out-of-sample testing / second test set.

Both splits would ever estimate the model performance. 


([[@ronenMachineLearningTrade2022]]) use a random test split and $k$-fold cross-validation. As sample
We perform adversarial validation t 


Angsichts der

A second question is,


- What do previous works do? 
- <mark style="background: #BBFABBA6;">What is enough data?</mark>
- covariance shift
- 

Parts of our trades are unlabelled, as .

Recall from earlier, that only for a subset of the dataset the labels are known.


We perform 


Data may however be shuffled in the subsets.

Data within folds may however be shuffled to to accelerate training.

<mark style="background: #ABF7F7A6;">You don't randomly split in time-series datasets because it doesn't respect the temporal order and causes _data-leakage_, e.g. unintentionally inferring the trend of future samples.</mark>

<mark style="background: #BBFABBA6;">In machine learning, train/test split splits the data randomly, as there’s no dependence from one observation to the other. That’s not the case with time series data. Here, you’ll want to use values at the rear of the dataset for testing and everything else for training.</mark>

https://gsarantitis.wordpress.com/2020/04/16/data-shift-in-machine-learning-what-is-it-and-how-to-detect-it/
https://datascience.stanford.edu/news/splitting-data-randomly-can-ruin-your-model
https://scikit-learn.org/stable/modules/cross_validation.html#timeseries-cv

![[accuracies.png]]
![[train-test-split.png]] ^a92764

We pre-train a model on unlabelled samples from the last year of the training period, as depicted in Fig-[[#^a92764]]. Given the significantly larger number of unlabelled customer trades, the pre-training period is reduced to one year to facilitate training on the available computing resources. Within the period, we filter out trades where the true label can be inferred, to avoid overlaps with the supervised training set. This is essential for self-training, as labelled and unlabelled data are fed to the model simultaneously. 




**Notes:**
[[👨‍🍳Train-Test-split notes]]