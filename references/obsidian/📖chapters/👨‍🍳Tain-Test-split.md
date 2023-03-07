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

The used split deviates from previous works and is motivated by two observations from the [[🚏Exploratory Data Analysis]]:

- What do previous works do? 
- <mark style="background: #BBFABBA6;">What is enough data?</mark>
- covariance shift
- 

This makes the hold-out method highly 

Parts of our trades are unlabelled, as .

Recall from earlier, that only for a subset of the dataset the labels are known.

We perform 

Data may however be shuffled in the subsets.

Data within folds may however be shuffled to to accelerate training.

Prior works used ... the presence of. 

Data is however , which would severley 

([[@hastietrevorElementsStatisticalLearning2009]]222) recommend the dataset into subsets based on the signal-to-noise-ratio in the data and the training sample size. Following common practice, we initially use 60 %  for training and 20 % each for validation and testing. Samples of one day are assigned to either one set to simplifying evaluation and the temporal ordering is maintained to avoid data leakage. Data within the training set may however be permuted to accelerate training.

![[train-test-split.png]] ^a92764 

Overall,  we use gls-ISE data from 2 May 2005 to 24 October 2013 to train and data between 25 October 2013 and 5 November 2015 to validate our models. The most recent trades until 31 May 2017 to assess the generalization error. The timespans for the gls-CBOE sample are adjusted accordingly. Here, the sets go from 1 January 1974 to 1 January 1974, 1 January 1974 to 1 January 1974, and 1 January 1974 to 1 January 1974, respectively, as visualized in Fig-[[#^a92764]].

We pre-train a model on unlabelled samples from the last year of the training period, as depicted in Fig-[[#^a92764]]. Given the significantly larger number of unlabelled customer trades, the pre-training period is reduced to one year to facilitate training on the available computing resources. Within the period, we filter out trades for which true label can be inferred, to avoid overlaps with the supervised training set. This is essential for self-training, as labelled and unlabelled data are provided to the model simultaneously. 

Question

Still, we want to Given our observations ragarding the data shift in Section [[🚏Exploratory Data Analysis]] we want to verify the appropriateness of the chosen split. 

We conduct this analysis on the training and validation set to avoid information leaking from the test set.
-  data shift + s

two sanity checks.

We verify the appropriateness of the split by studying the learning 

![[learning-curves-gbm.png]]

We employ adversarial validation to the



The next Section presents the training procedures. Our focus is on hyperparameter tuning on the validation set.

**Notes:**
[[👨‍🍳Train-Test-split notes]]