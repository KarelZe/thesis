eschews data leakage problems



The datasets (see cref-[[🌏Dataset]]) are split into three disjoint sets. The training set is used to fit the model to the data. The validation set dedicated to tuning the hyperparameters, and we reserve the test set for unbiased out-of-sample estimates. In absence of tunable hyperparameters, prior classical works test their classical rules in-sample (cp. [[@ellisAccuracyTradeClassification2000]]541) on in true out-of-sample settings (cp. [[@grauerOptionTradeClassification2022]]7--9) and ([[@chakrabartyTradeClassificationAlgorithms2007]]3814-3815). 

The test the gist for is more 
Data is however , which would severle
This makes the hold-out method highly 

 a strict temporal ordering must be maintained when splitting the data so that models can capture the temporal properties of the data and leakage between sets is minimised. 
Our data is generated through a temporal process, as such samples are not statistically independent. This outrules methods like the $k$-fold cross-validation or random test splits to arrive at suitable subsets, both of which assume samples to be i.i.d. ([[@lopezdepradoAdvancesFinancialMachine2018]] 104). 

As such leakage, 
 . Leakage takes place when the training set contains information that also appears in the testing set.
While previous works employed both techniques (split random train-test splits and $k$-fold-cross-validation. ) ([[@ronenMachineLearningTrade2022]]),  we use the conceptually  

This is especially important when data has high autocorrelation. Autocorrelation among points simply means that value at a point is similar to values around it. Take temperature for instance. Temperature at any moment is expected to be similar to the temperature in the previous minute. Thus, if we wish to predict temperature, we need to take special care in splitting the data. Specifically, we need to ensure that there is no data leakage between training, validation, and test sets that might exaggerate model performance. 

Serial dependence occurs when the value of a datapoint at one time is statistically dependent on another datapoint in another time. However, this attribute of time series data violates one of the fundamental assumptions of many statistical analyses  that data is statistically independent. (https://www.influxdata.com/blog/autocorrelation-in-time-series-data/)

 However, the random selection of test observations does not warrant independence from training observations when dependence structures exist in the data, i.e., when observations close to each other tend to have similar characteristics

While our train-test split minimizes data leakage, it We subsequently address these aspects to underpin our choice for the training split. 
The performed split is notably different from previous works. 


We use a classical train-test split to 
([[@ronenMachineLearningTrade2022]]) use a random test split and $k$-fold cross-validation. As sample

The used split deviates from previous works and is motivated by two observations from the [[🚏Exploratory Data Analysis]]:

A static training scheme is computationally cheap, but fails to leverage recent information for the prediction beyond the training set's cut-off point. One such example, are new underlyings as observed in cref-[[🚏Exploratory Data Analysis]]. An alternative are rolling scheme, that incorporates recent data through training a model on updated sets, as employed in ([[@ronenMachineLearningTrade2022]] 16). In a rolling scheme, fixed-size training and validation windows gradually shift forward in time, thereby dropping older observations and incorporating newer ones. However, due to the large number of model combinations and computational requirements of  Transformers and gradient-boosted trees, a rolling window approach becomes practically intractable. In absence of an update mechanism, our results can be interpreted as a lower bound.

Applying the holdout method we use the first 60 % of our dataset for training and the next 20 % for validation and testing. Samples of one day are assigned to either one set for precise evaluation and the temporal ordering is maintained to avoid data leakage. Data within the training set can be permuted to accelerate training.

![[train-test-split.png]] ^a92764 

Overall,  we use gls-ISE data from 2 May 2005 to 24 October 2013 to train and data between 25 October 2013 and 5 November 2015 to validate our models. The most recent trades until 31 May 2017 to assess the generalization error. The timespans for the gls-CBOE sample are adjusted accordingly. Here, the sets go from 1 January 1974 to 1 January 1974, 1 January 1974 to 1 January 1974, and 1 January 1974 to 1 January 1974, respectively, as visualized in Fig-[[#^a92764]].

We pre-train a model on unlabelled samples from the last year of the training period, as depicted in Fig-[[#^a92764]]. Given the significantly larger number of unlabelled customer trades, the pre-training period is reduced to one year to facilitate training on the available computing resources. Within the period, we filter out trades for which true label can be inferred, to avoid overlaps with the supervised training set. This is essential for self-training, as labelled and unlabelled data are provided to the model simultaneously. 

Our train-test-split, however, makes two implicit assumptions, we want to test for. First, the size of the resulting training set is large enough, for the model to capture regularities in the data. Second, all subsets are drawn from the same distribution. So that, fitting the classifier on the training set and optimizing on the validation set can provide good estimates for the test set.  We subsequently analyze both aspects to underpin our choice. The analysis is conducted on the training and validation set to avoid information leaking from the test set.

Optimally, samples in the train, validation, and test come from the same distribution. The presence of the data shift, as observed in cref-[[🚏Exploratory Data Analysis]] within the training set raises concerns that the assumption holds. We test for the similarity of the training and validation set and identify problematic features using *adversarial validation*. As such, we re-label all trades within the training set with 0 and all trades of the validation set with 1. We then train a classifier on a random subset of the composed data and predict the conformance to the train or validation set for the remaining samples. More specifically, we use a gradient boosting classifier, which gives competitive predictions with default hyperparameters and is least computationally demanding. As samples in the training set are more frequent than validation samples, performance is estimated using the gls-ROC-AUC, which is insensitive to class imbalances. Assuming train and test samples come from the same distribution, the obtained performance estimate is near a random guess.

![[adv-validation-gradient-boosting.png]]
(describe plot but use different measure)

To study the appropriateness of the size of the training set, we study the *learning curves* of a classifier. Learning curves visualize the score of the classifier dependent on the size of the training set ([[@hastietrevorElementsStatisticalLearning2009]]243).  Moreover, learning curves provide insights into the bias and variance of the estimators. We train a gradient boosting model with default parameters on ten subsets of the training set and evaluate on the validation set. To maintain the temporal ordering, we start training of the 10 % most recent samples and add older observations as we progress. Gradient boosting is well-suited for this analysis for the reasons given above. 

![[learning-curves.png]]

From (cref-fig) (a) several observations can be drawn. Adding more training instances will likely improve the accuracy of the training set. The low training error also indicates that the chosen model is sufficiently complex to fit the data, resulting in a low bias. The accuracy for the validation set plateaus at 75.53 % after (...) samples, leaving a significant gap between the training and testing accuracy, indicating a high variance. Hence, adding older instances, before 1 January 1974, to the training set, improves the training performance, but merely affects validation performance when all samples are assigned an equal weight. Naturally, patterns in older observations of the training set might not be as relevant as more recent ones for classifying trades out-of-sample, which could explain the stalling performance for extended training sets. We incorporate this idea by weighting training samples with decaying weights over time. Results are shown in (cref-fig) (b) exemplary for an exponential weighting. Exponential weighting improves both overfitting and generalization performance as derived from the annealing of train and test scores and the overall improved performance with accuracies up to 75.76 %. <mark style="background: #FFB86CA6;">One question learning curves can not dissolve, is how much of the generalization error is irreduable.</mark>

While a smaller subsample of the training set would suffice in both cases, our training set encompasses the entire history to mitigate any sampling bias and we instead weight observations based on their temporal proximity, so that more recent observations have greater importance in the training set. From the high bias/low variance, we conclude that the gradient-boosting model with default parameters severely overfits the data. To address the overfitting, we emphasize regularization techniques in the training procedure as we show next. 

**Notes:**
[[👨‍🍳Train-Test-split notes]]

