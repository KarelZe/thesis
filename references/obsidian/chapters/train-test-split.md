- https://www.coursera.org/learn/machine-learning-projects#syllabus

**How:**
We perform a split into three disjoint sets.
**Sets:**
- Training set is used to fit the model to the data
- Validation set is there for tuning the hyperparameters. [[@hastietrevorElementsStatisticalLearning2009]] (p. 222) write "to estimate prediction error for model selection"
- Test set for unbiased, out-of-sample performance estimates. [[@hastietrevorElementsStatisticalLearning2009]] write "estimate generalization error of the model" (p. 222)
- Common splitting strategy should be dependent on the training sample size and signal-to-noise ratio in the data. [[@hastietrevorElementsStatisticalLearning2009]] (p. 222)
- A common split is e g. 50-25-25. [[@hastietrevorElementsStatisticalLearning2009]] (p. 222)
- We use a 60-20-20 split, and assign dates to be either in one set to simplify evaluation.
- Forego with the idea of a dynamic retraining.

**Why:**
The split is required to get unbiased performance estimates of our models. It is not required for classical rules, as these rules have no parameters to estimates or hyperparameters to tune.
To facilitate a fair comparsion we compare both classical rules and our machine learning approches on the common test set and neglect training and validation data for classical rules.

**Classical split over random split:**
A classical train test split is advantegous for a number of reasons:
- We maintain the temporal ordering within the data and avoid data leakage: e. g., from unknown `ROOT`s, as only trailing observations are used. (see similar reasoning in [[@lopezdepradoAdvancesFinancialMachine2018]] for trading strategies).
- The train set holds the most recent and thus most relevant observations.
- Work of [[@grauerOptionTradeClassification2022]] showed that the classification performance deterioriates over time. Thus, most recent data poses the most rigorous test conditions due to the identical data basis.
- Splitting time-correlated data randomly can bias the results and correlations are often non-obvious e. g., `ROOT`s, advent of etfs. She advocates to split data *by time* to avoid leakage (See [[@huyenDesigningMachineLearning]] (p. 137)).
- [[@ronenMachineLearningTrade2022]] performed a 70-30 % random split. This can be problematic for obvious reasons.
**Classical split over CV:**
- computational complexity
- Observations in finance are often not iid. The test set is used multiple times during model development resulting in a testing and selection bias [[@lopezdepradoAdvancesFinancialMachine2018]]. Serial correlation might be less of an issue here.
- use $k$ fold cross validation if possible (see motivation in e. g. [[@banachewiczKaggleBookData2022]] or [[@batesCrossvalidationWhatDoes2022]])
- A nice way to visualize that the models do not overfit is to show how much errors vary across the test folds.
- On cross-validation cite [[@batesCrossvalidationWhatDoes2022]]
**Moving window:**
- Why no moving window. Reason about computational complexity.
**Evaluate similarity of train, test and validation set:**
- Perform [[adversarial_validation]] or https://medium.com/mlearning-ai/adversarial-validation-battling-overfitting-334372b950ba. More of a practioner's approach than a scientific approach though. 
- discuss how split is chosen? Try to align with other works.
- compare distributions of data as part of the data analysis?
- Think about using a $\chi^2$ test to estimate the similarity between train and test set. Came up with this idea while reading [[@aitkenIntradayAnalysisProbability1995]]. Could help finding features or feature transformations that yield a similar train and test set.
- Write how target variable is distributed in each set. 
 Show that a stratified train-test-split is likely not necessary to maintain the distribution of the target variable.
- Plot learning curves to estimate whether performance will increase with the number of samples. Use it to motivate semi-supervised learning.  [Plotting Learning Curves — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) and [Tutorial: Learning Curves for Machine Learning in Python for Data Science (dataquest.io)](https://www.dataquest.io/blog/learning-curves-machine-learning/)
![[learning-curves-samples 1.png]]
