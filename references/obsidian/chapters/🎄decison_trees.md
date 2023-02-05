A decision tree  splits the feature space $\mathbb{R}^m$ into several disjoint regions $R$ through a sequence of recursive splits. A single split leads to two new sub-regions for a binary decision tree, whose shape depends on the attribute used for splitting and the previously performed splits. The tree is grown in size until a minimum threshold for the number of samples within a node or some other stopping criterion applies [[@breimanClassificationRegressionTrees2017]] (p. 42). Trees can be utilised for classification and regression tasks. 
Dispite our overall focus on (trade) classification, only the regression variant is introduced, as its the variant used in the *gradient boosting algorithm* [[@friedmanAdditiveLogisticRegression2000]] (p. 9). 

A region corresponds to a node in the tree. For each terminal node of the tree or unsplit region, the response variable $f(x)$ is constant [[@breimanClassificationRegressionTrees2017]] (p. 229). For a tree with $M$ regions, the response for the numerical input $x$ can be modelled as $$f(x)=\sum_{m=1}^{M} \gamma_{m} \mathbb{I}\left(x \in R_{m}\right),$$ with $\mathbb{I}$ being the indicator function for region conformance and $\gamma_m$ being the region's constant [@hastietrevorElementsStatisticalLearning2009]. In the regression case, $\gamma_m$ is simply the average over all response variables of this particular region. As $\gamma_m$ is shared among all samples within the region, the estimates of the tree are similar to a histogram approximating the true regression surface. 

So far, it remains open how the best split can be found. The best split is where the deviation of all regions estimates and the true response variables diminishes. Over the entire tree, this error can be captured in the SSE given by $$E(M)=\frac{1}{N} \sum_{m \in M} \sum_{i \in R_m}\left(y_{i}-f(x_i)\right)^{2},$$ which is subsequently minimized [@breimanClassificationRegressionTrees2017]. 

<mark style="background: #FF5582A6;">TODO: Add link between binary classification and regression as footnote? See [[@breimanClassificationRegressionTrees2017]]. </mark>

Following [[@breimanClassificationRegressionTrees2017]] we scan all combinations of possible $m$ and potential split values $s$ and choose the split, that leads to the largest improvement in the error of the child nodes compared to their parent node $E(m)$: $$\Delta E(s, m)=E(m)-E\left(m_{l}\right)-E\left(m_{r}\right).$$
<mark style="background: #FF5582A6;">TODO: Introduce the term regularization -> constraint / limit the complexity of the model. -> impose constraints on the depth of the tree</mark>

<mark style="background: #FF5582A6;">FIXME: Introduce idea of bias and variance trade-off more throughly using [[@hastietrevorElementsStatisticalLearning2009]]; a catchy definition can also be found in [[@schapireBoostingMarginNew1998]])</mark>

<mark style="background: #ABF7F7A6;">TODO: Why do we use quantization in practice?</mark> On quantization in regression trees see [[@shiQuantizedTrainingGradient2022]]

Trivially, growing deeper trees leads to an improvement in the SSE. Considering the extreme, where each sample is in its region, the tree would achieve the highest fit in-sample but perform poorly on out-of-sample data. To reduce the sensitivity of the tree to changes in the training data, hence *variance*, *cost complexity pruning* procedures are employed. Yet, if the decision tree is too simplistic, consequently underfits the data, a high bias contributes to the model's overall expected error. 

<mark style="background: #FF5582A6;">FIXME: Make drawbacks clearer. Trees are grown greedily. Thus, only the current split is being considered. </mark>

<mark style="background: #FF5582A6;">FIXME: More attractive opening: The prediction accuracy can be further enhanced by growing an ensemble of decision trees [[@breimanRandomForests2001]]. </mark>

The expected error of the tree can be decreased through combining multiple trees in an ensemble. Two popular approaches include *bagging* [[@breimanBaggingPredictors1996]] and *boosting* [[@schapireStrengthWeakLearnability1990]]. Both differ with regard to the error term being minimized, which is both reflected in the training procedure and complexity of the ensemble members. Most notably, bagging aims at decreasing the variance, whereas boosting addresses both bias and variance [[@schapireBoostingMarginNew1998]] (p. 1672) (also [[@breimanRandomForests2001]] (p. 29).

Next, ([[🐈Gradient Boosting]]) we derive gradient boosting, a variant of boosting introduced by [[@friedmanGreedyFunctionApproximation2001]], for the binary classification case. Random Forests [[@breimanRandomForests2001]], a type of bagging is briefly covered in our model discussion in section (...).