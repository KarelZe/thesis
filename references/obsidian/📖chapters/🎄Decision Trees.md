
Decision trees can be used in classification and regression. Counterintuitive to our initial problem framing of trade classification as a probabilistic classification task ([[🍪Selection Of Supervised Approaches]]), the focus is on regression trees only, as it is the prevailing prediction model used within the gradient boosting algorithm ([[@friedmanAdditiveLogisticRegression2000]]9). The ensemble method later adapts to classification.

A decision tree splits the feature space $\mathbb{R}^p$ into several disjoint regions $R$ through a sequence of recursive splits. For a binary decision tree, a single split leads to two new sub-regions, whose shape is determined by the features considered for splitting and the preceding splits. Trees are grown in depth until a minimum threshold for the number of samples within a node or some other stopping criterion applies ([[@breimanClassificationRegressionTrees2017]]42). 

A region corresponds to a terminal node in the tree. For each terminal node of the tree or unsplit region, the predicted response value is constant for the entire region and shared by all its samples  ([[@breimanClassificationRegressionTrees2017]] 229). 
For a tree with $M$ regions $R_1, R_2,\ldots, R_M$,  and some numerical input $x$ the tree can be modelled as: $$f(x)=\sum_{m=1}^{M} c_{m} \mathbb{I}\left(x \in R_{m}\right),$$where $\mathbb{I}$ is the indicator function for region conformance and $c_m$  the region's constant ([[@hastietrevorElementsStatisticalLearning2009]]326). In the regression case, $c_m$ is the mean of all target variables $y_i$ in the specific region. Since all samples of a region share a common response value, the tree estimates resemble a histogram that approximates the true regression surface.

---

<mark style="background: #FFB86CA6;">(greedy)</mark>
So far, it remains open how the best split can be found. The best split is where the deviation of all regions estimates and the true response variables diminishes. Over the entire tree, this error can be captured in the SSE given by $$E(M)=\frac{1}{N} \sum_{m \in M} \sum_{i \in R_m}\left(y_{i}-f(x_i)\right)^{2},$$ which is subsequently minimized [[@breimanClassificationRegressionTrees2017]]. 

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