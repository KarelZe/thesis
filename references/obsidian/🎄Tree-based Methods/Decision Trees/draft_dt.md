# Decision Trees

A decision tree splits the feature space $\mathbb{R}^m$  into several disjoint regions $R$ through a sequence recursive splits. For a binary decision tree a single split leads to two new sub-regions, with their shape being dependent on the attribute used for splitting and the previously performed splits. The tree is grown in size until a minimum threshold for the number of samples within a node or some other stopping criterion applies. ([[@breimanClassificationRegressionTrees2017]][[@tibshiraniElementsStatisticalLearning]]) Trees are applicable to both classification and regression tasks. Our introduction focuses on the regression setting only.

A region corresponds to a node in the tree. For each terminal node of the tree or unsplit region, the response variable $f(x)$ is constant.  ([[@breimanClassificationRegressionTrees2017]]) That is, for a tree with $M$ regions the response for $x$ can be modelled as:
$$
f(x)=\sum_{m=1}^{M} c_{m} \mathbb{1}\left(x \in R_{m}\right),
$$
with $\mathbb{1}$ being the indicator function for region conformance and $c_m$ being the region's constant. [[@tibshiraniElementsStatisticalLearning]]  For a regression case $c_m$ is simply the average over all response variables of this particular region. As $c_m$ is shared among all samples within the region, the estimates of the tree are similar to a histogram approximating the true regression surface, as visualized below:

![[regression_surface_dt.png]] (see [[@breimanClassificationRegressionTrees2017]])

So far it remains open, how the best split can be found. From the visual representation it can be seen, that the best split is the one, where the deviation of all regions estimates and the true response variables diminishes. This can be captured for the entire tree in the sum of squares errors given by:

$$
R(T)=\frac{1}{N} \sum_{t \in \widetilde{T}} \sum_{n \in T}\left(y_{n}-\bar{y}(t)\right)^{2},
$$

which is subsequently minimized [[@breimanClassificationRegressionTrees2017]].  Following [[@breimanClassificationRegressionTrees2017]] we scan all combinations of split variable $t$ and potential split values $s$ and choose the split, that leads to the largest improvement between the error of the parent node $R(t)$ its child nodes:

$$
\Delta R(s, t)=R(t)-R\left(t_{L}\right)-R\left(t_{R}\right).
$$

Growing deeper trees trivially leads to an improvement in the SSE. Considering the extreme, where each sample is it's own region, the tree would achieve the highest fit, but would perform poorly on unseen data. To obtain trees, that generalize,  *cost complexity pruning* procedures are employed. An introduction is given in ... .


(close approx., small sum of squared errors, finding global minium is not always feasabile)

- Concept of decision tree
- Describe drawback of decision tree
	- Growing an ensemble of trees leads to an increase in accuracy [[🎄Tree-based Methods/@breimanRandomForests2001]].  Popular approaches include bagging and random forests.
	- There are multiple ways how to build $k$ trees on a single data set e. g. perform bootstrapping, randomly split, use random subset of features [[🎄Tree-based Methods/@breimanRandomForests2001]] (see section: link between bagging and boosting)
	- Derive how bagging is the intuitive extension to standard decision trees (see discussion in [[@breimanBaggingPredictors1996]]).
	- Introduce idea of bootstrapping in bagging [[@breimanBaggingPredictors1996]].
- Introduce notion of strong learner
- Transfer to Gradient Boosting
	- What problem does it solve, that Random Forests can not solve?
