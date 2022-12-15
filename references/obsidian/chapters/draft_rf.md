# Random Forests

As stated previously, decision trees suffer from a number of drawbacks. One of them being prone to overfitting. To mitigate the high variance, several trees can be combined to form an *ensemble* of trees. The final prediction is then jointly estimated from all member within the ensemble.

Popular ensemble methods for trees include bagging and random forests [[🎄Tree-based Methods/Random Forests/@breimanRandomForests2001]]. Also, Boosting is another approach that learns a sequence base learners such as simplified decision trees. [[@hastietrevorElementsStatisticalLearning2009]] (p. 587) We present boosting as part of section [[🐈gradient-boosting]].

Bagged trees and Random Forests have in common to learn $B$ independent trees. However, they differ in whether learning is performed on a random subset of data or whether splits consider only a portion of all features.

Let's consider the regression case only. With bagging each tree is trained on a random subset of data, drawn with replacement from the training set. Learning a predictor on a so-called bootstraped sample, still causes the single tree to overfit. Especially, if trees are deep. Pruning the trees, selecting the best performing ones and averaging their estimates to a bagged predictor, helps to improve the accuracy [[@breimanBaggingPredictors1996]].

Besides this, averaging the estimates of several trees, bagging maintaines the desirable low-bias property of a single tree, assuming trees are grown large enough to capture subtleties of the data, while also improving on the variance. [[@hastietrevorElementsStatisticalLearning2009]]

Yet, one issue bagging can not resolve is, that bagged trees are not independent [[@hastietrevorElementsStatisticalLearning2009]]. This is due to the limitation, that all trees select their best split attributes from the same set of features. If features dominate in the bootstrap samples, they will yield similiar splitting sequences and thus highly correlated trees.

The variant of Bagging named *Random Forests* addresses the high correlation among trees, by considering only a random subset of all features for splitting. Random forests for regression, as introduced by [[🎄Tree-based Methods/Random Forests/@breimanRandomForests2001]]  consist of $B$ trees, that are grown in parallel to form a forest. At each split only a random subset of $m$ features is considered for splitting. Typically, $m$ is chosen to be the $\sqrt{p}$ of all $p$ input variables [[@hastietrevorElementsStatisticalLearning2009]].

The random forest predictor is then estimated as the average overall the set of $\left\{T\left(x ; \Theta_{b}\right)\right\}_{1}^{B}$  trees:
$$
\hat{f}_{\mathrm{rf}}^{B}(x)=\frac{1}{B} \sum_{b=1}^{B} T\left(x ; \Theta_{b}\right),
$$

with $\Theta_{b}$ being a parameter vector of the  $b$-th tree [[@hastietrevorElementsStatisticalLearning2009]].

As the variables considered for splitting differ from one split and one tree to another, the trees are less similar and hence correlated. Random Forests achieve a comparable accuracy to Boosting or even outperform them. As trees do not depend on previously built trees, they can be trained in parallel. These advantages come at the cost of of lower interpretability compared to decision trees. [[🎄Tree-based Methods/Random Forests/@breimanRandomForests2001]]

In the next section we discuss Boosting approaches, that grow trees in an adaptive manner.






