
title: Random Forests
authors: Leo Breiman
year: 2001
tags :  #bagging #boosting #rf #supervised-learning #bootstrap
status : #📦 
related: 
- [[@tibshiraniElementsStatisticalLearning]]
cited in:
- [[@banachewiczKaggleBookData2022]]

## Motivation for ensembles
Growing an ensemble of trees leads to an increase in accuracy.  Popular approaches include bagging and random forests. (p. 5)

## Link between bagging and random forests

The common element in all of these procedures is that for the $k$ th tree, a random vector $\Theta_{k}$ is generated, independent of the past random vectors $\Theta_{1, \ldots,} \Theta_{k-1}$ but with the same distribution; and a tree is grown using the training set and $\Theta_{k}$, resulting in a classifier $h\left(\mathbf{x}, \Theta_{k}\right)$ where $\mathbf{x}$ is an input vector. For instance, in bagging the random vector $\Theta$ is generated as the counts in $N$ boxes resulting from $N$ darts thrown at random at the boxes, where $N$ is number of examples in the training set. (...). The nature and dimensionality of $\Theta$ depends on its use in tree construction. (p. 6)

(With Random Forests $\Theta$ would be the random sample the features considered for splitting.) (compare algo in [[@tibshiraniElementsStatisticalLearning]], p. 588)

After a large number of trees is generated, they vote for the most popular class. We call these procedures random forests. (p.6)

## Definition
A random forest is a classifier consisting of a collection of tree-structured classifiers $\left\{h\left(\mathbf{x}, \Theta_{k}\right), k=1, \ldots\right\}$ where the $\left\{\Theta_{k}\right\}$ are independent identically distributed random vectors and each tree casts a unit vote for the most popular class at input $\mathbf{x}$.

## Procedure

At each node  a random selection of features is used to determine the split. The question how many features are considered is important (see also [[@tibshiraniElementsStatisticalLearning]]). For guidance internal estimates of the generalization error, the classifier strength and the dependence are computed.


## Advantages:
- accuracy is as good as boosting approach or better
- robust to outliers or noise
- faster than bagging or boosting
- contains interal estimates of error, strength, correlation and variable importance
- simple and easily parallelizable (p. 10)


## Random Forests for regression

Random forests for regression are formed by growing trees depending on a random vector $\Theta$ such that the tree predictor $h(\mathbf{x}, \Theta)$ takes on numerical values as opposed to class labels. The output values are numerical and we assume that the training set is independently drawn from the distribution of the random vector $Y, X$. The random forest predictor is formed by taking the average over $k$ of the trees $\left\{h\left(\mathbf{x}, \Theta_{k}\right)\right\}$. (p. 25, 26)