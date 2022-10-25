
title: CatBoost: unbiased boosting with categorical features
authors: Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin
year: 2019
tags :  #supervised-learning #catboost #gbm #decision #target-leakage #gradient_boosting 
status : #📦 
related: 
- [[🎄Tree-based Methods/Gradient Boosting/@friedmanGreedyFunctionApproximation2001]]
- [[@hastietrevorElementsStatisticalLearning2009]]
# Notes

## Gradient Boosting
Gradient boosting is essentially a process of constructing an ensemble predictor by performing gradient descent in a functional space. (p. 1)

## Problems of Gradient boosting
- A prediction model $F$ obtained after several steps of boosting relies on the targets of all training examples, this can leads to a model shift for a training example and hence to a prediction shift of the learned model.
- Prior approaches used for handling categories is converting categories to their target statistics. A target statistic is a statistical model itself, it can also cause target leakage and a prediction shift.

## Novelty

- **Ordering principle:** Solves the *prediction shift*.
- **Algorithm to process categorical features:** Adresses the target leakage and prediction shift from above.


## Decision Trees

- A decision tree is built by a recursive partition of the feature space $\mathbb{R}^m$ into several disjoint regions (*tree nodes*) according to the values of some splitting attirbute $a$.
- Attributes are usually binary variables that identify that some feature $x^{k}$ exceeds some *threshold* that is, $a=\mathbb{1}_{\left\{x^{k}>t\right\}}$, where $x^{k}$ is either numerical or binary feature, in the latter case $t=0.5$. Other variants exist to perform non-binary splits e. g. for categorical features.
- Each final region (leaf of the tree)is assigned to a value, which is an estimate of the response $y$ in the region for the regression task or the predicted class label in the case of a classification problem.
- In this way a decision tree $h$ can be written as:
$$
h(\mathrm{x})=\sum_{j=1}^{J} b_{j} \mathbb{1}_{\left\{\mathrm{x} \in R_{j}\right\},}
$$
where $R_{j}$ are the disjoint regions corresponding to the leaves of the tree. (p. 2) ($b_j$ is probably some constant / weight.)

They cite [[@hastietrevorElementsStatisticalLearning2009]] and [[@breimanClassificationRegressionTrees2017]].

## Gradient Boosting

A gradient boosting procedure builds iteratively a sequence of approximations $F^{t}: \mathbb{R}^{m} \rightarrow \mathbb{R}$, $t=0,1, \ldots$ in a greedy fashion. Namely, $F^{t}$ is obtained from the previous approximation $F^{t-1}$ in an additive manner: $F^{t}=F^{t-1}+\alpha h^{t}$, where $\alpha$ is a step size and function $h^{t}: \mathbb{R}^{m} \rightarrow \mathbb{R}$ (a base predictor) is chosen from a family of functions $H$ in order to minimize the expected loss:
$$
h^{t}=\underset{h \in H}{\arg \min } \mathcal{L}\left(F^{t-1}+h\right)=\underset{h \in H}{\arg \min } \mathbb{E} L\left(y, F^{t-1}(\mathrm{x})+h(\mathrm{x})\right) .
$$
The minimization problem is usually approached by the Newton method using a second-order approximation of $\mathcal{L}\left(F^{t-1}+h^{t}\right)$ at $F^{t-1}$ or by taking a (negative) gradient step. Both methods are kinds of functional gradient descent. In particular, the gradient step $h^{t}$ is chosen in such a way that $h^{t}(\mathrm{x})$ approximates $-g^{t}(\mathrm{x}, y)$, where $g^{t}(\mathrm{x}, y):=\left.\frac{\partial L(y, s)}{\partial s}\right|_{s=F^{t-1}(\mathbf{x})} .$ Usually, the least-squares approximation is used:
$$
h^{t}=\underset{h \in H}{\arg \min } \mathbb{E}\left(-g^{t}(\mathrm{x}, y)-h(\mathrm{x})\right)^{2}.
$$
They cite [[🎄Tree-based Methods/Gradient Boosting/@friedmanGreedyFunctionApproximation2001]]. (p. 2)

## Link between Gradient Boosting and Decision Trees

In Gradient Boosting a a tree is constructed to approximate the negative gradient, so it solves a regression problem. (p. 2)