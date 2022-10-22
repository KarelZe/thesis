
title: Gradient Boost Part 2 (of 4): Regression Details
authors: StatQuest
year: 2019
tags :  #supervised-learning #gbm #decision #gradient_boosting 
status : #📦 
related: 
- [[🎄Tree-based Methods/Gradient Boosting/@friedmanGreedyFunctionApproximation2001]]
- [[@hastietrevorElementsStatisticalLearning2009]]

- Ada-Boost only uses stumps, Gradient Boosting approaches
- Gradient Boosting starts with a single leaf instead of a tree. Leaf represents inital guess for all weights. First guess is average value
- Gradient Boosting builds a tree based on the errors
- Gradien Boosting builds trees with fixed size, but not only stumps. Gradient Boost also scales the trees based on the errors.

## Example
- First the average is calculated
- Based on the errors of the leaf a tree is built. The error is called pseudo residual. It is basically the difference between the mean and the true value. Different features are used for splitting criterion of the tree.
- The residuals in the leaf are replaced with the leaf's average.
- Learning rate is used to scale the contibution from a new tree e. g. 71.2 + learning rate * leaf's mean. Using the learning rate reduces the variance. The learning rate is the same for all trees.
- One does than calculate new residuals from the improved prediction with the learning rate. All residuals are smaller than before after adding a new tree. For each tree the branches can be different.


$x_i$s characteristcs, $y_i$s predicted values.
- loss function evaluates, how well we predict the target variable
- loss function is squared residual$\frac{1}{2} (observed-predicted)^2$. Motivation of using the fraction in front is that it scales only linearly, when applying the change rule only the minus in front remains.
- In the first step we initialize th emodel with a constant value. $\gamma$ refers to the predicted value. We start with a predicted value, taht minimizes the sum of squared residuals.
- We first take the derivative with respect to predicted for each observation, set equal to zero and solve. We end up with average of observed weights. Initial predicted value is average and just a leaf.
- We make $M$ trees. $m$ refers to the $m$-th tree.
- Compute $r_{i m}=-\left[\frac{\partial L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F\left(x_{i}\right)}\right]_{F(x)=F_{m-1}(x)}$ for $i=1, \ldots, n$ this is equal to (Observed-predicted), as the derivative is given by -1 * (observed-predicted) or because of linear scaling -0.5 * (observed -predicted). In the first iteration we plug in the mean for the residual. Using the formula we calculate all residuals. $r_{i,m}$s are called pseudoresiduals.
- One leaf in tree is $R_{1,1}$, another $R_{2,1}$.
- A new $\gamma$ is calculated if several residuals are in a region. The summation is similar to the averaging at the very beginning. If theere is just a single residual in the leaf, $\gamma$ is the residual itself. Otherwise the average of all residuals in a leaf. This has to do with the chosen loss function.
- We take the derivative with respect to $\gamma$ and set equal to 0.
- Finally, we make a new prediction for each sample using the last prediction, the learning rate $\nu$ and the  tree. We add up the output values, $\gamma$s for all the leaves, $R_{j,m}$ in which $x$ can be found in. $\nu$ is between 0 and 1. A small learning rate improves accuracy in the long run.
- We then update the prediction based on the previous predictions and the learning rate.

## Main steps

Step 2: for $m=1$ to $M:$
(A) Compute $r_{i m}=-\left[\frac{\partial L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F\left(x_{i}\right)}\right]_{F(x)=F_{m-1}(x)}$ for $i=1, \ldots, n$
(B) Fit a regression tree to the $r_{i m}$ values and create terminal regions $R_{j m}$, for $j=1 \ldots J_{m}$
(C) For $j=1 \ldots J_{m}$ compute $\gamma_{j m}=\underset{\gamma}{\operatorname{argmin}} \sum_{x_{i} \in R_{i j}} L\left(y_{i}, F_{m-1}\left(x_{i}\right)+\gamma\right)$
(D) Update $F_{m}(x)=F_{m-1}(x)+\nu \sum_{j=1}^{J_{m}} \gamma_{j m} I\left(x \in R_{j m}\right)$