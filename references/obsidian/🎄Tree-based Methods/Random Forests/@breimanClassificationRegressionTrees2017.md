
title: Classification And Regression Trees
authors: Leo Breiman, Jerome H. Friedman, Richard A. Olshen, Charles J. Stone
year: 2017
tags :  #dt #decision-trees #regression #supervised-learning 
status : #📦 
related: 
- [[@hastietrevorElementsStatisticalLearning2009]]

Referenced in: [[@prokhorenkovaCatBoostUnbiasedBoosting2018]] [[@hastietrevorElementsStatisticalLearning2009]]

## Regression Trees

## Preliminary

In regression, a case consists of data $(x, y)$ where $x$ falls in a measurement space $X$ and $y$ is a real valued number. The variable $y$ is usually called the response or dependent variable. The variables in $x$ are variously referred to as the independent variables, the predictor variables, the carriers, etc We will stay with the terminology of $x$ as the measurement vector consisting of measured variables and $y$ as the response variable. (p. 221)

A prediction rule or predictor is a function $d(x)$ defined on $X$ taking real values; that is, $d(x)$ is real-valued function on $X$. (p. 221)

## Splitting and criterion

A tree structured predictor partitions the space $X$ into a sequence of binary splits into terminal nodes. Within the terminal node $t$  the response value y(t) is constant.

**Visualization**
Due to the predictor $d(x)$ being constant over each terminal node, the tree can be thought of as a histogram estimate of the (true) regression surface (p. 229):

![[regression_surface_dt 1.png]]

In the regression case the value of $y(t)$ that minimizes $R(d)$ is the average of $y_n$ for all cases $(x_n,y_n)$ falling into a region $t$, that is, minimizing $y(t)$ is:
$$\bar{y}(t)=\frac{1}{N(t)} \sum_{x_{n} \in t} {y}_{n}$$

where the sum is over all $y_n$ such that $x_n \in t$ and $N(t)$ is the total no. of cases in that region.

The overall error between the prediction and the target is given by $R(t)$, which is calculated as (p. 231):

$$
R(T)=\frac{1}{N} \sum_{t \in \widetilde{T}} \sum_{n \in T}\left(y_{n}-\bar{y}(t)\right)^{2}
$$

## Best Split

So for each node $t$ the sum of squares is calculated, which is the total squared deviations of the $y_n$ form their nodes nodes average. Summing over all $t \in \widetilde{T}$ gives the total sum of squares. Deviding by $N$ gives the average.

A regression tree is formed by iteratively splitting nodes, so that a maximum decreae in $R(T)$ is obtained. Formally, for any split $s$ of $t$ into $t_{L}$ and $t_{R}$, let
$$
\Delta R(s, t)=R(t)-R\left(t_{L}\right)-R\left(t_{R}\right)
$$

The best split in regression, is the one that splits the $x$ variables into high response values and low response values.

Formally, the best split $s^{*}$ to be a split such that (p. 232)

$\Delta R\left(\mathfrak{s}^{*}, t \right)=\max _{\mathfrak{s} \in S} \Delta R(\Delta, t)$

## Connection between regression and binary classification trees

An alternative form of the criterion is interesting. Let $\mathrm{p}(\mathrm{t})=N(\mathrm{t}) / N$ be the resubstitution estimate for the probability that a case chosen at random from the underlying theoretical distribution falls into node t. Define
so that $R(t)=\mathrm{s}^2(t) p(t)$, and
$$
s^2(t)=\frac{1}{N(t)} \sum_{x_n \in t}\left(y_n-\bar{y}(t)\right)^2
$$
Note that $s^2(t)$ is the sample variance of the $y_n$ values in the node $t$. Then the best split of $t$ minimizes the weighted variance
$$
p_L s^2\left(t_L\right)+p_R s^2\left(t_R\right),
$$
where $p_L$ and $p_R$ are the proportion of cases in $t$ that go left and right, respectively.
<mark style="background: #D2B3FFA6;">We noted in Chapter 3 that in two-class problems, the impurity criterion given by $p(1 \mid t) p(2 \mid t)$ is equal to the node variance computed by assigning the value 0 to every class 1 object, and 1 to every class 2 object. Therefore, there is a strong family connection between two-class trees and regression trees.</mark>