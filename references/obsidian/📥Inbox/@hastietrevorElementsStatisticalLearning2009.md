
title: The Elements of Statistical Learning
authors: Sami Tibshirani, Harry Friedman, Trevor Hastie
year: 2009
tags :  #bagging, #decision-trees, #dt, #random-forest #gbm #gbm #cost-complexity-pruning
status : #📦 
related: 
- [[@jamesIntroductionStatisticalLearning2013]] (simpler)

## Decision Trees

Tree-based methods partition the feature space into a set of rectangles, and then fit a simple model (like a constant).

With binary trees binary splits are performed to recursively split the space into two regions.

In a regression problem for the input variables $X_1$ and $X_2$  the continous response $Y$ is calculated as the mean of $Y$ of each region, after splitting into two distcint regions.

The variable (dimension) and split-point is chosen that achieves a best fit. A recursive splitting procedure is employed until some stopping rules applies.

Calculating the prediction for a a region $R_m$ goes like:

### Decision trees for regression

The data set consist of $p$ inputs and a response, for each of $N$ observations, we have a tuple: $\left(x_{i}, y_{i}\right)$ for $i=1,2, \ldots, N$, with $x_{i}=\left(x_{i 1}, x_{i 2}, \ldots, x_{i p}\right)$.

The algorithm (e. g. CART) decides on the splitting variables, split points and also what topology the tree should have.

For a tree, that splits a partition into $M$ regions  $R_{1}, R_{2}, \ldots, R_{M}$, and we model the response $f(x)$ as a constant $c_{m}$ in each region:
$$
f(x)=\sum_{m=1}^{M} c_{m} \mathbb{1}\left(x \in R_{m}\right),
$$

with $\mathbb{1}$ being the indicator function of affilation to a region and $c_m$ being a constant.

In the regression context the **sum of squares** defined as:
$\sum\left(y_{i}-f\left(x_{i}\right)\right)^{2}$ should be minimized. The best $\hat{c}_{m}$ is just the average of $y_{i}$ in a region $R_{m}$:
$$
\hat{c}_{m}=\operatorname{ave}\left(y_{i} \mid x_{i} \in R_{m}\right).
$$

(probably derived through by calculating the derivative with respect to $\hat{y}$ and setting it to zero.)

It is computationally not feasible to find the best binary portion that, yields the minimum of the sum of squares. Therefore, a greedy alogorithm is used. Starting with all of the data, a splitting variable $j$ and split point $s$ are considered and define the pair half-planes:
$$
R_{1}(j, s)=\left\{X \mid X_{j} \leq s\right\} \text { and } R_{2}(j, s)=\left\{X \mid X_{j}>s\right\}
$$
Then we seek the splitting variable $j$ and split point $s$ that solve
$$
\min _{j, s}\left[\min _{c_{1}} \sum_{x_{i} \in R_{1}(j, s)}\left(y_{i}-c_{1}\right)^{2}+\min _{c_{2}} \sum_{x_{i} \in R_{2}(j, s)}\left(y_{i}-c_{2}\right)^{2}\right] .
$$
For any choice $j$ and $s$, the inner minimization is solved by
$$
\hat{c}_{1}=\operatorname{ave}\left(y_{i} \mid x_{i} \in R_{1}(j, s)\right) \text { and } \hat{c}_{2}=\operatorname{ave}\left(y_{i} \mid x_{i} \in R_{2}(j, s)\right) \text {. }
$$

For ech splitting variable, the determination of the split point $s$ can be done quickly by scanning through all of the inputs, determination of the best pair $(j, s)$ is feasible. (p. 307)



### Cost complexity pruning

Growing a very large tree can lead to overfitting the data, while a small tree might not capture the important structure (p. 307)

Overfitting is addressed by reducing the size of the tree. First, a large tree $T_0$ is grown, splitting is only stopped once a minimum node size (say 5) is reached. Then some *cost-complexity pruning* is applied. (p. 308)

## Bagging

Bagging was introduced in [[@breimanBaggingPredictors1996]]

The main caveat of Bagging is that bagged trees are not independent. Random forests improve on bagging by reducing the correlation between the sampled trees. (p. 286)

Bagging a model leads to a loss of the simple structure of a model e. g. a A bagged tree doesn't conform to a tree any longer.  (p. 286)
(See also discussion in [[@breimanBaggingPredictors1996]]).

Bagging addresses the variance of an estimated prediction function. It is therefore useful for reducing the variance of an estimated prediction function. Works well for high-variance and low-bias procedures like trees. In the **regression case** the same decision tree is fitted to a bootstrap-sampled version of the training data and averaging is done on the result.

Bagging average many noisy but approximately unbiased models, and thereby reduce the variance. Trees are well suited for bagging, since they capture complex interaction structures in data and also have a low bias, if trees are grown to a certain depth (p. 588)

Bagging doesn't affect the bias of the an average of $B$ trees as the expection of an average is the same as of any of them. This is in contrast to boosting, where the trees are grown in an adaptive way to remove bias, and hence are not i.d. (587)

## Boosting

Boosting was proposed as an ensemble method. Although unlike bagging, the ensemble of weak lerners evolves over time, and the members cast a weighted vote. Boosting appears to dominate bagging. (p. 587)

In Boosting trees are not i. d., as they are grown in an adaptive way (sequentially?).


## Gradient Boosting models

Motivation of boosting is to combine the output of several weak learners to form a powerful comitee / ensemble.

### AdaBoost

In classification the purpose of boosting is to sequentially apply a weak classification algorithm (that is one that is only slightly better than random guessing) and repeadly apply it to modified versions of the data. Thereby a sequence of weak classifiers $G_{m}(x), m=1,2, \ldots, M$ is generated. The predictions of all classifiers $G(x)$ are than combined through a weighted majority vote:

$$
G(x)=\operatorname{sign}\left(\sum_{m=1}^{M} \alpha_{m} G_{m}(x)\right) \text {. }
$$

Here $\alpha_{1}, \alpha_{2}, \ldots, \alpha_{M}$ are computed by the boosting algorithm, and weight the contribution of each respective $G_{m}(x)$.

Weights $w_{1}, w_{2}, \ldots, w_{N}$ are applied to the training observations $\left(x_{i}, y_{i}\right), i=1,2, \ldots, N$. Starting with equally distributed weights, the weights are modified and the classification algorithm will be reapplied. For previously (previous classifier) correctly classified samples, the weights will be decreased and for missclassified samples increased. The successive classifier is therefore forced to focus on training observations previously missclassified in the sequence. (p. 338)

### Boosting as additive expansion

Boosting is a way of fitting an additive expansion in a set  of elementary "basis functions". Generally, basis function expansions take the form
$$
f(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right),
$$
where $\beta_{m}, m=1,2, \ldots, M$ are the expansion coefficients, and $b(x ; \gamma) \in \mathbb{R}$ are usually simple functions of the multivariate argument $x$, characterized by a set of parameters $\gamma$.

For trees $\gamma$ would be the split variables and split points at internal nodes and predictions at terminal nodes.

Models are fitted by minimizing a loss function averaged over all the training data, such as the squared-error or a likelihood-based loss function:
$$
\min _{\left\{\beta_{m}, \gamma_{m}\right\}_{1}^{M}} \sum_{i=1}^{N} L\left(y_{i}, \sum_{m=1}^{M} \beta_{m} b\left(x_{i} ; \gamma_{m}\right)\right)
$$

Optimization is hard, if base learners are different. A simpler variant would be to use the same $b(x_i,y)$

Alternatively, a **forward stagewise additive modeling** can be used, where new basis functions are sequentially added to the expansion without adjusting the parameters and coefficients to those that have been added.  

That is, at each iteration $m$, one solves for the optimal basis function $b\left(x ; \gamma_{m}\right)$ and corresponding coefficient $\beta_{m}$ to add to the current expansion $f_{m-1}(x)$. This produces $f_{m}(x)$, and the process is repeated. Previously added terms are not modified.

For squared-error loss:
$$
L(y, f(x))=(y-f(x))^{2}
$$
one has
$$
\begin{aligned}
L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\beta b\left(x_{i} ; \gamma\right)\right) &=\left(y_{i}-f_{m-1}\left(x_{i}\right)-\beta b\left(x_{i} ; \gamma\right)\right)^{2} \\
&=\left(r_{i m}-\beta b\left(x_{i} ; \gamma\right)\right)^{2}
\end{aligned}
$$
where $r_{i m}=y_{i}-f_{m-1}\left(x_{i}\right)$ is simply the residual of the current model on the $i$ th observation. Thus, for squared-error loss, the term $\beta_{m} b\left(x ; \gamma_{m}\right)$ that achieves the best fit on the residuals is added to the expansion at each step.

### Loss functions
(not so important)

### Boosting Trees

As trees split the space into disjoint regions $R_{j}, j=1,2, \ldots, J$, a predictive rule for each sample $x$ can be set up as:
$x \in R_{j} \Rightarrow f(x)=\gamma_{j}$ with  a constant $\gamma_{j}$ per region.

Thus a tree can be formally expressed as
$$
T(x ; \Theta)=\sum_{j=1}^{J} \gamma_{j} \mathbb{I}\left(x \in R_{j}\right),
$$
with parameters $\Theta=\left\{R_{j}, \gamma_{j}\right\}_{1}^{J} . J$ is usually treated as a meta-parameter. The parameters are found by minimizing the empirical risk
$$
\hat{\Theta}=\arg \min _{\Theta} \sum_{j=1}^{J} \sum_{x_{i} \in R_{j}} L\left(y_{i}, \gamma_{j}\right) .
$$

For regression $\gamma_j$ given $R_j$ is the mean of all $y_i$ falling into region $R_j$.

Finding the regions $R_j$ is often times difficult. A greedy, top-down recursive partitioning algorithm or a smoother approximation of the loss function is used to find the $R_j$. (p. 356)

(No notes on classification)

### Numerical Optimization with Gradient Boosting

The loss in using $f(x)$ to predict $y$ on the training data is
$$
L(f)=\sum_{i=1}^{N} L\left(y_{i}, f\left(x_{i}\right)\right) .
$$
The goal is to minimize $L(f)$ with respect to $f$, where here $f(x)$ is constrained to be a sum of trees. Ignoring this constraint, minimizing can be viewed as a numerical optimization
$$
\hat{\mathbf{f}}=\arg \min _{\mathbf{f}} L(\mathbf{f}),
$$
where the "parameters" $f \in \mathbb{R}^{N}$ are the values of the approximating function $f\left(x_{i}\right)$ at each of the $N$ data points $x_{i}$ :
$$
\left.\mathbf{f}=\left\{f\left(x_{1}\right), f\left(x_{2}\right)\right), \ldots, f\left(x_{N}\right)\right\} .
$$

A steepest descent approach can be used to calculate the parameter vector $\mathbf{f}$

### Gradient Boosting
If minimizing loss on the training data were the only goal, steepest descent would be the preferred strategy. The gradient is trivial to calculate for any differentiable loss function $L(y, f(x))$.

However, the gradient $(10.35)$ is defined only at the training data points $x_{i}$, whereas the ultimate goal is to generalize $f_{M}(x)$ to new data not represented in the training set.

A possible resolution to this dilemma is to induce a tree $T\left(x ; \Theta_{m}\right)$ at the $m$ th iteration whose predictions $\mathbf{t}_{m}$ are as close as possible to the negative gradient. Using squared error to measure closeness, this leads us to
$$
\tilde{\Theta}_{m}=\arg \min _{\Theta} \sum_{i=1}^{N}\left(-g_{i m}-T\left(x_{i} ; \Theta\right)\right)^{2} .
$$
That is, one fits the tree $T$ to the negative gradient values (10.35) by least squares. (p. 359)

For squared error loss, the negative gradient is just the ordinary residual $-g_{i m}=y_{i}-f_{m-1}\left(x_{i}\right)$, so that $(10.37)$ on its own is equivalent standard least squares boosting. (p. 360)

## Random Forests
Idea was introduced in [[@breimanRandomForests2001]]

Random Forests are a modification to the Bagging approach, yielding in a large collection of de-correlated trees, and then averaging them. (p. 587)

Random Forests improv eht evariance reduction of bagging by reducing the correlation between the trees, without increasing the variance too much. This is done by growing the trees on a random selection of input variables. (p. 588)

Before each split a subset of m $m \leq p$ input variables is selected as a candidate for splitting.

Typically values for $m$ are $\sqrt{p}$ or even as low as 1 .
After $B$ such trees $\left\{T\left(x ; \Theta_{b}\right)\right\}_{1}^{B}$ are grown, the random forest (regression) predictor is
$$
\hat{f}_{\mathrm{rf}}^{B}(x)=\frac{1}{B} \sum_{b=1}^{B} T\left(x ; \Theta_{b}\right)
$$

with $\Theta_{b}$ being a vector that characterizes the parameters of the  $b$-th Random Forest Tree. (p. 588-589)

**Advantages**
- Similar performance to boosting
- Simpler to train and tune

When used for regression, the predictions from each tree at a target point $x$ are simply averaged.  For regression, the default value for $m$ is $\lfloor p / 3\rfloor$ and the minimum node size is five. (p. 592)

For regression, the default value for $m$ is $\lfloor p / 3\rfloor$ and the minimum node size is five.

### Variable importance of Random Forests

Variable importance can be retrieved from the forest. At each split in each tree, the improvement in the split-criterion is the importance measure attributed to the splitting variable, and is accumulated
over all the trees in the forest separately for each variable. (p. 593)


## Annotations

# Annotations  
(14/12/2022, 06:57:13)

“It is difficult to give a general rule on how to choose the number of observations in each of the three parts, as this depends on the signal-tonoise ratio in the data and the training sample size. A typical split might be 50% for training, and 25% each for validation and testing:” ([Hastie, Trevor et al., 2009, p. 241](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=241&annotation=MLMYR5JW))

“Of course, the main caveat here is “independent,” and bagged trees are not. Figure 8.11 illustrates the power of a consensus vote in a simulated example, where only 30% of the voters have some knowledge. In Chapter 15 we see how random forests improve on bagging by reducing the correlation between the sampled trees. Note that when we bag a model, any simple structure in the model is lost. As an example, a bagged tree is no longer a tree. For interpretation of the model this is clearly a drawback. More stable procedures like nearest neighbors are typically not affected much by bagging. Unfortunately, the unstable models most helped by bagging are unstable because of the emphasis on interpretability, and this is lost in the bagging process.” ([Hastie, Trevor et al., 2009, p. 305](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=305&annotation=4LX84RFV))

“Tree-based methods partition the feature space into a set of rectangles, and then fit a simple model (like a constant) in each one.” ([Hastie, Trevor et al., 2009, p. 324](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=324&annotation=A27D7VJP))

“Let’s consider a regression problem with continuous response Y and inputs X1 and X2,” ([Hastie, Trevor et al., 2009, p. 324](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=324&annotation=MVXY4839))

“To simplify matters, we restrict attention to recursive binary partitions like that in the top right panel” ([Hastie, Trevor et al., 2009, p. 324](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=324&annotation=6XLICF4M))

“We first split the space into two regions, and model the response by the mean of Y in each region.” ([Hastie, Trevor et al., 2009, p. 324](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=324&annotation=RUKJDW6J))

“We now turn to the question of how to grow a regression tree. Our data consists of p inputs and a response, for each of N observations: that is, (xi, yi) for i = 1, 2, . . . , N , with xi = (xi1, xi2, . . . , xip). The algorithm needs to automatically decide on the splitting variables and split points, and also what topology (shape) the tree should have. Suppose first that we have a partition into M regions R1, R2, . . . , RM , and we model the response as a constant cm in each region: f (x) = M ∑ m=1” ([Hastie, Trevor et al., 2009, p. 326](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=326&annotation=9JDM8LEL))

“∑(yi − f (xi))2, it is easy to see that the best ˆ cm is just the average of yi in region Rm: ˆ cm = ave(yi|xi ∈” ([Hastie, Trevor et al., 2009, p. 326](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=326&annotation=NA9G7NDT))

“Now finding the best binary partition in terms of minimum sum of squares is generally computationally infeasible. Hence we proceed with a greedy algorithm. Starting with all of the data, consider a splitting variable j and split point s,” ([Hastie, Trevor et al., 2009, p. 326](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=326&annotation=MIYD4IEF))

“For each splitting variable, the determination of the split point s can be done very quickly and hence by scanning through all of the inputs, determination of the best pair (j, s) is feasible.” ([Hastie, Trevor et al., 2009, p. 326](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=326&annotation=28H5RAJP))

“Having found the best split, we partition the data into the two resulting regions and repeat the splitting process on each of the two regions. Then this process is repeated on all of the resulting regions.” ([Hastie, Trevor et al., 2009, p. 326](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=326&annotation=7G75HKWE))

“How large should we grow the tree? Clearly a very large tree might overfit the data, while a small tree might not capture the important structure.” ([Hastie, Trevor et al., 2009, p. 326](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=326&annotation=NDK97SAQ))

“Then this large tree is pruned using cost-complexity pruning, which we now describe.” ([Hastie, Trevor et al., 2009, p. 327](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=327&annotation=LJVZXUAN))

“Besides the size of the constituent trees, J, the other meta-parameter of gradient boosting is the number of boosting iterations M . Each iteration usually reduces the training risk L(fM ), so that for M large enough this risk can be made arbitrarily small. However, fitting the training data too well can lead to overfitting, which degrades the risk on future predictions. Thus, there is an optimal number M ∗ minimizing future risk that is application dependent. A convenient way to estimate M ∗ is to monitor prediction risk as a function of M on a validation sample. The value of M that minimizes this risk is taken to be an estimate of M ∗. This is analogous to the early stopping strategy often used with neural networks (Section 11.4)” ([Hastie, Trevor et al., 2009, p. 383](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=383&annotation=XPVQIJ7W))

“10.12.1 Shrinkage Controlling the value of M is not the only possible regularization strategy. As with ridge regression and neural networks, shrinkage techniques can be employed as well (see Sections 3.4.1 and 11.5). The simplest implementation of shrinkage in the context of boosting is to scale the contribution of each tree by a factor 0 < ν < 1 when it is added to the current approximation. That is, line 2(d) of Algorithm 10.3 is replaced by fm(x) = fm−1(x) + ν · J ∑ j=1 γjmI(x ∈ Rjm). (10.41) The parameter ν can be regarded as controlling the learning rate of the boosting procedure. Smaller values of ν (more shrinkage) result in larger training risk for the same number of iterations M . Thus, both ν and M control prediction risk on the training data. However, these parameters d” ([Hastie, Trevor et al., 2009, p. 383](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=383&annotation=EQTQJLT3))

“10.12 Regularization 365 not operate independently. Smaller values of ν lead to larger values of M for the same training risk, so that there is a tradeoff between them.” ([Hastie, Trevor et al., 2009, p. 384](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=384&annotation=WTWRE2XE))

“Empirically it has been found (Friedman, 2001) that smaller values of ν favor better test error, and require correspondingly larger values of M . In fact, the best strategy appears to be to set ν to be very small (ν < 0.1) and then choose M by early stopping.” ([Hastie, Trevor et al., 2009, p. 384](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=384&annotation=DXHCNIPE))

“We saw in Section 8.7 that bootstrap averaging (bagging) improves the performance of a noisy classifier through averaging. Chapter 15 discusses in some detail the variance-reduction mechanism of this sampling followed by averaging. We can exploit the same device in gradient boosting, both to improve performance and computational efficiency. With stochastic gradient boosting (Friedman, 1999), at each iteration we sample a fraction η of the training observations (without replacement), and grow the next tree using that subsample. The rest of the algorithm is identical. A typical value for η can be 1 2 , although for large N , η can be substantially smaller than 1 2.” ([Hastie, Trevor et al., 2009, p. 384](zotero://select/library/items/FF777NTD)) ([pdf](zotero://open-pdf/library/items/N3FXKVYN?page=384&annotation=LHRLAS5P))