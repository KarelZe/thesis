One approach that aims to reduce the bias, is *gradient boosting*, which was popularized by [[@friedmanGreedyFunctionApproximation2001]]. *Gradient Boosting* is different from the afore-mentioned approach, as it sequentially adds the approximations of several over-simplified models (so-called *weak-lerners*) to an ensemble estimate. Shallow trees are commonly used as weak learners [[@friedmanGreedyFunctionApproximation2001]]. Due to the sequential ensemble building, the construction of the tree is only dependent on the trees previously built. 

Recall from section [[🎄decison_trees]] that a tree can be expressed as

$$
T(x ; \Theta)=\sum_{j=1}^{J} \gamma_{j} I\left(x \in R_{j}\right).
$$
Several trees can be combined to a *boosted tree* through summation:

$$
f_{M}(x)=\sum_{m=1}^{M} T\left(x ; \Theta_{m}\right)
$$

Here, $M$ sets the number of iterations or trees being built. Instead of fitting trees directly to the data, as with Bagging, the model in the $m$-th iteration is fitted to the residuals of the previous model. The previously built trees are not altered, but required for later trees.

At each iteration one seeks to find the optimal parameter set $\Theta_{m}=\left\{R_{j m}, \gamma_{j m}\right\}_{1}^{J_{m}}$ for the $m$ th iteration, that minimizes the loss between the true estimate and the estimate from the current model $f_{m-1}\left(x_{i}\right)$ and the newly added tree, updating the residuals:

$$
\hat{\Theta}_{m}=\arg \min _{\Theta_{m}} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+T\left(x_{i} ; \Theta_{m}\right)\right).
$$
<mark style="background: #FFB86CA6;">The tree being added, is selected greedily, thereby considering only the improvement in the current iteration and neglecting the impact on the full grown ensemble. (for citation on "greedy" see [[@prokhorenkovaCatBoostUnbiasedBoosting2018]])</mark>

A common loss function for regression is the *squared error loss*, which can be solved for analytically by taking the derivative and setting it equal to zero. The predictions of a tree $T\left(x_{i} ; \Theta_{m}\right)$, that yield the maximum decrease of equation (...) are similar to the components of the *negative gradient descent*. The major drawback is however, that gradient is only defined for data points $x_i$ seen during training, contradicting the creation of a generalizing model $f_{M}(x)$. A more robust approach can be found in *gradient boosting*.

Focusing only on the update step, which is executed $m = 1,\cdots, M$-times, *gradient boosting* starts by calculating the negative gradient of the loss between the observed value for the $i$-th sample and its current predicted value:
$$
r_{i m}=-\left[\frac{\partial L\left(y_{i}, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f=f_{m-1}}.
$$
The components of the negative gradient are referred to as *pseudo residuals*.

Subsequently, a regression tree is then fit on these pseudo residuals. The $m$-th regression tree contains $J$ terminal regions denoted by $R_{j m}, j=1,2, \ldots, J_{m}$. The predicted estimate $\gamma_{j,m}$ for the $j$-th region is obtained by minimizing e. g. the squared loss:

$$
\gamma_{j m}=\arg \min _{\gamma} \sum_{x_{i} \in R_{j m}} \mathcal{L}\left(y_{i}, f_{m-1}\left(x_{i}\right)+\gamma\right)
$$
<mark style="background: #FFB8EBA6;">TODO: Where does weighting of the loss function comes into play? See [[@prokhorenkovaCatBoostUnbiasedBoosting2018]] paper. Also see https://catboost.ai/en/docs/concepts/loss-functions-classification#Logit </mark>

In addition to imposing constraints on the individual trees,  the boosted ensemble can also be regularized by extending the loss (Eq. (...)) for a penalty term.  

<mark style="background: #FFB8EBA6;">TODO: See [[@chenXGBoostScalableTree2016]] for detailed explanation. Idea can also be found in [[@friedmanAdditiveLogisticRegression2000]]. Also [[@hastietrevorElementsStatisticalLearning2009]] (p. 617) could be interesting. </mark>

Recall from chapter ([[🎄decison_trees]]) that the estimate $\gamma_{jm}$ is constant for the entire region. As before the best estimate is simply the average over all residuals.

An improved estimate for $x$ is calculated from the previous estimate by adding the tree, fitted on the residuals as shown in equation (...). The later moves the prediction towards the greatest descent and thus improves the overall prediction.

$$
f_{m}(x)=f_{m-1}(x)+\nu \sum_{j=1}^{J_{m}} \gamma_{j m} \mathbb{I}\left(x \in R_{j m}\right).
$$
After $M$ iterations we obtain the final estimate calculated as: $\hat{f}(x)=f_{M}(x)$. To avoid overfitting the residuals, only proportional steps towards the negative gradient are taken. The contribution of each tree is controlled by the learning rate $\nu \in \left(0, 1\right]$. 

<mark style="background: #BBFABBA6;">A small learning rate not just gives room for subsequent trees too ... / avoids overfitting the residuals</mark>

<mark style="background: #D2B3FFA6;">The learning rate and the size of the ensemble are deeply intertwined.

Another method to regularizing 


While a small learning rate slows down learning, it hel

- learning rate and size of ensembles must be tuned togehter
- Friedman suggests to study the `LOF` and grow largest possible ensembles, which is possible 

The learning rate $\nu$ and the size of the ensemble $M$ are deeply intertwined [[@friedmanGreedyFunctionApproximation2001]]. 
</mark>

<mark style="background: #ABF7F7A6;">- Above we assumed $M$ to be fixed,
- observation similar to decision trees. One can overfit the training set. 
- [[@hastietrevorElementsStatisticalLearning2009]] suggest to monitor the prediction risk on the validation sample. "monitor prediction risk as a function of $M$" 
- We monitor using learning curves.
- It's better to use smallest possible learning rate and choose $M$ as large as possible. Given the sequential nature of the algorithm no computation need to be repeated.</mark>

Shrinkage (Eq. (...)) and (...) are not the only options to regularize gradient-boosted trees. Other techniques constrain the amount of data seen during training by fitting each trees on a random subset of samples, as proposed in [[@friedmanStochasticGradientBoosting2002]] (p. 3), or on a subset of features, as popularized by [[@chenXGBoostScalableTree2016]] (p. 3) (Footnote: Note the link to random forests and Bagging. Bagging by Breiman was sthe reason to implement stochastic gradient boosting in the first place (see [[@friedmanStochasticGradientBoosting2002]] (p. 4).) We denote the fraction of samples seen during training with $\eta$. Both approaches can not just improve generalization performance but also the computational complexity, due to the simplified splitting procedures [[@chenXGBoostScalableTree2016]] (p. 3) and [[@friedmanStochasticGradientBoosting2002]] (p.10). (also written in [[@hastietrevorElementsStatisticalLearning2009]] p. 365)

So far we made several simplifying assumptions, that don't hold in real-world data sets:
<mark style="background: #FF5582A6;">
- all input input is numerical
- data sets is free of missing data
- it's computationally feasible to evaluate all possible splits from all features
- Strategy how tree is grown
</mark>

We address these issues.
-  https://developers.google.com/machine-learning/decision-forests/gradient-boosting
- Variants of GBM, comparison: [CatBoost vs. LightGBM vs. XGBoost | by Kay Jan Wong | Towards Data Science](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924) (e. g., symmetric, balanced trees vs. asymetric trees) or see kaggle book for differences between xgboost ([[@chenXGBoostScalableTree2016]]), lightgbm ([[@keLightGBMHighlyEfficient2017]]), catboost ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]) etc. [[@banachewiczKaggleBookData2022]]
- How can missing values be handled in decision trees? (see [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]] as a primer)
  How can categorical data be handled in decision trees?
- Start of with gradient boosted trees for regression. Gradient boosted trees for classification are derived from that principle.
- cover desirable properties of gradient boosted trees
- for handling of missing values see [[@twalaGoodMethodsCoping2008]]. Send missing value to whether side, that leads to the largest information gain (Found in [[@josseConsistencySupervisedLearning2020]])
- [[@chenXGBoostScalableTree2016]] use second order methods for optimization.

### Adaptions for Probablistic Classification
- Explain how the Gradient Boosting Procedure for the regression case, can be extended to the classifcation case
- Discuss the problem of obtainining good probability estimates from a boosted decision tree. See e. g., [[@caruanaObtainingCalibratedProbabilities]] or [[@friedmanAdditiveLogisticRegression2000]] (Note paper is commenting about boosting, gradient boosting has not been published at the time)
- Observations in [[@tanhaSemisupervisedSelftrainingDecision2017]] on poor probability estimates are equally applicable.
- https://catboost.ai/news/catboost-enables-fast-gradient-boosting-on-decision-trees-using-gpus
