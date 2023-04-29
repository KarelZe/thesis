Incorporate this: 
CatBoost, as well as all standard gradient boosting implementations, builds each new tree to approximate the gradients of the current model. However, all classical boosting algorithms suffer from overfitting caused by the problem of biased pointwise gradient estimates. Gradients used at each step are estimated using the same data points the current model was built on. This leads to a shift of the distribution of estimated gradients in any domain of feature space in comparison with the true distribution of gradients in this domain, which leads to overfitting. The idea of biased gradients was discussed in previous literature [1] [9]. [9] J. H. Friedman. Stochastic gradient boosting. Computational Statistics & Data Analysis, 38(4):367–378, 2002. [1] L. Breiman. Out-of-bag estimation, 1996



One approach that aims to reduce the bias and the variance, is *gradient boosting*, which was introduced by ([[@friedmanGreedyFunctionApproximation2001]]9). *Gradient Boosting* iteratively combines oversimplified models, the *weak learners*, to obtain an improved accurate ensemble estimate. This chapter follows ([[@friedmanGreedyFunctionApproximation2001]]9) to derive gradient-boosted decision trees for binary classification.

Recall, by cref-problem we perform *binary probabilistic classification* and by cref-[[🔢Trade initiator]] we defined the labels are, $y \in \{-1,1\}$. For gradient boosting, instead of modelling the class-conditional probabilities directly, we model the conditional *log odds* instead, which can be interpreted as the probability of observing class $1$ or a buyer-initiated trade, and covert to class-conditional probabilities as needed.

Following ([[@friedmanStochasticGradientBoosting2002]]9) we set the loss function to be the cross-entropy loss, given by:
$$
L_{\mathrm{CE}} \colon \mathbb{R}^2 \to \mathbb{R} \quad L_{\mathrm{CE}}(y, F) = \log(1+\exp(-2yF))
$$
%%
The logistic loss is sometimes called cross-entropy loss. It is also known as log loss (In this case, the binary label is often denoted by {−1,+1}).
%%
where:
$$
	F(\mathbf{x}) = \frac{1}{2} \log \left[\frac{\operatorname{Pr}(y=1\mid \mathbf{x})}{\operatorname{Pr}(y=-1\mid \mathbf{x})}\right]
$$
$F(\mathbf{x})$ is the model's prediction in terms of conditional *log-odds*. The cross-entropy loss is a reasonable choice as a loss function, as it is suitable for binary classification, convex, and twice differentiable, a property we exploit later. It is is visualized in cref-fig.
![[cross-entropy-loss.png]]
(y = loss, Margin y-f(x), x=0 log 2)

We first intialise the model with a naïve prediction, based on the average class $\bar{y}$ from all training samples:
$$
F_0(\mathbf{x})= \frac{1}{2} \log \left[\frac{1+\bar{y}}{1-\bar{y}}\right].
$$
Expectedly, $F_0(x)$ is a poor estimate, capturing hardly any regularities of the data. Gradient-boosting solves this issue by adding weak learners to the ensemble. New trees are added greedily,  one per iteration $m$ with $m=1,2,\cdots M$. The weak learner in the $m$-th iteration is chosen to approximate the *pseudo residual* $r_i$, which is the negative gradient of the observed value of the $i$-th sample and the current estimate:
$$
r_i=-\left[\frac{\partial L_{\mathrm{CM}}\left(y_i, F\left(\mathbf{x}_i\right)\right)}{\partial F\left(\mathbf{x}_i\right)}\right]_{F(\mathbf{x})=F_{m-1}(\mathbf{x})}=2 y_i /\left(1+\exp \left(2 y_i F_{m-1}\left(\mathbf{x}_i\right)\right)\right) .
$$
<mark style="background: #ABF7F7A6;">“Steepest descent can be viewed as a very greedy strategy, since −gm is the local direction in IRN for which L(f ) is most rapidly decreasing at f = fm−1.” (Hastie, Trevor et al., 2009, p. 378)</mark>

Typically, regression trees (cp. [[🎄Decision Trees]]) are chosen as a weak learner, since they are computationally cheap and can produce continuous estimates for the residual. The $m$-th regression tree contains $J$ terminal regions, denoted by $R_{j m}, j=1,2, \ldots, J_{m}$. We search for an estimate $\gamma_{j,m}$ for the terminal node $R_{jm}$ that minimizes the cross-entropy over all samples within the node:
$$
\gamma_{j m}=\arg \min _\gamma \sum_{\mathbf{x}_i \in R_{j m}} \log \left(1+\exp \left(-2 y_i\left(F_{m-1}\left(\mathbf{x}_i\right)+\gamma\right)\right)\right)
$$
cref-eq cannot be solved in closed-form and is typically approached by the Newton-Raphson method with a second order-order approximation of the loss. Following ([[@friedmanAdditiveLogisticRegression2000]]??) this is -(footnote with calculus. Figure out second order polynomial):
$$
\gamma_{j m}=\sum_{\mathbf{x}_i \in R_{j m}} r_i / \sum_{\mathbf{x}_i \in R_{j m}}\left|r_i\right|\left(2-\left|r_i\right|\right)
$$
with $r_i$ given by cref-eq.

An improved estimate for $\mathbf{x}$ is calculated from the previous estimate by adding the new regression tree to the ensemble. The later moves the prediction towards the greatest descent and thus improves the overall prediction. The updated model is given by:
$$
F_{m}(\mathbf{x})=F_{m-1}(\mathbf{x})+\nu \sum_{j=1}^{J_{m}} \gamma_{j m} \mathbb{I}\left(\mathbf{x} \in R_{j m}\right).
$$
After $M$ iterations we obtain the final estimate calculated as: $F_m(\mathbf{x})$. To avoid overfitting the residuals, only proportional steps towards the negative gradient are taken, which is controlled by by the learning rate $\nu \in \left(0, 1\right]$ ([[@friedmanGreedyFunctionApproximation2001]] 13). The learning rate $\nu$ and the size of the ensemble $M$ are deeply intertwined and thus require to be tuned together ([[@friedmanGreedyFunctionApproximation2001]] 13). 

Gradient boosting is still prone to overfitting. One solution, documented in ([[@hastietrevorElementsStatisticalLearning2009]]384), is to employ *early stopping*, whereby the ensemble is only grown in size, as long as adding more weak learners leads to an decrease in loss on on the validation set. Another approach is to limit the amount of data seen during training by fitting each trees on a random subset of samples, as proposed in ([[@friedmanStochasticGradientBoosting2002]]3), or on a subset of features, as popularized by ([[@chenXGBoostScalableTree2016]] 3). Another alternative to extend the loss function for a $\ell_2$ regularization term and penalize the model for complexity, as proposed in ([[@chenXGBoostScalableTree2016]]2).  ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]6) grow *oblivious trees*, which use the same splitting criterion for all nodes of one level in a tree. The rationale is, that these arguably simple trees, and achieve an imperfect fit, which regularises the model.

In recent years, variants of gradient boosting appeared in literature. Prominent examples include *lightgbm* ([[@keLightGBMHighlyEfficient2017]]), *xgboost* ([[@chenXGBoostScalableTree2016]]1--13), and *catboost* ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]]1--23), which differ mainly differ by the policy how trees are grown and how overfitting is addressed. Performance-wise, differences between the implementations are minor, as a short swift through empirical studies suggest (cp. [[@grinsztajnWhyTreebasedModels2022]]??) or ([[@gorishniyRevisitingDeepLearning2021]]??).

As we noted at the beginning, $F_M(\mathbf{x})$ models the logg-odds. We can recover the class-conditional probabilities $p(y \mid \mathbf{x})$ by taking the inverse:
$$
p(y \mid \mathbf{x}) = 1 /\left(1+\exp(-2yF_M(\mathbf{x})\right).
$$
and get the majority class by eq-simple-classifier. 
%%
“2.1 Predictive Modeling Formally, we aim to model a target variable Y ∈ Y given some feature vector X ∈ X based on training data {(xi, yi)}n i=1 that has been sampled according to the joint distribution of X and Y . We focus on models in the form of a single-valued scoring function f : X → R. For instance, in regression problems (Y = R), f typically models the conditional expectation of the target, i.e., f (x) ≈ E(Y | X = x), whereas in binary classification problems (Y = {−1, 1}), f typically models the conditional log odds, i.e., f (x) ≈ ln P (Y = 1 | X = x)/P (Y = −1 | X = x) and the conditional probabilities p(y | x) are recovered by the sigmoid transform p(y | x) = σ(f (x)) = (1 + exp(−yf (x)))−1 .” (Boley et al., 2021, pp. -)
%%

**Notes:**
[[🐈Gradient Boosting notes]]
