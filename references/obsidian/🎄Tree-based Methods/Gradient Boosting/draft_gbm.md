One approach that aims to reduce the bias, is *gradient boosting*, which was popularized by (Friedman...). *Gradient Boosting* is different from the afore-mentioned appraoches, as it sequentially adds the approximations of several over-simplified models (so-called
*weak-lerners*) to an ensemble estimate. Shallow trees are commonly used as weak learners. Due to the sequential ensemble building, the construction of the tree is dependent on the trees already built.

Recall from section [[draft_dt]] that a tree can be expressed as

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
A common loss function for regression is the *squared error loss*, which can be solved for analytically by taking the derivative and setting it equal to zero. The predictions of a tree $T\left(x_{i} ; \Theta_{m}\right)$, that yield the maximum decrease of equation (...) are similar to the components of the *negative gradient descent*. The major drawback is however, that gradient is only defined for data points $x_i$ seen during training, contradicting the creation of a generalizing model $f_{M}(x)$. A more robust approach can be found in *gradient boosting*.

Focusing only on the update step, which is executed $m = 1,\cdots, M$-times, *gradient boosting* starts by calculating the negative gradient of the loss between the observed value for the $i$-th sample and its current predicted value:
$$
r_{i m}=-\left[\frac{\partial L\left(y_{i}, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f=f_{m-1}}.
$$
The components of the negative gradient are referred to as *pseudo residuals*.

Subsequently, a regression tree is then fit on these pseudo residuals. The $m$-th regression tree contains $J$ terminal regions denoted by $R_{j m}, j=1,2, \ldots, J_{m}$. The predicted estimate $\gamma_{j,m}$ for the $j$-th region is obtained by minimizing e. g. the squared loss:

$$
\gamma_{j m}=\arg \min _{\gamma} \sum_{x_{i} \in R_{j m}} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\gamma\right)
$$
Recall from chapter (...) that the estimate $\gamma_{jm}$ is constant for the entire region. As before the best estimate is simply the average over all residuals.

An improved estimate for $x$ is calculated from the previous estimate by adding the tree, fitted on the residuals as shown in equation (...). The later moves the prediction towards the greatest descent and thus improves the overall prediction.

$$
f_{m}(x)=f_{m-1}(x)+\nu \sum_{j=1}^{J_{m}} \gamma_{j m} \mathbbm{I}\left(x \in R_{j m}\right).
$$
To avoid overfitting,  only proportional step towards the negative gradient are taken. The pace is controlled by the learning rate $\nu \in \left(0, 1\right]$. While a small learning rate slows down learning, it allows for more different shaped trees to attack the residuals. (james)

After $M$ iterations we obtain the final estimate calculated as: $\hat{f}(x)=f_{M}(x)$.






Our explanation is based on [[@hastietrevorElementsStatisticalLearning2009]].


-   Introduce idea of bootstrapping in bagging [[@breimanBaggingPredictors1996]].
- Introduce notion of strong learner
- Transfer to Gradient Boosting
- What problem does it solve, that Random Forests can not solve?
- ensemble of weak prediction models (most oftenly trees)
- Simple explanation given in [[@guEmpiricalAssetPricing2020]]
- Explanation [[@rossiMachineLearning]]



**Flash cards**

Assume we seek to fit a gradient boosting model to the $\left(X_{1}, y_{1}\right),\left(X_{2}, y_{2}\right), \ldots,\left(X_{n}, y_{n}\right)$, where $X_{i}$ are the explanatory variables for the $i^{\text {th }}$ sample and $y_{i}$ is its dependent variable.

In the first step, using gradient boosting we fit a base learner $f_{0}(X)$ for modeling. We define the corresponding prediction residuals for first regression tree $e_{i 0}=y_{i}-f_{0}\left(X_{i}\right)$ for $i=1,2, \ldots, n$. In the next step, a regression tree $f_{1}\left(X_{i}\right)$ to $\left(X_{1}, e_{10}\right),\left(X_{2}, e_{20}\right), \ldots,\left(X_{n}, e_{n 0}\right)$ is fit, and a recovery rate prediction will be equal to $f_{0}\left(X_{i}\right)+e_{i 1}$.
Here $e_{i 1}$ is the corresponding predicted residuals from $f_{1}\left(X_{i}\right)$. The gradient boosting estimation after $B$ iterations is defined as:
$$
\hat{y}_{i}=f_{0}\left(X_{i}\right)+\sum_{b=1}^{B} e_{i b}
$$



