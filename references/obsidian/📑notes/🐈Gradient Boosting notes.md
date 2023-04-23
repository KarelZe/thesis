
Over


## Previous attempt
One approach that aims to reduce the bias, is *gradient boosting*, which was popularized by [[@friedmanGreedyFunctionApproximation2001]]. *Gradient Boosting* is different from the afore-mentioned approach, as it sequentially adds the approximations of several over-simplified models (so-called *weak-lerners*) to an ensemble estimate. Shallow trees are commonly used as weak learners [[@friedmanGreedyFunctionApproximation2001]]. Due to the sequential ensemble building, the construction of the tree is only dependent on the trees previously built. 

Recall from section [[🎄Decision Trees]] that a tree can be expressed as

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

Recall from chapter ([[🎄Decision Trees]]) that the estimate $\gamma_{jm}$ is constant for the entire region. As before the best estimate is simply the average over all residuals.

An improved estimate for $x$ is calculated from the previous estimate by adding the tree, fitted on the residuals as shown in equation (...). The later moves the prediction towards the greatest descent and thus improves the overall prediction.

$$
f_{m}(x)=f_{m-1}(x)+\nu \sum_{j=1}^{J_{m}} \gamma_{j m} \mathbb{I}\left(x \in R_{j m}\right).
$$
After $M$ iterations we obtain the final estimate calculated as: $\hat{f}(x)=f_{M}(x)$. To avoid overfitting the residuals, only proportional steps towards the negative gradient are taken. The contribution of each tree is controlled by the learning rate $\nu \in \left(0, 1\right]$. 







When a [function’s](https://www.statisticshowto.com/types-of-functions/) variable represents a probability, p (as in the function above), it’s called the **logit function** .


Die Umkehrfunktion des Logits ist die logistische Funktion (manchmal auch Expit oder Sigmoid genannt):
$$
F_{\text {logistisch }}:=\operatorname{logit}^{-1}(s)=\frac{\exp (s)}{1+\exp (s)}=\frac{1}{1+\exp (-s)} .
$$

## Literature

**Data set:** Formally, we aim to model a target variable $Y \in \mathbb{Y}$ given some feature vector $X \in \mathbb{X}$ based on training data $\left\{\left(x_i, y_i\right)\right\}_{i=1}^n$ that has been sampled according to the joint distribution of $X$ and $Y$. (Add to section [[🍪Selection Of Supervised Approaches]]) We focus on models in the form of a single-valued scoring function $f: \mathbb{X} \rightarrow \mathbb{R}$. For instance, in regression problems $(\mathbb{Y}=\mathbb{R}), f$ typically models the conditional expectation of the target, i.e., $f(x) \approx E(Y \mid X=x)$, whereas in binary classification problems $(\mathbb{Y}=\{-1,1\}), f$ typically models the conditional $\log$ odds, i.e., $f(x) \approx \ln P(Y=1 \mid X=x) / P(Y=-1 \mid X=x)$ and the conditional probabilities $p(y \mid x)$ are recovered by the sigmoid transform
$$
p(y \mid x)=\sigma(f(x))=(1+\exp (-y f(x)))^{-1} .
$$
(found here[[@boleyBetterShortGreedy2021]]; do not cite but like their presentation)
**Dataset:** where $\mathcal{D}=\left\{\boldsymbol{x}^{(i)}, y^{(i)}\right\}_{i=1}^N$ is the training dataset. Each set of parameters can be considered a hypothesis or explanation about how the world works. Samples from the posterior should yield explanations consistent with the observations of the world contained within the training data $\mathcal{D}$. However, on data far from $\mathcal{D}$ each set of parameters can yield different predictions. Therefore, estimates of knowledge uncertainty can be obtained by examining the diversity of predictions.([[@malininUncertaintyGradientBoosting2021]])

Generally, the meaning of a score $f(x)$ is encapsulated in a positive loss function $l(y, f(x))$ that represents the cost of predicting $f(x)$ when the true target value is $y$. Specific examples are the squared loss $l(y, f(x))=(y-f(x))^2$ for regression problems and the $\operatorname{logistic} \operatorname{loss} l(y, f(x))=\log (1+\exp (-y f(x)))$ for classification problems. However, we only assume that $l$ is twice differentiable and convex in its second argument. (found here[[@boleyBetterShortGreedy2021]]; do not cite but like their presentation)


**Gradient Boosting with Uncertainity:🐈** Gradient boosting is a powerful machine learning technique especially useful on tasks containing heterogeneous features. It iteratively combines weak models, such as decision trees, to obtain more accurate predictions. Formally, given a dataset $\mathcal{D}$ and a loss function $L: \mathbb{R}^2 \rightarrow \mathbb{R}$, the gradient boosting algorithm (Friedman, 2001) iteratively constructs a model $F: X \rightarrow \mathbb{R}$ to minimize the empirical risk $\mathcal{L}(F \mid \mathcal{D})=\mathbb{E}_{\mathcal{D}}[L(F(\boldsymbol{x}), y)]$. At each iteration $t$ the model is updated as:
$$
F^{(t)}(\boldsymbol{x})=F^{(t-1)}(\boldsymbol{x})+\epsilon h^{(t)}(\boldsymbol{x})
$$
where $F^{(t-1)}$ is a model constructed at the previous iteration, $h^{(t)}(\boldsymbol{x}) \in \mathcal{H}$ is a weak learner chosen from some family of functionds $\mathcal{H}$, and $\epsilon$ is learning rate. The weak learner $h^{(t)}$ is usually chosen to approximate the negative gradient $-g^{(t)}(\boldsymbol{x}, y):=-\left.\frac{\partial L(y, s)}{\partial s}\right|_{s=F^{(t-1)}(\boldsymbol{x})}$ :
$$
h^{(t)}=\underset{h \in \mathcal{H}}{\arg \min } \mathbb{E}_{\mathcal{D}}\left[\left(-g^{(t)}(\boldsymbol{x}, y)-h(\boldsymbol{x})\right)^2\right] .
$$
A weak learner $h^{(t)}$ is associated with parameters $\phi^{(t)} \in \mathbb{R}^d$. We write $h^{(t)}\left(\boldsymbol{x}, \boldsymbol{\phi}^{(t)}\right)$ to reflect this dependence. The set of weak learners $\mathcal{H}$ often consists of shallow decision trees, which are models that recursively partition the feature space into disjoint regions called leaves. Each leaf $R_j$ of the tree is assigned to a value, which is the estimated response $y$ in the corresponding region. We can write $h\left(\boldsymbol{x}, \boldsymbol{\phi}^{(t)}\right)=\sum_{j=1}^d \phi_j^{(t)} \mathbf{1}_{\left\{\boldsymbol{x} \in R_j\right\}}$, so the decision tree is a linear function of $\boldsymbol{\phi}^{(t)}$. The final GBDT model $F$ is a sum of decision trees (7) and the parameters of the full model are denoted by $\boldsymbol{\theta}$. ([[@malininUncertaintyGradientBoosting2021]] do not cite but like the presentation)

**CatBoost: 🐈** 
Assume we observe a dataset of examples $\mathcal{D}=\left\{\left(\mathbf{x}_k, y_k\right)\right\}_{k=1 . . n}$, where $\mathbf{x}_k=\left(x_k^1, \ldots, x_k^m\right)$ is a random vector of $m$ features and $y_k \in \mathbb{R}$ is a target, which can be either binary or a numerical response. Examples $\left(\mathbf{x}_k, y_k\right)$ are independent and identically distributed according to some unknown distribution $P(\cdot, \cdot)$. The goal of a learning task is to train a function $F: \mathbb{R}^m \rightarrow \mathbb{R}$ which minimizes the expected loss $\mathcal{L}(F):=\mathbb{E} L(y, F(\mathbf{x}))$. Here $L(\cdot, \cdot)$ is a smooth loss function and $(\mathbf{x}, y)$ is a test example sampled from $P$ independently of the training set $\mathcal{D}$.
A gradient boosting procedure [12] builds iteratively a sequence of approximations $F^t: \mathbb{R}^m \rightarrow \mathbb{R}$, $t=0,1, \ldots$ in a greedy fashion. Namely, $F^t$ is obtained from the previous approximation $F^{t-1}$ in an additive manner: $F^t=F^{t-1}+\alpha h^t$, where $\alpha$ is a step size and function $h^t: \mathbb{R}^m \rightarrow \mathbb{R}$ (a base predictor) is chosen from a family of functions $H$ in order to minimize the expected loss:
$$
h^t=\underset{h \in H}{\arg \min } \mathcal{L}\left(F^{t-1}+h\right)=\underset{h \in H}{\arg \min } \mathbb{E} L\left(y, F^{t-1}(\mathbf{x})+h(\mathbf{x})\right)
$$
The minimization problem is usually approached by the Newton method using a second-order approximation of $\mathcal{L}\left(F^{t-1}+h^t\right)$ at $F^{t-1}$ or by taking a (negative) gradient step. Both methods are kinds of functional gradient descent [10,, 24t]. In particular, the gradient step $h^t$ is chosen in such a way that $h^t(\mathbf{x})$ approximates $-g^t(\mathbf{x}, y)$, where $g^t(\mathbf{x}, y):=\left.\frac{\partial L(y, s)}{\partial s}\right|_{s=F^{t-1}(\mathbf{x})}$. Usually, the least-squares approximation is used:
$$
h^t=\underset{h \in H}{\arg \min } \mathbb{E}\left(-g^t(\mathbf{x}, y)-h(\mathbf{x})\right)^2
$$
CatBoost is an implementation of gradient boosting, which uses binary decision trees as base predictors. A decision tree $[4,10,27]$ is a model built by a recursive partition of the feature space $\mathbb{R}^m$ into several disjoint regions (tree nodes) according to the values of some splitting attributes $a$. Attributes are usually binary variables that identify that some feature $x^k$ exceeds some threshold $t$, that is, $a=\mathbb{1}_{\left\{x^k>t\right\}}$, where $x^k$ is either numerical or binary feature, in the latter case $\left.t=0.5\right\}$ Each final region (leaf of the tree) is assigned to a value, which is an estimate of the response $y$ in the region for the regression task or the predicted class label in the case of classification problem 3 In this way, a decision tree $h$ can be written as
$$
h(\mathbf{x})=\sum_{j=1}^J b_j \mathbb{1}_{\left\{\mathbf{x} \in R_j\right\}},
$$
where $R_j$ are the disjoint regions corresponding to the leaves of the tree. ([[@prokhorenkovaCatBoostUnbiasedBoosting2018]])


## Google Machine Learning course
Gradient boosting (optional unit)
Send feedback
In regression problems, it makes sense to define the signed error as the difference between the prediction and the label. However, in other types of problems this strategy often leads to poor results. A better strategy used in gradient boosting is to:
- Define a loss function similar to the loss functions used in neural networks. For example, the entropy (also known as log loss) for a classification problem.
- Train the weak model to predict the gradient of the loss according to the strong model output.
Formally, given a loss function $L(y, p)$ where $y$ is a label and $p$ a prediction, the pseudo response $z_i$ used to train the weak model at step $i$ is:
$$
z_i=\frac{\partial L\left(y, F_i\right)}{\partial F_i}
$$
where:
- $F_i$ is the prediction of the strong model.
The preceding example was a regression problem: The objective is to predict a numerical value. In the case of regression, squared error is a common loss function:
$$
L(y, p)=(y-p)^2
$$
In this case, the gradient is:
$$
z=\frac{\partial L\left(y, F_i\right)}{\partial F_i}=\frac{\partial(y-p)^2}{\partial p}=2(y-p)=2 \text { signed error }
$$
In order words, the gradient is the signed error from our example with a factor of 2 . Note that constant factors do not matter because of the shrinkage. Note that this equivalence is only true for regression problems with squared error loss. For other supervised learning problems (for example, classification, ranking, regression with percentile loss), there is no equivalence between the gradient and a signed error.

## Leaf and structure optimization with Newton's method step

Newton's method is an optimization method like gradient descent. However, unlike the gradient descent that only uses the gradient of the function to optimize, Newton's method uses both the gradient (first derivative) and the second derivative of the function for optimization.
A step of gradient descent is as follows:
$$
x_{i+1}=x_i-\frac{d f}{d x}\left(x_i\right)=x_i-f^{\prime}\left(x_i\right)
$$
and Newton's method as as follows:
$$
x_{i+1}=x_i-\frac{\frac{d f}{d x}\left(x_i\right)}{\frac{d^2 f}{d^2 x}\left(x_i\right)}=x_i-\frac{f^{\prime}\left(x_i\right)}{f^{\prime \prime}\left(x_i\right)}
$$
Optionally, Newton's method can be integrated to the training of gradient boosted trees in two ways:
1. Once a tree is trained, a step of Newton is applied on each leaf and overrides its value. The tree structure is untouched; only the leaf values change.
2. During the growth of a tree, conditions are selected according to a score that includes a component of the Newton formula. The structure of the tree is impacted.


Like bagging and boosting, gradient boosting is a methodology applied on top of another machine learning algorithm. Informally, gradient boosting involves two types of models:
- a "weak" machine learning model, which is typically a decision tree.
- a "strong" machine learning model, which is composed of multiple weak models.
In gradient boosting, at each step, a new weak model is trained to predict the "error" of the current strong model (which is called the pseudo response). We will detail "error" later. For now, assume "error" is the difference between the prediction and a regressive label. The weak model (that is, the "error") is then added to the strong model with a negative sign to reduce the error of the strong model.
Gradient boosting is iterative. Each iteration invokes the following formula:
$$
F_{i+1}=F_i-f_i
$$
Where:
- $F_i$ is the strong model at step $i$.
- $f_i$ is the weak model at step $i$.
This operation repeats until a stopping criterion is met, such as a maximum number of iterations or if the (strong) model begins to overfit as measured on a separate validation dataset.
Let's illustrate gradient boosting on a simple regression dataset where:
- The objective is to predict $y$ from $x$.
- The strong model is initialized to be a zero constant: $F_0(x)=0$.

## Some inspiration




![[Pasted image 20230405172011.png]]
![[Pasted image 20230405171940.png]]




Our introduction assumes that all input is numerical, which is true for the mehrheit of our dataset.


$$
    \operatorname{l}_{0-1} \colon \mathcal{Y} \times \mathcal{Y} \to \left[0, 1\right], \quad \operatorname{l}_{0-1}(y, \widehat{y}) = \mathbb{I}\left(y_{i}\neq \widehat{y}_{i}\right).
$$

 The model now models:
$$
f(x) = \ln\left(\frac{P(Y=1 \mid X=x)}{1-P(Y=1\mid X=x)}\right)
$$

<mark style="background: #FFB86CA6;">If $p$ is a probability, then $p /(1-p)$ is the corresponding odds; the logit of the probability is the logarithm of the odds, i.e.:
$$
\operatorname{logit}(p)=\ln \left(\frac{p}{1-p}\right)=\ln (p)-\ln (1-p)=-\ln \left(\frac{1}{p}-1\right)=2 \operatorname{atanh}(2 p-1)
$$
(Wikipedia) https://en.wikipedia.org/wiki/Logit</mark> -> odds of success = probability of an even happening divided by prbability of an event not happening. 

We can still recover the conditional probabilities $p(y \mid x)$ for a trade to be buyer- or seller-initiated with the Sigmoid transform, given by:
$$
p(y \mid x)=\sigma(f(x))=\frac{1}{1 + \exp(-y f(x))}.
$$
Like before, we require a differentiable loss function. A common replacement for the previous square loss, is the the log-loss (binary cross-entropy loss. -> seems to be equivalent for binary case https://stackoverflow.com/questions/50913508/what-is-the-difference-between-cross-entropy-and-log-loss-error).  $\operatorname{logistic} \operatorname{loss} l(y, f(x))=\log (1+\exp (-y f(x)))$

![[Pasted image 20230420130156.png]]
![[Pasted image 20230420131925.png]]
![[Pasted image 20230420132045.png]]
![[Pasted image 20230420132109.png]]

High level idea:
![[Pasted image 20230421085845.png]]

Require a loss function, i.e., *binomial loss* / cross entropy loss:
![[Pasted image 20230421091555.png]]
(copied from Coors) (This is analogous to Friedman paper https://jerryfriedman.su.domains/ftp/trebst.pdf)
(based on this conversion?)
![[Pasted image 20230421105233.png]]
(from [[@friedmanAdditiveLogisticRegression2000]])

y is classification target (-1,1) and F(x) is the logodds (with constant?). 
![[Pasted image 20230421094859.png]]
(log(odds) = log(p/(1-p)))


![[Pasted image 20230421094933.png]]
(in hasties this baby is called residual)
![[Pasted image 20230421095146.png]]
(verified derivative)


Start by making an initial prediction:
![[Pasted image 20230421091802.png]]

(how to arrive at formulation of )

Start with naive prediction / initialize model with majority class (?) in terms of log odds? z. B. log(2/1) log(yes / no)
![[Pasted image 20230421100927.png]]
(0.5 is constant from above ) (ybar is the average label over all classes)
![[Pasted image 20230421102232.png]]
(example)

TODO: calculate manually. Similar to:
![[Pasted image 20230421102928.png]]

![[Pasted image 20230421102951.png]]

![[Pasted image 20230421103126.png]]


![[Pasted image 20230420143656.png]]


![[Pasted image 20230421103451.png]]

Here (even simpler, just add as there is only a single term)

![[Pasted image 20230421103540.png]]

(hard to solve for. Use newton-Raphson step instead)

![[Pasted image 20230421104203.png]]
(cant find this in the original paper)

![[Pasted image 20230421104256.png]]
![[Pasted image 20230421104418.png]]
Next, update the model = add tree

![[Pasted image 20230421104525.png]]

The final prediction is given in terms of log-odds. To obtain class probabilities, we can use logistic function?
![[Pasted image 20230421104723.png]]

for integration of regularization term see: https://arxiv.org/pdf/2001.07248.pdf

![[Pasted image 20230421145319.png]]
![[Pasted image 20230421145359.png]]