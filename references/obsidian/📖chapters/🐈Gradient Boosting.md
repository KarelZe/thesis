
By our problem framing, the focus is on *binary classification*. Other than in the regression case, the target is no longer continuous ( $\mathcal{Y}\in \mathbb{R}$), but rather discrete ($\mathcal{Y}\in \{-1,1\}$). Instead of modelling the class-conditional probabilities directly, we model the conditional *log odds*, which can be interpreted as the probability of observing class $1$ or buyer-initiated trade.  The model now models:
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
![[Pasted image 20230420143656.png]]
![[1680261360742.jpg]]
**scratch pad:**

- understand what log odds
- logistic function
- handling categorical features
- What missing features?

for integration of regularization term see: https://arxiv.org/pdf/2001.07248.pdf

binning them first. This idea by Ke et al. (2017) is called histogram implementation. Thus, the resulting trees grow leaf wise instead of level wise compared to other gradient boosting machines. The concept is shown in Figure 6. It shows that the presorted states are ke

Practical implementations. CatBoost: Symmetric trees (cheap to construct + act as regularizer)

**Notes:**
[[🐈Gradient Boosting notes]]
