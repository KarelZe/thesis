
The quality of the classifier is better, the higher the share of trades is where the predicted trade initiator equals the true trade initiator. We assess the quality our model’s prediction in terms of *accuracy*. Formally, accuracy can be stated as:
$$
\operatorname{Accuracy} = 1 - \mathcal{L}_{0-1}(\boldsymbol{y}, \hat{\boldsymbol{y}})
$$
where $\mathcal{L}_{0-1}(\cdot)$ is the zero-one loss given by:
$$
 \mathcal{L}_{0-1}(\boldsymbol{y}, \hat{\boldsymbol{y}}) = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(\boldsymbol{y}_{i}\neq \hat{\boldsymbol{y}}_{i}\right).
$$Intuitively, from the  zero-one loss we obtain the error rate on the dataset, as for every misclassified trade we count a loss of  one and normalise by the number of samples $N$, which gives use the normalised zero-one loss.

Our dataset is balanced and buyer-initiated trades predicted as seller, hence --*false positives* -- and buyer-initiated trades predicted as seller -- *false negatives* -- have similar associated costs, which makes the accuracy a reasoned choice as a performance metric. As the zero-one loss and in consequence the accuracy is not differentiable, we cannot use it in optimisation, but use it as as a early stopping criterion to halt training or as an optimisation target in the hyperparameter search. We report the accuracy on the test set.

Additionally, due to the probabilistic framing, we estimate the binary cross-entropy loss, which gives insights into the uncertainty of the predicted class beyond the mere correctness of the prediction. As classes are $y=\{-1,1\}$, we set the loss is given by:

<mark style="background: #FF5582A6;">(todo)</mark>
<mark style="background: #ADCCFFA6;">(make clear what are logits / probs)</mark>

Another loss criterion with the same population minimizer is the binomial negative log-likelihood or deviance (also known as cross-entropy), interpreting $f$ as the logit transform. Let
$$
p(x)=\operatorname{Pr}(Y=1 \mid x)=\frac{e^{f(x)}}{e^{-f(x)}+e^{f(x)}}=\frac{1}{1+e^{-2 f(x)}}
$$
and define $Y^{\prime}=(Y+1) / 2 \in\{0,1\}$. Then the binomial log-likelihood loss function is
$$
l(Y, p(x))=Y^{\prime} \log p(x)+\left(1-Y^{\prime}\right) \log (1-p(x))
$$
or equivalently the deviance is
$$
-l(Y, f(x))=\log \left(1+e^{-2 Y f(x)}\right)
$$
Since the population maximizer of log-likelihood is at the true probabilities $p(x)=\operatorname{Pr}(Y=1 \mid x)$, we see from (10.17) that the population minimizers of the deviance $\mathrm{E}_{Y \mid x}[-l(Y, f(x))]$ and $\mathrm{E}_{Y \mid x}\left[e^{-Y f(x)}\right]$ are the same. Thus, using either criterion leads to the same solution at the population level. Note that $e^{-Y f}$ itself is not a proper log-likelihood, since it is not the logarithm of any probability mass function for a binary random variable $Y \in\{-1,1\}$. ([[@hastietrevorElementsStatisticalLearning2009]] 365)

![[Pasted image 20230528103328.png]]
https://yuan-du.com/post/2020-12-13-loss-functions/decision-theory/
(not sure if correct. Might be the logits!)
$$
\mathcal{L}_{\text{BCE}}(\mathbf{y}, \mathbf{p}) = \frac{1}{N} \sum_{i=1}^N \log(1 + \exp(-2\mathbf{y}_i \mathbf{p}_i))
$$
where $\mathbf{p}$ is a vector of the predicted class probablities

It is a measure of uncertainty ([you may call it entropy](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)), so a low Log Loss means a low uncertainty/entropy of your model. Log Loss is similar to the Accuracy, but it will favor models that distinguish more strongly the classes.
**_Log Loss it useful to compare models not only on their output but on their probabilistic outcome._**
https://medium.com/@fzammito/whats-considered-a-good-log-loss-in-machine-learning-a529d400632d

**Notes:**
[[🧭Evaluation metric notes]]