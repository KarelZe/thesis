
The quality of the classifier is better, the higher the share of trades is where the predicted trade initiator equals the true trade initiator. We assess the quality our model’s prediction in terms of *accuracy*. Formally, accuracy can be stated as:
$$
\operatorname{Accuracy} = 1 - \mathcal{L}_{0-1}(\boldsymbol{y}, \hat{\boldsymbol{y}})
$$
where $\mathcal{L}_{0-1}(\cdot)$ is the zero-one loss given by:
$$
 \mathcal{L}_{0-1}(\boldsymbol{y}, \hat{\boldsymbol{y}}) = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(\boldsymbol{y}_{i}\neq \hat{\boldsymbol{y}}_{i}\right).
$$Intuitively, from the  zero-one loss we obtain the error rate on the dataset, as for every misclassified trade we count a loss of  one and normalise by the number of samples $N$, which gives use the normalised zero-one loss.

Our dataset is balanced and buyer-initiated trades predicted as seller, hence --*false positives* -- and buyer-initiated trades predicted as seller -- *false negatives* -- have similar associated costs, which makes the accuracy a reasoned choice as a performance metric. As the zero-one loss and in consequence the accuracy is not differentiable, we cannot use it in optimisation, but use it as as a early stopping criterion to halt training or as an optimisation target in the hyperparameter search. We report the accuracy on the test set.

**Notes:**
[[🧭Evaluation metric notes]]