
All classical trade classification rules from (cref [[🔢Basic rules]]) perform *discrete classification* and assign a class to the trade. Naturally, a more powerful insight is to not just obtain the most probable class, but also the associated class probabilities for a trade to be a buy or sell. This gives additional insights into the confidence of the prediction, but calls for a *probabilistic classifier*.  

Thus, we frame trade signing as a probabilistic classification problem. This is similar to the work of ([[@easleyDiscerningInformationTrade2016]] 272), who alter the tick rule and GLSC-BVC algorithm to obtain the probability estimates of a buy from individual or aggregated trades, but with a sole focus on trade signing on a trade-by-trade basis. The probabilistic view enables us  a richer evaluation in (cref [[🏅Results]]), but constraints our selection to supervised classifiers, capable of producing probability estimates. To maintain comparability, classical trade signing rules need to be modified to yield both the predicted class (buy or sell) and the class probability.

We introduce some more notation, we will use throughout. Each data instances consists of a feature vector and the target. The former is given by $\boldsymbol{x} \in \mathbb{R}^m$ and described by a random variable $X$. Features in $\boldsymbol{x}$ may be numerical, e.g., the previous trade price or categorical e. g., the option type. Like before, the target or trade initiator is given by $y \in \mathcal{Y}$ and described by a random variable $Y$. Each data instance is sampled from a joint probability distribution $p^*(X, Y)$. The training set with $N$ i.i.d. samples drawn from $p^*$ is denoted by $\mathcal{D}_N=\left\{\left(x_i, y_i\right)\right\}_{i=1}^N$. 

For our machine learning classifiers, we aim to model $p_{\theta}(y \mid \boldsymbol{x})$ by fitting a classifier with the parameters $\theta$ on the training set. As classical trade classification rules produce no probability estimates, we use a simple classifier instead:
$$
p(y\mid \boldsymbol{x})= \begin{cases}1, & \text { if } y=\hat{y} \\ 0, & \text { else }.\end{cases}
$$
As such, if a trade predicted as a sell, i. e. $\hat{y} = -1$,  we would assign a probability of $1$ for being a sell, and a probability of zero for being a buy and vice versa. 

Given the estimated class probabilities, we retrieve the most probable class in $\mathcal{Y}$ as:
$$
\hat{y}=\arg\max_{y \in \mathcal{Y}} p(y \mid \mathbf{x}).
$$
Cref-eq and cref-eq allow us to switch between a discrete and probabilistic formulation for  trade classification rules. Since, the class probability estimates are either $0$ or $1$, no insight on the confidence of the prediction is gained. Yet, it provides evaluate classical trade classification rules and probabilistic classifiers in machine learning. In the subsequent section we provide a short discussion to select state-of-the-art classifiers, which we consider for our empirical study.
