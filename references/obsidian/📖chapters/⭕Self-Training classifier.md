### Semi-supervised Methods

Our supervised approaches depend on the availability of the trade initiator as the true label. Yet, obtaining the label is often restricted to the rare cases, where the trade initiator is provided by the exchange or for subsets of trades where the initiator can be inferred from matching procedures, which may bias the selection. Unlabelled trades, though, are abundant and can help to improve generalization performance of the classifier.

Semi-supervised methods leverage partially-labelled data by learning an algorithm on unlabelled instances alongside with true labels ([[@chapelleSemisupervisedLearning2006]]6) ([[@zhuSemiSupervisedLearningLiterature]]6). They are centred around the assumption of *smoothness*, which states that if two samples $\boldsymbol{x}_{1}$ and $\boldsymbol{x}_{2}$ are nearby in a high-density region, their class labels $y_{1}$ and $y_{2}$ should also be similar. Vice versa, if datapoints are separated by a low-density region, their labels may be different ([[@chapelleSemisupervisedLearning2006]] 5). 

Applied to trade classification, we implicitly assume that trades with similar features, such as a common trade price and quotes, conform to the same class. The purpose of unlabelled trades is help efficiently determine the boundary around regions of neighbouring trades resulting in an improved classification.

The semi-supervised setting requires to extend our notation from cref-problem framing, as we have to divide the dataset into labelled and unlabelled instances. Henceforth, $\boldsymbol{X}_{l} = \left[\boldsymbol{x}_{1},\ldots, \boldsymbol{x}_{l}\right]$  is used for instances, where the label $\boldsymbol{y}_{l} = \left[y_{1},\ldots, y_{l}\right]^{\top}$  is known and $\boldsymbol{X}_{u} = \left[\boldsymbol{x}_{l+1},\ldots, \boldsymbol{x}_{l+u}\right]^{\top}$ for unlabelled datapoints. Like before, all trades are ordered by the trade time.

Our coverage of semi-supervised methods includes *self-training classifiers* for gradient-boosting and *pre-training* of Transformers. We start with the  the self-training paradigm.

## Discussion

Self-training is advantegous, as it is widely adapted approach and doesn't require to alter the base algorithm, which maintains comparability with other approaches. Also the self-training paradigm is model-agnostic.


Self-training, also known as decision-directed or self-taught learning machine, is one of the earliest approach in semi-supervised learning $[41,19]$ that has risen in popularity in recent years. (https://arxiv.org/pdf/2202.12040.pdf)

---

### Self-Training
Self-training is a model-agnostic wrapper algorithm around a probabilistic classifier, that uses its own predictions as pseudo labels for unlabelled instances ([[@yarowskyUnsupervisedWordSense1995]]190). 

The classifier is initially fitted on labelled datapoints. The classifier is then used to assign labels, so-called pseudo labels, to the unlabelled instances. A subset of unlabelled instances with high-confidence predictions is selected, removed from the unlabelled dataset and added to to the labelled data. A new classifier is then retrained on the the labelled and pseudo-labelled instances ([[@yarowskyUnsupervisedWordSense1995]]190--192). The process is repeated for several iterations until a stopping criterion applies, such as the number of iterations is maxed out or no unlabelled instances are left for labelling. The complete algorithm is documented in cref-algo.

$$
\begin{aligned}
&\text { Algorithm 1. Self-Training }\\
&\begin{aligned}
& \text { Input }: S=\left(\mathbf{x}_i, y_i\right)_{1 \leqslant i \leqslant m}, X_{\mathcal{U}}=\left(\mathbf{x}_i\right)_{m+1 \leqslant i \leqslant m+u} \\
& k \leftarrow 0, X_{\mathcal{Q}} \leftarrow \emptyset \\
& \text { repeat } \\
& \quad \text { Train } f^{(k)} \text { on } S \cup X_{\mathcal{Z}} \\
& \quad \Pi_k \leftarrow\left\{\Phi_{\ell}\left(\mathbf{x}, f^{(k)}\right), \mathbf{x} \in X_{\mathcal{U}}\right\} \triangleright \text { Pseudo-labeling } \\
& X_{\mathcal{X}} \leftarrow X_{\mathcal{Q}} \cup \Pi_k \\
& X_{\mathcal{U}} \leftarrow X_{\mathcal{U}} \backslash\left\{\mathbf{x} \mid(\mathbf{x}, \tilde{y}) \in \Pi_k\right\} \\
& \quad k \leftarrow k+1 \\
& \text { until } X_{\mathcal{U}}=\emptyset \vee \Pi_k=\emptyset \\
& \text { Output }: f^{(k)}, X_{\mathcal{U}}, X_{\mathcal{X}}
\end{aligned}
\end{aligned}

$$

**Short discussion:**

**Notation:**
Combining self-training with gradient-boosting (cref-[[🐈Gradient Boosting]]), in each training iteration the classifier $h$ now minimizes the cross-entropy loss over $S$ and $X_{Q}$ jointly:
$$
\frac{1}{l} \sum_{(\boldsymbol{x}, y) \in S} \mathcal{L}(h(\boldsymbol{x}), y)+\frac{\gamma}{\left|X_Q\right|} \sum_{(\boldsymbol{x}, \tilde{y}) \in X_\alpha} \mathcal{L}(h(\boldsymbol{x}), \tilde{y})+\lambda\|h\|^2,
$$
where $\gamma$ is hyperparameter to control the impact of the pseudo-labelled data and $\lambda$ is a regularization parameter ([[@aminiSelfTrainingSurvey2023]]4). (Haffari G, Sarkar A (2007) Analysis of semi-supervised learning with the yarowsky algorithm. In: Uncertainty in Artificial Intelligence (UAI), pp 159–166) -> In general, self-training is a wrapper algorithm, and is hard to analyze. However, for specific base classifiers, theoretical analysis is feasible, for example [17] showed that the Yarowsky algorithm [45] minimizes an upper bound on a new definition of cross entropy based on a specific instantiation of the Bregman distance. (found in [[@tanhaSemisupervisedSelftrainingDecision2017]])

In every round only unlabelled instances are added to the training set, for which the predicted class probability exceeds a confidence threshold $\tau$. This has implications, as raised by  ([[@chenDebiasedSelfTrainingSemiSupervised2022]]2). The threshold $\tau$ becomes an important hyperparameter in controlling that no low-confidence labels are added to the training set, but a restriction to highly-confidence samples may lead to a *data bias* and over-confidence in the prediction. Self-training is prone to a *confirmation bias*, as confident but wrong pseudo labels are erroneously incorporated into the training set, which in effect leads to an propagation of errors in the subsequent training iterations. (What is a remedy?) Self-training requires callibrated probablities, which is problematic for decision trees ([[@tanhaSemisupervisedSelftrainingDecision2017]]). Not problematic for gradient-boosting, as we directly optimize the cross-entropy loss.

Also, self-training increases computational cost, as training is repeated over several iterations on a growing training set ([[@zophRethinkingPretrainingSelftraining2020]]9). Despite these limitations, the potentially improved decision boundary outweights the concerns. To to best of our knowledge, the application of self-training is novel in the context of trade classification.

An alternative to self-training is pre-training, which we study next in the context of Transformers.

**Notes:**
[[⭕Self-Training classifier notes]]