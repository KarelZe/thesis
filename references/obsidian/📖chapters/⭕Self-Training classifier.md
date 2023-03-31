



name of study `17malsep`

The application of semi-supervised is new in the context of trade classification with a on supervised or unsupervised methods.

[[@chapelleSemisupervisedLearning2006]]

An increasingly popular pre-training method is self-supervised learning. Self-supervised learning methods pre-train on a dataset without using labels with the hope to build more universal representations that work across a wider variety of tasks and datasets. 

“The goal ofsemi-sup ervised classiØcation istouseunlabeleddata toimpro vethegeneralization. The cluster assumption states that thedecision boundary should notcross high densityregions, butinstead lieinlow densityregions.” (Chapelle and Zien, 2005, p. 1)


> At each iteration, the self-training selects just a portion of unlabeled data for pseudolabeling, otherwise, all unlabeled examples would be pseudo-labeled after the first iteration, which would actually result in a classifier with performance identical to the
initial classifier [10]. 

This wrapper algorithm starts by learning a supervised classifier on the labeled training set $S$. Then, at each iteration, the current classifier selects a part of the unlabeled data, $X_Q$, and assigns pseudo-labels to them using the classifier's predictions. These pseudo-labeled unlabeled examples are removed from $X_{\mathcal{U}}$ and a new supervised classifier is trained over $S \cup X_Q$, by considering these pseudo-labeled unlabeled data as additional labeled examples. To do so, the classifier $h \in \mathcal{H}$ that is learned at the current iteration is the one that minimizes a regularized empirical loss over $S$ and $X_Q$ :
$$
\frac{1}{m} \sum_{(\mathbf{x}, y) \in S} \ell(h(\mathbf{x}), y)+\frac{\gamma}{\left|X_Q\right|} \sum_{(\mathbf{x}, \tilde{y}) \in X_\alpha} \ell(h(\mathbf{x}), \tilde{y})+\lambda\|h\|^2
$$
where $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow[0,1]$ is an instantaneous loss most often chosen to be the crossentropy loss, $\gamma$ is a hyperparameter for controlling the impact of pseudo-labeled data in learning, and $\lambda$ is the regularization hyperparameter. This process of pseudo-labeling and learning a new classifier continues until the unlabeled set $X_{\mathcal{U}}$ is empty or there is no more unlabeled data to pseudo-label. The pseudo-code of the self-training algorithm is shown in Algorithm 1.

The entire algorithm is reported in


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

**Notes:**
[[⭕Self-Training classifier notes]]