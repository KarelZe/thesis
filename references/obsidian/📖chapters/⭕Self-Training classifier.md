



Our supervised approaches depend on the availability of the trade initiator as the true label. Yet, obtaining the label is often restricted to the rare cases, where the trade initiator is provided by the exchange or to subsets of trades where the label can be inferred through matching procedures, which may bias the selection. Unlabelled trades, though, are abundant  and may improve generalization performance of the classifier. Semi-supervised methods leverage partially labelled data by learning an algorithm on unlabelled instances alongside with true labels ([[@chapelleSemisupervisedLearning2006]]6). 

For semi-supervised approaches to


Central hypothesis. The basic assumption in semi-supervised learning, called smoothness, stipulates that two examples in a high density region should have identical class labels [10, 2]. This means that if two points are part of the same group or cluster, their class labels will most likely be the same. If they are separated by a low density zone, on the other hand, their desired labels should be different. Hence, if the examples of the same class form a partition, unlabeled training data might aid in determining the partition boundary more efficiently than if just labeled training examples were utilized. (https://arxiv.org/pdf/2202.12040.pdf)

“However, there is an important prerequisite: that the distribution of examples, which the unlabeled data will help elucidate, be relevant for the classification problem. In a more mathematical formulation, one could say that the knowledge on p(x) that one gains through the unlabeled data has to carry information that is useful in the inference of p(y|x). If this is not the case, semi-supervised learning will not yield an improvement over supervised learning. It might even happen that using the unlabeled data degrades the prediction accuracy by misguiding the inference; this effect is investigated in detail in chapter 4.” ([[@chapelleSemisupervisedLearning2006]], 2006, p. 19)


In this survey we focus on self-training algorithms that follow this principle by assigning pseudo-labels to high-confidence unlabeled training examples and include these pseudo-labeled samples in the learning process

The application of semi-supervised methods is novel in trade classification.

Semi-supervised learning makes several 


*pseudo labels* / *proxy lbels* Other than true labels 



In this blog post, I will focus on a particular class of semi-supervised learning algorithms that produce _proxy labels_ on unlabelled data, which are used as targets together with the labelled data. These proxy labels are produced by the model itself or variants of it without any additional supervision; they thus do not reflect the ground truth but might still provide some signal for learning. In a sense, these labels can be considered _noisy_ or _weak_. I will highlight the connection to learning from noisy labels, weak supervision as well as other related topics in the end of this post. (https://www.ruder.io/semi-supervised/)


For trade classification the true label are frequently unknown, as the trade initiator is not provided by the exchange or labelling strategies work only on subsets of trades, such as . Unlabelled trades are readily available and of potential 

Basic assumption 
An increasingly popular pre-training method is self-supervised learning. Self-supervised learning methods pre-train on a dataset without using labels with the hope to build more universal representations that work across a wider variety of tasks and datasets. (komischer Thesis)



Self-training, also known as decision-directed or self-taught learning machine, is one of the earliest approach in semi-supervised learning $[41,19]$ that has risen in popularity in recent years. (https://arxiv.org/pdf/2202.12040.pdf)

“The goal ofsemi-sup ervised classiØcation istouseunlabeleddata toimpro vethegeneralization. The cluster assumption states that thedecision boundary should notcross high densityregions, butinstead lieinlow densityregions.” (Chapelle and Zien, 2005, p. 1)


> At each iteration, the self-training selects just a portion of unlabeled data for pseudolabeling, otherwise, all unlabeled examples would be pseudo-labeled after the first iteration, which would actually result in a classifier with performance identical to the
initial classifier [10]. ([[@chapelleSemisupervisedLearning2006]])

Self-training is a common algorithmic paradigm for leveraging unlabeled data with deep networks. Self-training methods train a model to fit pseudolabels, that is, predictions on unlabeled data made by a previously-learned model (Yarowsky, 1995; Grandvalet & Bengio, 2005; Lee, 2013). Recent work also extends these methods to enforce stability of predictions under input transformations such as adversarial perturbations (Miyato et al., 2018) and data augmentation (Xie et al., 2019). These approaches, known as input consistency regularization, have been successful in semi-supervised learning (Sohn et al., 2020; Xie et al., 2020), unsupervised domain adaptation (French et al., 2017; Shu et al., 2018), and unsupervised learning (Hu et al., 2017; Grill et al., 2020). (https://arxiv.org/pdf/2010.03622.pdf)

This wrapper algorithm starts by learning a supervised classifier on the labeled training set $S$. Then, at each iteration, the current classifier selects a part of the unlabeled data, $X_Q$, and assigns pseudo-labels to them using the classifier's predictions. These pseudo-labeled unlabeled examples are removed from $X_{\mathcal{U}}$ and a new supervised classifier is trained over $S \cup X_Q$, by considering these pseudo-labeled unlabeled data as additional labeled examples. To do so, the classifier $h \in \mathcal{H}$ that is learned at the current iteration is the one that minimizes a regularized empirical loss over $S$ and $X_Q$ :
$$
\frac{1}{m} \sum_{(\mathbf{x}, y) \in S} \ell(h(\mathbf{x}), y)+\frac{\gamma}{\left|X_Q\right|} \sum_{(\mathbf{x}, \tilde{y}) \in X_\alpha} \ell(h(\mathbf{x}), \tilde{y})+\lambda\|h\|^2
$$
where $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow[0,1]$ is an instantaneous loss most often chosen to be the crossentropy loss, $\gamma$ is a hyperparameter for controlling the impact of pseudo-labeled data in learning, and $\lambda$ is the regularization hyperparameter. This process of pseudo-labeling and learning a new classifier continues until the unlabeled set $X_{\mathcal{U}}$ is empty or there is no more unlabeled data to pseudo-label. The pseudo-code of the self-training algorithm is shown in Algorithm 1.

In this blog post, I will focus on a particular class of semi-supervised learning algorithms that produce _proxy labels_ on unlabelled data, which are used as targets together with the labelled data. These proxy labels are produced by the model itself or variants of it without any additional supervision; they thus do not reflect the ground truth but might still provide some signal for learning. In a sense, these labels can be considered _noisy_ or _weak_. I will highlight the connection to learning from noisy labels, weak supervision as well as other related topics in the end of this post. (https://www.ruder.io/semi-supervised/)

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

**Short discussion:**

Despite the empirical successes, theoretical progress in understanding how to use unlabeled data has lagged. Whereas 1 arXiv:2010.03622v5 [cs.LG] 20 Apr 2022 supervised learning is relatively well-understood, statistical tools for reasoning about unlabeled data are not as readily available. (https://arxiv.org/pdf/2010.03622.pdf)




An unsatisfactory aspect of self-learning is that the effect of the wrapper depends on the supervised method used inside it. If self-learning is used with empirical risk minimization and 1-0-loss, the unlabeled data will have no effect on the solution at all. If instead a margin maximizing method is used, as a result the decision boundary is pushed away from the unlabeled points (cf. chapter 6). In other cases it seems to be unclear what the self-learning is really doing, and which assumption it corresponds to.” ([[@chapelleSemisupervisedLearning2006]] 2006, p. 18)

In general, self-training is a wrapper algorithm, and is hard to analyze. However, for specific base classifiers, theoretical analysis is feasible, for example [17] showed that the Yarowsky algorithm [45] minimizes an upperbound on a new definition of cross entropy based on a specific instantiation of the Bregman distance. In this paper, we focus on using a decision tree learner as the base learner in self-training. We show that improving the probability estimation of the decision trees will improve the performance of a self-training algorithm. ([[@tanhaSemisupervisedSelftrainingDecision2017]])

There are also limitations to Self-training particularly because it requires much computation [[@zophRethinkingPretrainingSelftraining2020]]. Apart from that, the distribution of pseudo labels during one iteration may not match the distribution of labeled data during the Self-training process and result in poor performance [32]. To balance class distributions, data in majority classes are randomly discarded and data in minority classes are duplicated and are taken with the highest confidence [41]. What’s more, error propagation is an inevitable problem since pseudo labels are predicted by the teacher classifier instead of true labels. Random errors exist and will be propagated while using pseudo labels to train the student classifier. However the problem can be mitigated by adding noise to student model [41] and error forgetting [22]. Admittedly, limitations of large computation power needs, inevitable error propagation, and potential sub-optimal model training due to uneven distribution of pseudo labels exist when practicing Self-training. Advantages of inferring a better decision boundary than its teacher model, being flexible and scalable on data sources and quantity, and improving model performance, robustness and generalization when 9 utilizing a large amount of unlabeled data that Self-training can bring certainly worth further examination. (https://www.diva-portal.org/smash/get/diva2:1629474/FULLTEXT02)

## Discussion

**Transition to Pre-Training:**

Application possible to neural nets, but very

Our work argues for the scalability and generality of self-training (e.g., [8–10]). Recently, selftraining has shown significant progress in deep learning (e.g., image classification [11, 12], machine translation [13], and speech recognition [14, 47]). Most closely related to our work is Xie et al. [12] who also use strong data augmentation in self-training but for image classification. Closer in applications are semi-supervised learning for detection and segmentation (e.g., [48–52]), who only study self-training in isolation or without a comparison against ImageNet pre-training. They also do not consider the interactions between these training methods and data augmentations. ([[@zophRethinkingPretrainingSelftraining2020]])


elf-training is a common algorithmic paradigm for leveraging unlabeled data with deep networks. Self-training methods train a model to fit pseudolabels, that is, predictions on unlabeled data made by a previously-learned model (Yarowsky, 1995; Grandvalet & Bengio, 2005; Lee, 2013). Recent work also extends these methods to enforce stability of predictions under input transformations such as adversarial perturbations (Miyato et al., 2018) and data augmentation (Xie et al., 2019). These approaches, known as input consistency regularization, have been successful in semi-supervised learning (Sohn et al., 2020; Xie et al., 2020), unsupervised domain adaptation (French et al., 2017; Shu et al., 2018), and unsupervised learning (Hu et al., 2017; Grill et al., 2020). (https://arxiv.org/pdf/2010.03622.pdf)


Limitations. There are still limitations to current self-training techniques. In particular, self-training requires more compute than fine-tuning on a pre-trained model. The speedup thanks to pre-training ranges from 1.3x to 8x depending on the pre-trained model quality, strength of data augmentation, and dataset size. Good pre-trained models are also needed for low-data applications like PASCAL segmentation. The scalability, generality and flexibility of self-training. Our experimental results highlight important advantages of self-training. First, in terms of flexibility, self-training works well in every setup that we tried: low data regime, high data regime, weak data augmentation and strong data augmentation. Self-training also is effective with different architectures (ResNet, EfficientNet, SpineNet, FPN, NAS-FPN), data sources (ImageNet, OID, PASCAL, COCO) and tasks (Object Detection, Segmentation). Secondly, in terms of generality, self-training works well even when pre-training fails but also when pre-training succeeds. In terms of scalability, self-training proves to perform well as we have more labeled data and better models. One bitter lesson in machine learning is that most methods fail when we have more labeled data or more compute or better supervised training recipes, but that does not seem to apply to self-training. ([[@zophRethinkingPretrainingSelftraining2020]])

Self-training works across dataset sizes and is additive to pre-training. Next we analyze the performance of self-training as we vary the COCO labeled dataset size. As can be seen from Table 3, self-training benefits object detectors across dataset sizes, from small to large, regardless of pretraining methods. Most importantly, at the high data regime of 100% labeled set size, self-training significantly improves all models while pre-training hurts. In the low data regime of 20%, self-training enjoys the biggest gain of +3.4AP on top of Rand Init. This gain is bigger than the gain achieved by ImageNet Init (+2.6AP). Although the self-training gain is smaller than the gain by ImageNet++ Init, ImageNet++ Init uses 300M additional unlabeled images. Self-training is quite additive with pre-training even when using the same data source. For example, in the 20% data regime, utilizing an ImageNet pre-trained checkpoint yields a +2.6AP boost. Using both pre-training and self-training with ImageNet yields an additional +2.7AP gain. The additive benefit of combining pre-training and self-training is observed across all of the dataset sizes. ([[@zophRethinkingPretrainingSelftraining2020]])

**Notes:**
[[⭕Self-Training classifier notes]]