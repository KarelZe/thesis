

The application of semi-supervised methods is novel in trade classification.


----

Central hypothesis. The basic assumption in semi-supervised learning, called smoothness, stipulates that two examples in a high density region should have identical class labels [10, 2]. This means that if two points are part of the same group or cluster, their class labels will most likely be the same. If they are separated by a low density zone, on the other hand, their desired labels should be different. Hence, if the examples of the same class form a partition, unlabeled training data might aid in determining the partition boundary more efficiently than if just labeled training examples were utilized



In semi-supervised learning there is a small set of labeled data and a large pool of unlabeled data. Data points are divided into the points Xl = (x1,x2...,xl), for which labels Yl = {?1,-1} are provided, and the points Xu = (xlþ1,xlþ2; ...; xlþu), the labels of which are not known. We assume that labeled and unlabeled data are drawn independently from the same data distribution. In this paper we consider datasets for which nl nu, where nl and nu are the number of labeled data and unlabeled data respectively. 3.2 The self-training algorithm The self-training algorithm wraps around a base classifier and uses its own predictions through the training process [48]. A base learner is first trained on a small number of labeled examples, the initial training set. The classifier is then used to predict labels for unlabeled examples (prediction step) based on the classification confidence. Next, a subset S of the unlabeled examples, together with their predicted labels, is selected to train a new classifier (selection step). Typically, S consists of a few unlabeled examples with high-confidence predictions. The classifier is then re-trained on the new set of labeled examples, and the procedure is repeated (re-training step) until it reaches a stopping condition. As a base learner, we employ the decision tree classifier in self-training. The most wellknown algorithm for building decision trees is the C4.5 algorithm [32], an extension of Quinlan’s earlier ID3 algorithm. Decision trees are one of the most widely used classification methods, see [5, 10, 16]. They are fast and effective in many domains. They work well with little or no tweaking of parameters which has made them a popular tool for many domains [31]. This has motivated us to find a semi-supervised method for learning decision trees. Algorithm 1 presents the main structure of the self-training algorithm. (From https://link.springer.com/content/pdf/10.1007/s13042-015-0328-7.pdf?pdf=button)




Self-training. Our work is based on self-training (e.g., [71, 96, 68, 67]). Self-training first uses labeled data to train a good teacher model, then use the teacher model to label unlabeled data and finally use the labeled data and unlabeled data to jointly train a student model. In typical self-training with the teacher-student framework, noise injection to the student is not used by default, or the role of noise is not fully understood or justified. The main difference between our work and prior works is that we identify the importance of noise, and aggressively inject noise to make the student better. (https://arxiv.org/pdf/1911.04252.pdf)



----


Previous algorithms require

Self-training classifiers are a wrapper algorithm around a probabilistic classifier. They incorporate own predicted label  

Initially th e


Read https://arxiv.org/pdf/2010.03622.pdf

Self-training is a common algorithmic paradigm for leveraging unlabeled data with deep networks. Self-training methods train a model to fit pseudolabels, that is, predictions on unlabeled data made by a previously-learned model (Yarowsky, 1995; Grandvalet & Bengio, 2005; Lee, 2013).




- Possible extension could be [[@yarowskyUnsupervisedWordSense1995]]. See also Sklearn Self-Training Classifier.
https://ai.stanford.edu/blog/understanding-self-training/

Self-Training is not fully understood theoretically. See here: https://arxiv.org/pdf/2010.03622.pdf


- See [[@tanhaSemisupervisedSelftrainingDecision2017]] for discussion of self-training in conjunction with decision trees and random forests.
- pseudocode for self-supervised algorithm can be found in [[@tanhaSemisupervisedSelftrainingDecision2017]].
The self-training algorithm wraps around a base classifier and uses its own predictions through the training process [48]. A base learner is first trained on a small number of labeled examples, the initial training set. The classifier is then used to predict labels for unlabeled examples (prediction step) based on the classification confidence. Next, a subset S of the unlabeled examples, together with their predicted labels, is selected to train a new classifier (selection step). Typically, S consists of a few unlabeled examples with high-confidence predictions. The classifier is then re-trained on the new set of labeled examples, and the procedure is repeated (re-training step) until it reaches a stopping condition. As a base learner, we employ the decision tree classifier in self-training. The most wellknown algorithm for building decision trees is the C4.5 algorithm [32], an extension of Quinlan’s earlier ID3 algorithm. Decision trees are one of the most widely used classification methods, see [5, 10, 16]. They are fast and effective in many domains. They work well with little or no tweaking of parameters which has made them a popular tool for many domains [31]. This has motivated us to find a semi-supervised method for learning decision trees. Algorithm 1 presents the main structure of the self-training algorithm.
The goal of the selection step in Algorithm 1 is to find a set unlabeled examples with high-confidence predictions, above a threshold T. This is important, because selection of incorrect predictions will propagate to produce further classification errors. At each iteration the newly-labeled instances are added to the original labeled data for constructing a new classification model. The number of iterations in Algorithm 1 depends on the threshold T and also on the pre-defined maximal number of iterations, Itermax.


- 
- ![[pseudocode-selftraining.png]]
Explanation of Wrapper method
![[1680261360735.jpg]]

## Rough classification
https://researchcommons.waikato.ac.nz/bitstream/handle/10289/14678/thesis.pdf?sequence=4&isAllowed=y
![[Pasted image 20230326101313.png]]

## Wrapper Methods
https://researchcommons.waikato.ac.nz/bitstream/handle/10289/14678/thesis.pdf?sequence=4&isAllowed=y
The most widely used and oldest semi-supervised learning algorithms are based
on wrapper methods (Zhu, 2005). They employ one or more base learners and
iteratively use their confident predictions for retraining. In practice, the base
learner is first trained on the small set of available labelled data and employed
to predict labels for unlabelled data–commonly referred to as pseudo-labels.
One can use single or multiple base learners on the same or different subset
of the features. Self-training (also known as self-learning) is the most basic
approach based on the wrapper idea. A single classifier is trained iteratively
on initially labelled instances and employed to predict labels for unlabelled
instances. Self-training has been successfully applied to object detection problems in the era preceding deep learning (Rosenberg et al., 2005) and achieved
state-of-the-art. Considering deep learning, Pseudo-Label (Lee, 2013) is a simple self-training approach based on neural networks. A classifier is trained on
the initially labelled and pseudo-labelled data, starting with a small weight
of pseudo-labelled data. Pseudo-labels are less reliable at the start of training; therefore, the weights of examples with pseudo-labels are increased as the
training progresses.
There are different design decisions offered by self-training. This includes
the selection of pseudo-labels, reusing pseudo-labels for later iterations, and
stopping criteria (Rosenberg et al., 2005; Triguero et al., 2015). The selection of pseudo-labelled data has a significant impact on the performance of


##  Blog Sebastian Ruder
**Link** https://www.ruder.io/semi-supervised/

**Broad overview:**
Semi-supervised learning has a long history. For a (slightly outdated) overview, refer to Zhu (2005) [1] and Chapelle et al. (2006) [2]. Particularly recently, semi-supervised learning has seen some success, considerably reducing the error rate on important benchmarks. Semi-supervised learning also makes an appearance in [Amazon's annual letter to shareholders](https://www.sec.gov/Archives/edgar/data/1018724/000119312518121161/d456916dex991.htm?ref=ruder.io) where it is credited with reducing the amount of labelled data needed to achieve the same accuracy improvement by \(40\times\).

In this blog post, I will focus on a particular class of semi-supervised learning algorithms that produce _proxy labels_ on unlabelled data, which are used as targets together with the labelled data. These proxy labels are produced by the model itself or variants of it without any additional supervision; they thus do not reflect the ground truth but might still provide some signal for learning. In a sense, these labels can be considered _noisy_ or _weak_. I will highlight the connection to learning from noisy labels, weak supervision as well as other related topics in the end of this post. (https://www.ruder.io/semi-supervised/)

**Self-Training:**
Self-training (Yarowsky, 1995; McClosky et al., 2006) [4] [5] is one of the earliest and simplest approaches to semi-supervised learning and the most straightforward example of how a model's own predictions can be incorporated into training. As the name implies, self-training leverages a model's own predictions on unlabelled data in order to obtain additional information that can be used during training. Typically the most confident predictions are taken at face value, as detailed next.

Formally, self-training trains a model \(m\) on a labeled training set \(L\) and an unlabeled data set \(U\). At each iteration, the model provides predictions \(m(x)\) in the form of a probability distribution over the \(C\) classes for all unlabeled examples \(x\) in \(U\). If the probability assigned to the most likely class is higher than a predetermined threshold \(\tau\), \(x\) is added to the labeled examples with \(\DeclareMathOperator*{\argmax}{argmax} p(x) = \argmax m(x)\) as pseudo-label. This process is generally repeated for a fixed number of iterations or until no more predictions on unlabelled examples are confident. This instantiation is the most widely used and shown in Algorithm 1.  
![](https://raindrop-preview-prod.exentrich.workers.dev/img?url=https%3A%2F%2Fwww.ruder.io%2Fcontent%2Fimages%2F2018%2F03%2Fself-training.png)  
Classic self-training has shown mixed success. In parsing it proved successful with small datasets (Reichart, and Rappoport, 2007; Huang and Harper, 2009) [6] [7] or when a generative component is used together with a reranker when more data is available (McClosky et al., 2006; Suzuki and Isozaki , 2008) [8]. Some success was achieved with careful task-specific data selection (Petrov and McDonald, 2012) [9], while others report limited success on a variety of NLP tasks (He and Zhou, 2011; Plank, 2011; Van Asch and Daelemans, 2016; van der Goot et al., 2017) [10] [11] [12] [13].

The main downside of self-training is that the model is unable to correct its own mistakes. If the model's predictions on unlabelled data are confident but wrong, the erroneous data is nevertheless incorporated into training and the model's errors are amplified. This effect is exacerbated if the domain of the unlabelled data is different from that of the labelled data; in this case, the model's confidence will be a poor predictor of its performance.


## Self-Learning, self-training, etc.

Self-training is a commonly used technique for semi-supervised learning. In selftraining a classifier is first trained with the small amount of labeled data. The classifier is then used to classify the unlabeled data. Typically the most confident unlabeled points, together with their predicted labels, are added to the training set. The classifier is re-trained and the procedure repeated. Note the classifier uses its own predictions to teach itself. The procedure is also called self-teaching or bootstrapping (not to be confused with the statistical procedure with the same name). The generative model and EM approach of section 2 can be viewed as a special case of ‘soft’ self-training. One can imagine that a classification mistake can reinforce itself. Some algorithms try to avoid this by ‘unlearn’ unlabeled points if the prediction confidence drops below a threshold. Self-training has been applied to several natural language processing tasks. Yarowsky (1995) uses self-training for word sense disambiguation, e.g. deciding whether the word ‘plant’ means a living organism or a factory in a give context. Riloff et al. (2003) uses it to identify subjective nouns. Maeireizo et al. (2004) classify dialogues as ‘emotional’ or ‘non-emotional’ with a procedure involving two classifiers.Self-training has also been applied to parsing and machine translation. Rosenberg et al. (2005) apply self-training to object detection systems from images, and show the semi-supervised technique compares favorably with a stateof-the-art detector. Self-training is a wrapper algorithm, and is hard to analyze in general. However, for specific base learners, there has been some analyzer’s on convergence. See e.g. (Haffari & Sarkar, 2007; Culp & Michailidis, 2007).


“Probably the earliest idea about using unlabeled data in classification is selflearning, which is also known as self-training, self-labeling, or decision-directed self-learning learning. This is a wrapper-algorithm that repeatedly uses a supervised learning method. It starts by training on the labeled data only. In each step a part of the unlabeled points is labeled according to the current decision function; then the supervised method is retrained using its own predictions as additional labeled points. This idea has appeared in the literature already for some time (e.g., Scudder (1965); Fralick (1967); Agrawala (1970)). 

## Disadvantages of self-training
An unsatisfactory aspect of self-learning is that the effect of the wrapper depends on the supervised method used inside it. If self-learning is used with empirical risk minimization and 1-0-loss, the unlabeled data will have no effect on the solution at all. If instead a margin maximizing method is used, as a result the decision boundary is pushed away from the unlabeled points (cf. chapter 6). In other cases it seems to be unclear what the self-learning is really doing, and which assumption it corresponds to.” ([[@chapelleSemisupervisedLearning2006]] 2006, p. 18)

- Does not correct for its own mistakes. Only criterion is, that algorithm is certain, but might still be wrong.

## When to use semi-supervised learning
“In a more mathematical formulation, one could say that the knowledge on p(x) that one gains through the unlabeled data has to carry information that is useful in the inference of p(y|x). If this is not the case, semi-supervised learning will not yield an improvement over supervised learning. It might even happen that using the unlabeled data degrades the prediction accuracy by misguiding the inference; this effect is investigated in detail in chapter 4.” ([[@chapelleSemisupervisedLearning2006]], 2006, p. 19)


## Semi-supervised learning

“Semi-supervised learning (SSL) is halfway between supervised and unsupervised learning. In addition to unlabeled data, the algorithm is provided with some supervision information – but not necessarily for all examples. Often, this information will be the targets associated with some of the examples. In this case, the data standard setting of SSL set X =(xi)i∈[n] can be divided into two parts: the points Xl :=(x1,...,xl), for which labels Yl :=(y1,...,yl) are provided, and the points Xu :=(xl+1,...,xl+u), the labels of which are not known. This is “standard” semi-supervised learning as investigated in this book; most chapters will refer to this setting” (“Semi-supervised learning”, 2006, p. 17)

## Supervised / unsupervised learning

“Traditionally, there have been two fundamentally different types of tasks in machine learning. The first one is unsupervised learning.LetX =(x1,...,xn)beasetofn examples unsupervised learning (or points), where xi ∈ X for all i ∈ [n] := {1,...,n}. Typically it is assumed that the points are drawn i.i.d. (independently and identically distributed) from a common distribution on X. It is often convenient to define the (n × d)-matrix X =(xi⊤)i⊤∈[n] that contains the data points as its rows. The goal of unsupervised learning is to find interesting structure in the data X. It has been argued that the problem of unsupervised learning is fundamentally that of estimating a density which is likely to have generated X. However, there are also weaker forms of unsupervised learning, such as quantile estimation, clustering, outlier detection, and dimensionality reduction. The second task is supervised learning. The goal is to learn a mapping from supervised learning x to y, given a training set made of pairs (xi,yi). Here, the yi ∈ Y are called the labels or targets of the examples xi. If the labels are numbers, y =(yi)i⊤∈[n] denotes the column vector of labels. Again, a standard requirement is that the pairs (xi,yi) are sampled i.i.d. from some distribution which here ranges over X × Y. The task is well defined, since a mapping can be evaluated through its predictive performance on test examples. When Y = R or Y = Rd (or more generally, when the labels are continuous), the task is called regression. Most of this book will focus on classification (there is some work on regression in chapter 23), i.e., the case where y takes values in a finite set (discrete labels). There are two families of algorithms for supervised learning.” ([[@chapelleSemisupervisedLearning2006]], 2006, p. 16)

## Does it always improve?
In general, self-training is a wrapper algorithm, and is hard to analyze. However, for specific base classifiers, theoretical analysis is feasible, for example [17] showed that the Yarowsky algorithm [45] minimizes an upperbound on a new definition of cross entropy based on a specific instantiation of the Bregman distance. In this paper, we focus on using a decision tree learner as the base learner in self-training. We show that improving the probability estimation of the decision trees will improve the performance of a self-training algorithm. ([[@tanhaSemisupervisedSelftrainingDecision2017]])


## High-level idea / required probabilities
A self-training algorithm is an iterative method for semi-supervised learning, which wraps around a base learner. It uses its own predictions to assign labels to unlabeled data. Then, a set of newly-labeled data, which we call a set of high-confidence predictions, are selected to be added to the training set for the next iterations. The performance of the self-training algorithm strongly depends on the selected newly-labeled data at each iteration of the training procedure. This selection strategy is  based on confidence in the predictions and therefore it is vital to self-training that the confidence of prediction, which we will call here probability estimation, is measured correctly. There is a difference between learning algorithms that output a probability distribution, e.g. Bayesian methods, neural networks, logistic regression, marginbased classifiers, and algorithms that are normally seen as only outputting a classification model, like decision trees. Most of the current approaches to self-training utilize the first kind of learning algorithms as the base learner [25, 33, 34, 45]. In this paper we focus on self-training with a decision tree learner as the base learner. The goal is to show how to effectively use a decision tree classifier as the base learner in self-training. ([[@tanhaSemisupervisedSelftrainingDecision2017]])

High dependence on the correct estimate of the predicted probabilitoes. Why is it ok in our case to use self-training? -> Loss function etc.

## High-level idea
https://www.diva-portal.org/smash/get/diva2:1629474/FULLTEXT02
A basic Self-training method consists three steps [41] [42]: 1. Training a teacher model on labeled data. 2. Use the trained teacher model to predict pseudo labels on unlabeled data. 3. Training a student model on a combination of labeled data and pseudo-labeled data. This algorithm iterates by using the student as a teacher to pseudolabel the next batch of unlabeled data and train a new student.

## High-level idea

An increasingly popular pre-training method is self-supervised learning. Self-supervised learning methods pre-train on a dataset without using labels with the hope to build more universal representations that work across a wider variety of tasks and datasets. 

Self-training: We use a simple self-training method inspired by [9, 12, 48, 57] which consists of three steps. First, a teacher model is trained on the labeled data (e.g., COCO dataset). Then the teacher model generates pseudo labels on unlabeled data (e.g., ImageNet dataset). Finally, a student is trained to optimize the loss on human labels and pseudo labels jointly. Our experiments with various hyperparameters and data augmentations indicate that self-training with this standard loss function can be unstable. To address this problem, we implement a loss normalization technique, which is described in Appendix B.

## Pros / Cons

There are also limitations to Self-training particularly because it requires much computation [42]. Apart from that, the distribution of pseudo labels during one iteration may not match the distribution of labeled data during the Self-training process and result in poor performance [32]. To balance class distributions, data in majority classes are randomly discarded and data in minority classes are duplicated and are taken with the highest confidence [41]. What’s more, error propagation is an inevitable problem since pseudo labels are predicted by the teacher classifier instead of true labels. Random errors exist and will be propagated while using pseudo labels to train the student classifier. However the problem can be mitigated by adding noise to student model [41] and error forgetting [22]. Admittedly, limitations of large computation power needs, inevitable error propagation, and potential sub-optimal model training due to uneven distribution of pseudo labels exist when practicing Self-training. Advantages of inferring a better decision boundary than its teacher model, being flexible and scalable on data sources and quantity, and improving model performance, robustness and generalization when 9 utilizing a large amount of unlabeled data that Self-training can bring certainly worth further examination. (https://www.diva-portal.org/smash/get/diva2:1629474/FULLTEXT02)

## Discussion

Limitations. There are still limitations to current self-training techniques. In particular, self-training requires more compute than fine-tuning on a pre-trained model. The speedup thanks to pre-training ranges from 1.3x to 8x depending on the pre-trained model quality, strength of data augmentation, and dataset size. Good pre-trained models are also needed for low-data applications like PASCAL segmentation. The scalability, generality and flexibility of self-training. Our experimental results highlight important advantages of self-training. First, in terms of flexibility, self-training works well in every setup that we tried: low data regime, high data regime, weak data augmentation and strong data augmentation. Self-training also is effective with different architectures (ResNet, EfficientNet, SpineNet, FPN, NAS-FPN), data sources (ImageNet, OID, PASCAL, COCO) and tasks (Object Detection, Segmentation). Secondly, in terms of generality, self-training works well even when pre-training fails but also when pre-training succeeds. In terms of scalability, self-training proves to perform well as we have more labeled data and better models. One bitter lesson in machine learning is that most methods fail when we have more labeled data or more compute or better supervised training recipes, but that does not seem to apply to self-training. ([[@zophRethinkingPretrainingSelftraining2020]])

Self-training works across dataset sizes and is additive to pre-training. Next we analyze the performance of self-training as we vary the COCO labeled dataset size. As can be seen from Table 3, self-training benefits object detectors across dataset sizes, from small to large, regardless of pretraining methods. Most importantly, at the high data regime of 100% labeled set size, self-training significantly improves all models while pre-training hurts. In the low data regime of 20%, self-training enjoys the biggest gain of +3.4AP on top of Rand Init. This gain is bigger than the gain achieved by ImageNet Init (+2.6AP). Although the self-training gain is smaller than the gain by ImageNet++ Init, ImageNet++ Init uses 300M additional unlabeled images. Self-training is quite additive with pre-training even when using the same data source. For example, in the 20% data regime, utilizing an ImageNet pre-trained checkpoint yields a +2.6AP boost. Using both pre-training and self-training with ImageNet yields an additional +2.7AP gain. The additive benefit of combining pre-training and self-training is observed across all of the dataset sizes. ([[@zophRethinkingPretrainingSelftraining2020]])

## Application to Gradient Boostin
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8900737

## Nice notation
https://arxiv.org/pdf/2202.12040.pdf

Self-training, also known as decision-directed or self-taught learning machine, is one of the earliest approach in semi-supervised learning $[41,19]$ that has risen in popularity in recent years. (https://arxiv.org/pdf/2202.12040.pdf)

This wrapper algorithm starts by learning a supervised classifier on the labeled training set $S$. Then, at each iteration, the current classifier selects a part of the unlabeled data, $X_Q$, and assigns pseudo-labels to them using the classifier's predictions. These pseudo-labeled unlabeled examples are removed from $X_{\mathcal{U}}$ and a new supervised classifier is trained over $S \cup X_Q$, by considering these pseudo-labeled unlabeled data as additional labeled examples. To do so, the classifier $h \in \mathcal{H}$ that is learned at the current iteration is the one that minimizes a regularized empirical loss over $S$ and $X_Q$ :
$$
\frac{1}{m} \sum_{(\mathbf{x}, y) \in S} \ell(h(\mathbf{x}), y)+\frac{\gamma}{\left|X_Q\right|} \sum_{(\mathbf{x}, \tilde{y}) \in X_\alpha} \ell(h(\mathbf{x}), \tilde{y})+\lambda\|h\|^2
$$
where $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow[0,1]$ is an instantaneous loss most often chosen to be the crossentropy loss, $\gamma$ is a hyperparameter for controlling the impact of pseudo-labeled data in learning, and $\lambda$ is the regularization hyperparameter. This process of pseudo-labeling and learning a new classifier continues until the unlabeled set $X_{\mathcal{U}}$ is empty or there is no more unlabeled data to pseudo-label. The pseudo-code of the self-training algorithm is shown in Algorithm 1.

3.1 Pseudo-labeling strategies
At each iteration, the self-training selects just a portion of unlabeled data for pseudolabeling, otherwise, all unlabeled examples would be pseudo-labeled after the first iteration, which would actually result in a classifier with performance identical to the
initial classifier [10]. Thus, the implementation of self-training arises the following question: how to determine the subset of examples to pseudo-label?

A classical assumption, that stems from the low density separation hypothesis, is to suppose that the classifier learned at each step makes the majority of its mistakes on observations close to the decision boundary. In the case of binary classification, preliminary research suggested to assign pseudo-labels only to unlabeled observations for which the current classifier is the most confident [46]. Hence, considering thresholds $\theta^{-}$and $\theta^{+}$defined for respectively the negative and the positive classes, the pseudolabeler assigns a pseudo-label $\tilde{y}$ to an instance $\mathrm{x} \in X_{\mathcal{U}}$ such that:
$$
\tilde{y}= \begin{cases}+1, & \text { if } f(x,+1) \geqslant \theta^{+}, \\ -1, & \text { if } f(x,-1) \leqslant \theta^{-},\end{cases}
$$
and $\Phi_{\ell}(\mathbf{x}, f)=(\mathbf{x}, \tilde{y})$. An unlabeled example $\mathbf{x}$ that does not satisfy the conditions (1) is not pseudo-labeled; i.e. $\Phi_{\ell}(\mathbf{x}, f)=\emptyset$.

Intuitively, thresholds should be set to high absolute values as pseudo-labeling examples with low confidence would increase chances of assigning wrong labels. However, thresholds of very high value imply excessive trust in the confidence measure underlying the model, which, in reality, can be biased due to the small labeled sample size. Using several iterations makes also the situation more intricate as at every iteration the optimal threshold might be different. ([[@aminiSelfTrainingSurvey2023]])


## Notes on Yarowsky paper
[[@yarowskyUnsupervisedWordSense1995]]