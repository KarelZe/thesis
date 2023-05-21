*title:* Semi-Supervised Self-Training for Decision Tree Classifiers
*authors:* Jafar Tanha, Maarten Someren, Hamideh Afsarmanesh
*year:* 2017
*tags:* #semi-supervised #decision-trees #ensembles #random-forest #bagging #self-training #self-supervised 
*status:* #📦 
*related:*
- Decision trees are covered extensively in [[@breimanRandomForests2001]]
- [[@yarowskyUnsupervisedWordSense1995]] (this paper uses yarowskys algorithm with decision trees)
- [[@measeBoostedClassificationTrees]] (discusses why probabilities from decision trees are problematic)
- [[@provostTreeInductionProbabilityBased]] (discusses why probabilities from decision trees are problematic)
- [[@zhuSemiSupervisedLearningLiterature]] (cited, selftraining is subset of semi-supervised learning)

## Notes 
- Authors show that standard decision trees as a base learnt can not be effective in a self-training algorithm (see [[@yarowskyUnsupervisedWordSense1995]]) due to unreliable proability estimates / confidence in prediction used in obtaining the pseudo labels. The problem also persists for ensembles (Random Subspace Method, Random Forest), even though it is less pronounced. As the probabilites are combined from the single trees, the problems also carry over to the ensemble. Poor probability estimates are esspecially problematic in self-training as the probabilty estimation is at core (see how pseudo labels are obtained) and as the number of labelled data is typically small.
- Authors suggest techniques to obtain improved probability estimates estimates. These include, naive bayes tree, a combination of no-pruning, laplace correction, grafting, and distance based measures.
- Authors show that self-training and the use of unlabeled data does **not** improve the classification performance of decision trees.
- They give the following reasons for the poor proability estimates: 
	1. sample size at leaves is always small as number of labelled data is limited
	2. all instances within the leaf are assigned the same probability. The probability estimate is the proportion of the majority class at the leaf of a (pruned) decision tree. A trained decision tree uses the absolute class frequencies of each leaf of the tree as: $p(k \mid x)=\frac{K}{N}$ where $K$ is the number of instances of the class $k$ out of $N$ instances at a leaf. **Example:** if 45 of 50 samples in one leaf node belong to one class, all samples get a probability of 0.9. If there are just 3 samples with the same class in one leaf node, all are assigned a probability of 1.0.
- Possible techniques include Laplacian correction (= smoothing probability values at leaf nodes), ommit decision tree pruning (= grown trees to their full depth), Naive Bayesian Tree learner (= construct a local Naive Bayes Classifier at each leaf node), grafted decision trees (=search speace of attributes that are labelled but contain no or very sparse training data). Techniques can be combined and are combined in the paper.
- While these improvements do not yield a higher accuracy on **labelled data** only, it can increase the performance of single trees and ensembles in self-training. That is, they observe no difference in accuracy between algorithms trained on labelled and labelled + unlabeled data. If additional techniques like Laplacian correction or No-pruning are emplied, probability estimates improve, which leads to higher-confidence predictions (see Table 3)
- Ensembles outperfrom single classfiers (see Table 6-8). Yet the results for ensembles also indicate that improving classification accuracy and the probability estimate of the base learner are effective for improving the performance of ensemble in self-training.
- Paper focuses self-training with decision tree learner as the base learner only.
- Self-training is a wrapper algorithm, that is hard to analyse. For some base classifiers theoretical bounds could be established (see citation example for Yarowsky algorithm)
- Instead of using probability estimates only to select unlabelled examples authors suggest to select samples using a combination of a distance-based approach and the probability estimation.
- Commonly probabilites for random forests are obtained by averaging class probability distributions estimated by the relative class frequency, the Laplace estimate and the $m$-estimate. The standard is the relative frequency, which is illsuited for the use in self-training. -> Understand how it's done in e. g., [[@prokhorenkovaCatBoostUnbiasedBoosting2018]].
- Differences between supervised and self-training deminishes (Figure 4), if more labelled data is used to train the classifier.
- Random Forests improve the classification performance of self-training, when more labelled data is available. With more labelled data the random forest generates more diverse decision trees? -> Can one conclude labelled data is always prefered to self-training?
- They conlcude based on their experiments that the probability estimation of the tree classifier leads to a better selection metric for the self-training algorithm and in consequence to better predictions.

## Annotations

“We show that standard decision tree learning as the base learner cannot be effective in a self-training algorithm to semi-supervised learning. The main reason is that the basic decision tree learner does not produce reliable probability estimation to its predictions.” ([Tanha et al., 2017, p. 355](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=1&annotation=B5ANNEJD))

“Therefore, it cannot be a proper selection criterion in self-training.” ([Tanha et al., 2017, p. 355](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=1&annotation=R43TL48R))

“tree learner that produce better probability estimation than using the distributions at the leaves of the tree. We show that these modifications do not produce better performance when used on the labelled data only, but they do benefit more from the unlabeled data in self-training” ([Tanha et al., 2017, p. 355](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=1&annotation=XTIA54J4))

“The modifications that we consider are Naive Bayes Tree, a combination of No-pruning and Laplace correction, grafting, and using a distance-based measure.” ([Tanha et al., 2017, p. 355](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=1&annotation=KGC9UXWZ))

“Semi-supervised learning algorithms use not only the labelled data but also unlabeled data to construct a classifier. The goal of semi-supervised learning is to use unlabeled instances and combine the information in the unlabeled data with the explicit classification information of labelled data for improving the classification performance.” ([Tanha et al., 2017, p. 355](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=1&annotation=ZEP66B7E))

“A self-training algorithm is an iterative method for semi-supervised learning, which wraps around a base learner. It uses its own predictions to assign labels to unlabeled data. Then, a set of newly-labelled data, which we call a set of high-confidence predictions, are selected to be added to the training set for the next iterations. The performance of the self-training algorithm strongly depends on the selected newly-labelled data at each iteration of the training procedure.” ([Tanha et al., 2017, p. 355](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=1&annotation=RWPA9AZU))

“In this paper we focus on self-training with a decision tree learner as the base learner.” ([Tanha et al., 2017, p. 356](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=2&annotation=KNTQ8FNY))

“We will show that using this as the selection metric in self-training does not improve the classification performance of a self-training algorithm and thus the algorithm does not benefit from the unlabeled data.” ([Tanha et al., 2017, p. 356](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=2&annotation=ZLSXP3IZ))

“These include: (1) the sample size at the leaves is almost always small, there is a limited number of labelled data, and (2) all instances at a leaf get the same probability.” ([Tanha et al., 2017, p. 356](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=2&annotation=MHE2ARYC))

“Our hypothesis is that these modified decision tree learners will show classification accuracy similar to the standard decision tree learner when applied to the labelled data only, but will benefit from the unlabeled data when used as the base classifier in self-training, because they make better probability estimates which is vital for the selection step of selftraining.” ([Tanha et al., 2017, p. 356](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=2&annotation=C77IR265))

“We then extend our analysis from single decision trees to ensembles of decision trees, in particular the Random Subspace Method [19] and Random Forests [9].” ([Tanha et al., 2017, p. 356](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=2&annotation=GPGRFST6))

“In this case, probability is estimated by combining the predictions of multiple trees. However, if the trees in the ensemble suffer from poor probability estimation, the ensemble learner will not benefit much from self-training on unlabeled data. Using the modified decision tree learners as the base learner for the ensemble will improve the performance of self-training with the ensemble classifier as the base learner.” ([Tanha et al., 2017, p. 356](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=2&annotation=6FI7ETTD))

“In general, self-training is a wrapper algorithm, and is hard to analyse. However, for specific base classifiers, theoretical analysis is feasible, for example [17] showed that the Yarowsky algorithm [45] minimises an upperbound on a new definition of cross entropy based on a specific instantiation of the Bregman distance.” ([Tanha et al., 2017, p. 357](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=3&annotation=6F9SHNSB))

“The goal of the selection step in Algorithm 1 is to find a set unlabeled examples with high-confidence predictions, above a threshold T. This is important, because selection of incorrect predictions will propagate to produce further classification errors.” ([Tanha et al., 2017, p. 357](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=3&annotation=JX35W6SC))

“Although for many domains decision tree classifiers produce good classifiers, they provide poor probability estimates [28, 31].” ([Tanha et al., 2017, p. 357](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=3&annotation=UW9QVF9X))

“The reason is that the sample size at the leaves is almost always small, and all instances at a leaf get the same probability. The probability estimate is simply the proportion of the majority class at the leaf of a (pruned) decision tree. A trained decision tree indeed uses the absolute class frequencies of each leaf of the tree as follows: Int. J. Mach. Learn. & Cyber. (2017) 8:355–370 357 12” ([Tanha et al., 2017, p. 357](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=3&annotation=AFNQB872))

“pðkjxÞ¼K N ð1Þ where K is the number of instances of the class k out of N instances at a leaf. However, these probabilities are based on very few data points, due to the fragmentation of data over the decision tree. For example, if a leaf node has subset of 50 examples of which 45 examples belong to one class, then any example that corresponds to this leaf will get 0:9 probability where a leaf with 3 examples of one class get a probability of 1:0. In semi-supervised learning this problem is even more serious, because in applications the size of initial set of labelled data is typically quite small. Here, we consider several methods for improving the probability estimates at the leaves of decision trees.” ([Tanha et al., 2017, p. 358](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=4&annotation=FFEP4JAL))

“One candidate improvement is the Laplacian correction (or Laplace estimator) which smooths the probability values at the leaves of the decision tree [31].” ([Tanha et al., 2017, p. 358](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=4&annotation=NVPRFPQT))

“Another possible improvement is a decision tree learner that does not do any pruning. Although this introduces the risk of ‘‘overfitting’’, it may be a useful method because of the small amount of training data.” ([Tanha et al., 2017, p. 358](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=4&annotation=GCPMJTG3))

“In an NBTree, a local Naive Bayes Classifier is constructed at each leaf of decision tree that is built by a standard decision tree learning algorithm like C4.5.” ([Tanha et al., 2017, p. 358](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=4&annotation=5MVE2YLN))

“The idea behind grafting is that some regions in the data space are more sparsely populated.” ([Tanha et al., 2017, p. 358](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=4&annotation=EIVC3LWF))

“Another way to select from the unlabeled examples is to use a combination of distance-based approach and the probability estimation.” ([Tanha et al., 2017, p. 359](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=5&annotation=F5E4HGW5))

“In an ensemble classifier, probability estimation is estimated by combining the confidences of their components. This tends to improve both the classification accuracy and the probability estimation. However, if a standard decision tree learner is used as the base learner, then the problems, that we noted above, carry over to the ensemble. We therefore expect that improving the probability estimates of the base learner will enable the ensemble learner to benefit more from the unlabeled data than if the standard decision tree learner is used.” ([Tanha et al., 2017, p. 360](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=6&annotation=VNIQ358E))

“There are several ways to compute the probability estimation in a random forest such as averaging class probability distributions estimated by the relative class frequency, the Laplace estimate and the m-estimate respectively. The standard random forest uses the relative class frequency as its probability estimation which is not suitable for self-training as we discussed” ([Tanha et al., 2017, p. 360](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=6&annotation=SISZH3WL))

“In Table 3, the columns DT and ST-DT show the classification accuracy of J48 base learner and self-training respectively. As can be seen, self-training does not benefit from unlabeled data and there is no difference in accuracy between learning from the labelled data only and selftraining from labelled and unlabeled data.” ([Tanha et al., 2017, p. 362](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=8&annotation=M6C8WNEW))

“As can be seen, unlike the basic decision tree learner, C4.4 enables self-training to become effective for nearly all the datasets. The average improvement over the used datasets is 1.9 %. The reason for improvement is that using Laplacian correction and No-pruning give better rank for probability estimation of the decision tree, which leads to select a set of high-confidence predictions.” ([Tanha et al., 2017, p. 362](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=8&annotation=8XH4PHEE))

“The results show that RSM with C4.4graft as the base learner in self-training improves the classification performance of RSM on 13 out of 14 datasets and the average improvement over all datasets is 2.7 %. The same results are shown in Table 7 for RSM, when the NBTree is the base learner.” ([Tanha et al., 2017, p. 364](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=10&annotation=SZ46BJYQ))

“Finally, comparing Tables 6–8, shows that ensemble methods outperform the single classifier especially for web-page datasets. The results also verify that improving both the classification accuracy and the probability estimates of the base learner in self-training are effective for improving the performance.” ([Tanha et al., 2017, p. 364](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=10&annotation=KJU3Y5B6))

“Consistent with our hypothesis we observe that difference between supervised algorithms and self-training methods decreases when the number of labelled data increases. Another interesting observation is that RF improves the classification performance of self-training when more labelled data are available, because with more labelled data the bagging approach, used in RF, generates diverse decision trees.” ([Tanha et al., 2017, p. 365](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=11&annotation=2RKGDGA5))

“The main contribution of this paper is the observation that when a learning algorithm is used as the base learner in self-training, it is very important that the confidence of prediction is correctly estimated, probability estimation.” ([Tanha et al., 2017, p. 368](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=14&annotation=VT6SCXCX))

“The standard technique of using the distribution at the leaves of decision tree as probability estimation does not enable self-training with a decision tree learner to benefit from unlabeled data. The accuracy is the same as when the decision tree learner is applied to only the labelled data.” ([Tanha et al., 2017, p. 368](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=14&annotation=NQDXXEZJ))

“Although to a lesser extent, the same is true when the modified decision tree learners are used as the base learner in an ensemble learner.” ([Tanha et al., 2017, p. 368](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=14&annotation=DZT76FK9))

“Based on the results of the experiments we conclude that improving the probability estimation of the tree classifiers leads to better selection metric for the self-training algorithm and produces better classification model.” ([Tanha et al., 2017, p. 368](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=14&annotation=S76ZEL6S))

“We observe that using Laplacian correction, No-pruning, grafting, and NBTree produce better probability estimation in tree classifiers.” ([Tanha et al., 2017, p. 368](zotero://select/library/items/6SLU5KR5)) ([pdf](zotero://open-pdf/library/items/XAIIBM8Q?page=14&annotation=WIEINWI2))