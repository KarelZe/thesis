*title:* Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks
*authors:* Dong-Hyun Lee
*year:* 2013
*tags:* #semi-supervised #pseudo-labelling #deep-learning #supervised-learning 
*status:* #📦 
*related:*
- [[@chapelleSemiSupervisedClassificationLow2005]]
- [[@zhuSemiSupervisedLearningLiterature]]
- [[@devlinBERTPretrainingDeep2019]]
- [[@clarkELECTRAPretrainingText2020]]

## Notes 📍
- Authors propose a simple way of training neural networks in a [[semi-supervised]] fashion. The network is trained with labeled and unlabeled data simultaneously instead of using separated pre-training and finetuning. In absence of true labels, for unlabeled data pseudo labels are assigned from the class with the highest class probablity ($y_i^{\prime}= \begin{cases}1 & \text { if } i=\operatorname{argmax} \min _{i^{\prime}} f_{i^{\prime}}(x) \\ 0 & \text { otherwise }\end{cases}$) and re-calculated every weight update. Therefore pseudo-labels are treated similar to manual labels.
- Due to an imbalance of unlabelled and labelled data in the training set, it's important to adjust the loss function given by:
$$
L=\frac{1}{n} \sum_{m=1}^n \sum_{i=1}^C L\left(y_i^m, f_i^m\right)+\alpha(t) \frac{1}{n^{\prime}} \sum_{m=1}^{n^{\prime}} \sum_{i=1}^C L\left(y_i^{\prime m}, f_i^{\prime m}\right),
$$
	where $n$ is the number of mini-batch in labeled data for SGD, $n^{\prime}$ for unlabeled data, $f_i^m$ is the output units of $m$ 's sample in labeled data, $y_i^m$ is the label of that, $f_i^{\prime m}$ for unlabeled data, $y_i^{\prime m}$ is the pseudo-label of that for unlabeled data, $\alpha(t)$ is a coefficient balancing them.
	Setting a good $\alpha(t)$ is important. If $\alpha(t)$ is too high, it hinders training even for labeled data. If $\alpha(t)$ is too small one won't benefit from unlabeled data.
- Can be coupled with dropout [[@hintonImprovingNeuralNetworks2012]] and denoising AE.
- Pseudo labeling omits the caculation of an expensive similarity matrix and achieves state-of-the-art-performance.
- Pseudo labelling requires a change in the loss and network architecture. Also, setting the weighting scheme $\alpha_{t}$ is very brittle.  So even though it's conceptually simple it would impact a comparsion between models. Better use pre-training but be aware of wastefulness of e. g., BERT.

## Annotations 📖

“Basically, the proposed network is trained in a supervised fashion with labeled and unlabeled data simultaneously. For unlabeled data, Pseudo-Label s, just picking up the class which has the maximum predicted probability, are used as if they were true labels.” ([Lee, p. 2](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=2&annotation=B9LRSNJP))

“This is in effect equivalent to Entropy Regularization.” ([Lee, p. 2](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=2&annotation=A3IA8ZG6))

“In a first phase, unsupervised pre-training, the weights of all layers are initialized by this layer-wise unsupervised training. In a second phase, fine-tuning, the weights are trained globally with labels using backpropagation algorithm in a supervised fashion. All of these methods also work in a semi-supervised fashion. We have only to use extra unlabeled data for unsupervised pretraining” ([Lee, p. 2](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=2&annotation=98LJ6VTE))

“In this article we propose the simpler way of training neural network in a semi-supervised fashion. Basically, the proposed network is trained in a supervised fashion with labeled and unlabeled data simultaneously. For unlabeled data, Pseudo-Label s, just picking up the class which has the maximum predicted probability every weights update, are used as if they were true labels.” ([Lee, p. 2](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=2&annotation=QY5579XT))

“In principle, this method can combine almost all neural network models and training methods. In our experiments, Denoising Auto-Encoder (Vincent et al., 2008) and Dropout (Hinton et al., 2012) boost up the performance” ([Lee, p. 2](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=2&annotation=78IY4EGR))

“This method is in effect equivalent to Entropy Regularization (Grandvalet et al., 2006). The conditional entropy of the class probabilities can be used for a measure of class overlap. By minimizing the entropy for unlabeled data, the overlap of class probability distribution can be reduced.” ([Lee, p. 2](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=2&annotation=KURE3RBX))

“Because in each weights update we train a different sub-model by omitting a half of hidden units, this training procedure is similar to bagging (Breiman, 1996), where many different networks are trained on different subsets of th” ([Lee, p. 3](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=3&annotation=SQKDR79X))

“Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks data. But dropout is different from bagging in that all of the sub-models share same weights.” ([Lee, p. 4](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=4&annotation=2XJ4YAXU))

“Pseudo-Label are target classes for unlabeled data as if they were true labels. We just pick up the class which has maximum predicted probability for each unlabeled sample y′ i= { 1 if i = argmaxi′ fi′ (x) 0 otherwise We use Pseudo-Label in a fine-tuning phase with Dropout. The pre-trained network is trained in a supervised fashion with labeled and unlabeled data simultaneously. For unlabeled data, Pseudo-Label s recalculated every weights update are used for the same loss function of supervised learning task” ([Lee, p. 4](zotero://select/library/items/7QE4ZTFQ))

“Because the total number of labeled data and unlabeled data is quite different and the training balance between them is quite important for the network performance, the overall loss function is L= 1 n n ∑ m=1 C ∑ i=1 L(ym i ,fm i )+α(t) 1 n′ n′ ∑ m=1 C ∑ i=1 L(y′m i , f ′m i ), (15) where n is the number of mini-batch in labeled data for SGD, n′ for unlabeled data, f m i is the output units of m’s sample in labeled data, ym i is the label of that, f ′m i for unlabeled data, y′m i is the pseudo-label of that for unlabeled data, α(t) is a coefficient balancing them. The proper scheduling of α(t) is very important for the network performance. If α(t) is too high, it disturbs training even for labeled data. Whereas if α(t) is too small, we cannot use benefit from unlabeled data. Furthermore, the deterministic annealing process, by which α(t) is slowly increased, is expected to help the optimization process to avoid poor local minima (Grandvalet et al., 2006) so that the pseudo-labels of unlabeled data are similar to true labels as much as possible.” ([Lee, p. 4](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=4&annotation=H5CRAJLQ))

“The goal of semi-supervised learning is to improve generalization performance using unlabeled data. The cluster assumption states that the decision boundary should lie in low-density regions to improve generalization performance (Chapelle et al., 2005).” ([Lee, p. 4](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=4&annotation=PSMYFXRX))

“Figure 1 shows t-SNE (Van der Maaten et al., 2008) 2D embedding results of the network output of MNIST test data (not included in unlabeled data). The neural network was trained with 600 labeled data and with or without 60000 unlabeled data and Pseudo-Labels. Though the train error is zero in the two cases, the network outputs of test data is more condensed near 1-ofK code by training with unlabeled data and PseudoLabels, in other words, the entropy of (17) is minimized” ([Lee, p. 5](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=5&annotation=4UDET5W5))

“Without complex training scheme and computationally expensive similarity matrix, the proposed method shows the state-of-the-art performance.” ([Lee, p. 6](zotero://select/library/items/7QE4ZTFQ)) ([pdf](zotero://open-pdf/library/items/MTK4LZKA?page=6&annotation=45S3NKJI))