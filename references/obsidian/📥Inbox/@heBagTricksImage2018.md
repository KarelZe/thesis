*title:* Bag of Tricks for Image Classification with Convolutional Neural Networks
*authors:* Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
*year:* 2018
*tags:* #lr #lr-scheduling #sgd #lr-warmup
#mix-up #weight-decay
*status:* #📥
*related:*

## Notes 📍

## Annotations 📖

“Much of the recent progress made in image classification research can be credited to training procedure refinements, such as changes in data augmentations and optimization methods. In the literature, however, most refinements are either briefly mentioned as implementation details or only visible in source code. In this paper, we will examine a collection of such refinements and empirically evaluate their impact on the final model accuracy through ablation study.” ([He et al., 2018, p. 1](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=1&annotation=TRSG78LA))

“Mini-batch SGD groups multiple samples to a minibatch to increase parallelism and decrease communication costs. Using large batch size, however, may slow down the training progress. For convex problems, convergence rate decreases as batch size increases” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=3SBPIGZM))

“Linear scaling learning rate. In mini-batch SGD, gradient descending is a random process because the examples are randomly selected in each batch. Increasing the batch size does not change the expectation of the stochastic gradient but reduces its variance. In other words, a large batch size reduces the noise in the gradient, so we may increase the learning rate to make a larger progress along the opposite of the gradient direction.” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=UED7NKN4))

“Learning rate warmup. At the beginning of the training, all parameters are typically random values and therefore far away from the final solution. Using a too large learning rate may result in numerical instability. In the warmup heuristic, we use a small learning rate at the beginning and then switch back to the initial learning rate when the training process is stable” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=VWCNM5T5))

“o bias decay. The weight decay is often applied to all learnable parameters including both weights and bias. It’s equivalent to applying an L2 regularization to all parameters to drive their values towards 0” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=IXMM6NCU))

“Other parameters, including the biases and γ and β in BN layers, are left unregularized.” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=UWEP6UN3))

“Despite the performance benefit, a reduced precision has a narrower range that makes results more likely to be out-ofrange and then disturb the training progress.” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=F4YWNXWL))

“Learning rate adjustment is crucial to the training. After the learning rate warmup described in Section 3.1, we typically steadily decrease the value from the initial learning rate. The widely used strategy is exponentially decaying the learning rate.” ([He et al., 2018, p. 5](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=5&annotation=SM7KDSAE))

“In contrast to it, Loshchilov et al.  propose a cosine annealing strategy. An simplified version is decreasing the learning rate from the initial value to 0 by following the cosine function.” ([He et al., 2018, p. 5](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=5&annotation=3KEXINRD))

“These scores can be normalized by the softmax operator to obtain predicted probabilities.” ([He et al., 2018, p. 6](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=6&annotation=642TRFFD))

“The idea of label smoothing was first proposed to train Inception-v2. It changes the construction of the true probability to qi = { 1 − ε if i = y, ε/(K − 1) otherwise, (4) where ε is a small constant. Now the optimal solution becomes z∗ i= { log((K − 1)(1 − ε)/ε) + α if i = y, α otherwise, (5) where α can be an arbitrary real number. This encourages a finite output from the fully-connected layer and can generalize better.” ([He et al., 2018, p. 6](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=6&annotation=T4FARF6A))

“In knowledge distillation , we use a teacher model to help train the current model, which is called the student model. The teacher model is often a pre-trained model with higher accuracy, so by imitation, the student model is able to improve its own accuracy while keeping the model complexity the same.” ([He et al., 2018, p. 6](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=6&annotation=V5EJ72GK))

“Here we consider another augmentation method called mixup. In mixup, each time we randomly sample two examples (xi, yi) and (xj, yj).” ([He et al., 2018, p. 7](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=7&annotation=Q5UDEXYG))

“In this paper, we survey a dozen tricks to train deep convolutional neural networks to improve model accuracy.” ([He et al., 2018, p. 8](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=8&annotation=JXHEIADN))

“More excitingly, stacking all of them together leads to a significantly higher accuracy. In addition, these improved pre-trained models sho” ([He et al., 2018, p. 8](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=8&annotation=ZZXA3Q8G))

“strong advantages in transfer learning, which improve both object detection and semantic segmentation. We believe the benefits can extend to broader domains where classification base models are favored.” ([He et al., 2018, p. 9](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=9&annotation=I7J6VUS4))