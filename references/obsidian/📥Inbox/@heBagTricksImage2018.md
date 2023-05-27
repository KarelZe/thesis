*title:* Bag of Tricks for Image Classification with Convolutional Neural Networks
*authors:* Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
*year:* 2018
*tags:* #lr #lr-scheduling #sgd #lr-warmup
#mix-up #weight-decay
*status:* #📦 
*related:*
- [[@loshchilovSGDRStochasticGradient2017]] (cited in paper)
- [[@shavittRegularizationLearningNetworks2018]] (also aims for generalisation)

## Notes 📍
- Authors examine a collection refinements in training neural networks commonly used in image classification research and evaluate their impact on the final model accuracy.
- They use ResNet-50, Inception-V3 and MobileNet as their baseline. Thus study is not 100 % comparable, as
**Efficient trading:**
- larger batch sizes degrade the validation performance of the model compared to models with a smaller batch size. Authors discuss the following heuristics to improve accuracy while training on larger batches: 
	- **Linear scaling learning rate:** Use a coarse learning rate, as due to the larger batch the noise in the gradient of SGD is reduced, but the expectation is the same.
	- **Learning rate warmup:** Use a small learning rate at the beginning then switch back to initial learning rate once the training process has stabilised.
	- **Disable bias decay:** Apply no L2 regularisation on bias terms and the $\gamma$ and $\beta$ term in batch normalisation layers.
	- **Use FP 16:** Calculate all parameters with floating point (FP16) precision. Can have performance benefit, but might have a narrower range that makes results more likely be out-of-bounds or disturb the training process.

- **Training refinements:**
	- **Cosine learning rate decay:** Instead of (exponentially) decaying the learning rate after a warm-up phase, use a cosine annealing strategy as proposed in [[@loshchilovSGDRStochasticGradient2017]]. 
	- **Label smoothing:** Instead of sofmax in the last layer of the image classification network, use an alternative method to calculate the output scores that is less prone to overfitting. Label smoothing changes the construction of the true probability to
	$$
		q_i= \begin{cases}1-\varepsilon & \text { if } i=y \\ \varepsilon /(K-1) & \text { otherwise }\end{cases}
	$$
		where $\varepsilon$ is a small constant. Now the optimal solution becomes
		$$
		z_i^*= \begin{cases}\log ((K-1)(1-\varepsilon) / \varepsilon)+\alpha & \text { if } i=y \\ \alpha & \text { otherwise }\end{cases}
		$$
		where $\alpha$ can be an arbitrary real number. This encourages a finite output from the fully-connected layer and can generalise better.
	- **Knowledge distallation:**
		-  Use a teacher model to help train the current / the so-called student model. The teacher model is often pre-trained with higher accuracy. Through mimicing  the student model can improve its own accuracy, while the model complexity remains the same. A distillation loss is used.
	- **Mixup:**
		- In mixup, each time we randomly sample two examples $\left(x_i, y_i\right)$ and $\left(x_j, y_j\right)$. Then we form a new example by a weighted linear interpolation of these two examples:
$$
\begin{aligned}
&\hat{x}=\lambda x_i+(1-\lambda) x_j, \\
&\hat{y}=\lambda y_i+(1-\lambda) y_j,
\end{aligned}
$$
				where $\lambda \in[0,1]$ is a random number drawn from the $\operatorname{Beta}(\alpha, \alpha)$ distribution (distribution from 0 to 1). In mixup training, we only use the new example $(\hat{x}, \hat{y})$.
- **Conclusion:**
	- Authors apply several techniques that only require minor changes to the architecture such as learning rate scheduling. The techniques can be combined. Stacking improves the accruacy and has strong advantages in transfer learning.

## Annotations 📖

“Much of the recent progress made in image classification research can be credited to training procedure refinements, such as changes in data augmentations and optimisation methods. In the literature, however, most refinements are either briefly mentioned as implementation details or only visible in source code. In this paper, we will examine a collection of such refinements and empirically evaluate their impact on the final model accuracy through ablation study.” ([He et al., 2018, p. 1](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=1&annotation=TRSG78LA))

“Mini-batch SGD groups multiple samples to a minibatch to increase parallelism and decrease communication costs. Using large batch size, however, may slow down the training progress. For convex problems, convergence rate decreases as batch size increases” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=3SBPIGZM))

“Linear scaling learning rate. In mini-batch SGD, gradient descending is a random process because the examples are randomly selected in each batch. Increasing the batch size does not change the expectation of the stochastic gradient but reduces its variance. In other words, a large batch size reduces the noise in the gradient, so we may increase the learning rate to make a larger progress along the opposite of the gradient direction.” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=UED7NKN4))

“Learning rate warmup. At the beginning of the training, all parameters are typically random values and therefore far away from the final solution. Using a too large learning rate may result in numerical instability. In the warmup heuristic, we use a small learning rate at the beginning and then switch back to the initial learning rate when the training process is stable” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=VWCNM5T5))

“o bias decay. The weight decay is often applied to all learnable parameters including both weights and bias. It’s equivalent to applying an L2 regularisation to all parameters to drive their values towards 0” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=IXMM6NCU))

“Other parameters, including the biases and γ and β in BN layers, are left unregularized.” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=UWEP6UN3))

“Despite the performance benefit, a reduced precision has a narrower range that makes results more likely to be out-ofrange and then disturb the training progress.” ([He et al., 2018, p. 3](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=3&annotation=F4YWNXWL))

“Learning rate adjustment is crucial to the training. After the learning rate warmup described in Section 3.1, we typically steadily decrease the value from the initial learning rate. The widely used strategy is exponentially decaying the learning rate.” ([He et al., 2018, p. 5](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=5&annotation=SM7KDSAE))

“In contrast to it, Loshchilov et al.  propose a cosine annealing strategy. An simplified version is decreasing the learning rate from the initial value to 0 by following the cosine function.” ([He et al., 2018, p. 5](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=5&annotation=3KEXINRD))

“These scores can be normalised by the softmax operator to obtain predicted probabilities.” ([He et al., 2018, p. 6](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=6&annotation=642TRFFD))

“The idea of label smoothing was first proposed to train Inception-v2. It changes the construction of the true probability to qi = { 1 − ε if i = y, ε/(K − 1) otherwise, (4) where ε is a small constant. Now the optimal solution becomes z∗ i= { log((K − 1)(1 − ε)/ε) + α if i = y, α otherwise, (5) where α can be an arbitrary real number. This encourages a finite output from the fully-connected layer and can generalise better.” ([He et al., 2018, p. 6](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=6&annotation=T4FARF6A))

“In knowledge distillation , we use a teacher model to help train the current model, which is called the student model. The teacher model is often a pre-trained model with higher accuracy, so by imitation, the student model is able to improve its own accuracy while keeping the model complexity the same.” ([He et al., 2018, p. 6](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=6&annotation=V5EJ72GK))

“Here we consider another augmentation method called mixup. In mixup, each time we randomly sample two examples (xi, yi) and (xj, yj).” ([He et al., 2018, p. 7](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=7&annotation=Q5UDEXYG))

“In this paper, we survey a dozen tricks to train deep convolutional neural networks to improve model accuracy.” ([He et al., 2018, p. 8](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=8&annotation=JXHEIADN))

“More excitingly, stacking all of them together leads to a significantly higher accuracy. In addition, these improved pre-trained models sho” ([He et al., 2018, p. 8](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=8&annotation=ZZXA3Q8G))

“strong advantages in transfer learning, which improve both object detection and semantic segmentation. We believe the benefits can extend to broader domains where classification base models are favoured.” ([He et al., 2018, p. 9](zotero://select/library/items/F7UZXV8E)) ([pdf](zotero://open-pdf/library/items/WTS9X9U5?page=9&annotation=I7J6VUS4))