*title:* Cyclical Learning Rates for Training Neural Networks
*authors:* Leslie N. Smith
*year:* 2017
*tags:* #lr #lr-scheduling #cyclic #deep-learning #neural_network 
*status:* #📥
*related:*
- [[@loshchilovSGDRStochasticGradient2017]]
- [[@huangSnapshotEnsemblesTrain2017]]
*code:*
- https://github.com/bckenstler/CLR

## Notes 
- Cyclical learning rates are an approach to set the learning rate of a neural net without the need to tune the learning rate in multiple experiments. Only setting a min and max bound is required, which can be obtained from a  single model run for a few epochs.
- The intuition behind CLR is, that an increase in lr might harm in the short term, but lead to better results in the long run due to an improved behaviour when optimising towards saddle points instead of poor local minima.
- The lr cycles between a minimum and a maximum lr rate and then resets after a certain step size. Hence, a triangular like training policy. (see picture below)
- Adaptive learning rates are similar but fundamentally different from cyclical learning rates. Both can be coupled.
- paper is predecessor to [[@loshchilovSGDRStochasticGradient2017]]
- Paper contains some practical advice to set all hyperparameters. Tests are performed in computer vision domain, however.

![[cyclical-lr.png]]
## Annotations

“It is well known that too small a learning rate will make a training algorithm converge slowly while too large a learning rate will make the training algorithm diverge (see Beningo)” ([Smith, 2017, p. 1](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=1&annotation=PBM87L43))

“In addition, this cyclical learning rate (CLR) method practically eliminates the need to tune the learning rate yet achieve near optimal classification accuracy.” ([Smith, 2017, p. 1](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=1&annotation=KL2GXG4R))

“A methodology for setting the global learning rates for training neural networks that eliminates the need to perform numerous experiments to find the best values and schedule with essentially no additional computation.” ([Smith, 2017, p. 1](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=1&annotation=H5X4R39W))

“allowing the learning rate to rise and fall is beneficial overall even though it might temporarily harm the network’s performance.” ([Smith, 2017, p. 2](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=2&annotation=7UWVSIXX))

“Adaptive learning rates are fundamentally different from CLR policies, and CLR can be combined with adaptive learning rates” ([Smith, 2017, p. 2](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=2&annotation=7G2N9X4U))

“The essence of this learning rate policy comes from the observation that increasing the learning rate might have a short term negative effect and yet achieve a longer term beneficial effect.” ([Smith, 2017, p. 2](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=2&annotation=QC3XJQR4))

“This led to adopting a triangular window (linearly increasing then linearly decreasing), which is illustrated in Figure 2, because it is the simplest function that incorporates this idea. The rest of this paper refers to this as the triangular learning rate policy” ([Smith, 2017, p. 2](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=2&annotation=YAMDENRP))

“An intuitive understanding of why CLR methods work comes from considering the loss function topology. Dauphin et al. argue that the difficulty in minimising the loss arises from saddle points rather than poor local minima” ([Smith, 2017, p. 2](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=2&annotation=4QL9SALK))

“Saddle points have small gradients that slow the learning process. However, increasing the learning rate allows more rapid traversal of saddle point plateaus. A more practical reason as to why CLR works is that, by following the methods in Section 3.3, it is likely the optimum learning rate will be between the bounds and near optimal learning rates will be used throughout training.” ([Smith, 2017, p. 3](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=3&annotation=J6QT3L6Z))

“The length of a cycle and the input parameter stepsize can be easily computed from the number of iterations in an epoch. An epoch is calculated by dividing the number of training images by the batchsize used.” ([Smith, 2017, p. 3](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=3&annotation=U8X5W6NJ))

“There is a simple way to estimate reasonable minimum and maximum boundary values with one training run of the network for a few epochs. It is a “LR range test”; run your model for several epochs while letting the learning rate increase linearly between low and high LR values. This test is enormously valuable whenever you are facing a new architecture or dataset.” ([Smith, 2017, p. 3](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=3&annotation=X9T6NG6M))

“The triangular learning rate policy provides a simple mechanism to do this. For example, in Caffe, set base lr to the minimum value and set max lr to the maximum value. Set both the stepsize and max iter to the same number of iterations. In this case, the learning rate will increase linearly from the minimum value to the maximum value during this short run. Next, plot the accuracy versus learning rate.” ([Smith, 2017, p. 3](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=3&annotation=ELMCTLQ3))

“A short run of only a few epochs where the learning rate linearly increases is sufficient to estimate boundary learning rates for the CLR policies. Then a policy where the learning rate cyclically varies between these bounds is sufficient to obtain near optimal classification results, often with fewer iterations.” ([Smith, 2017, p. 8](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=8&annotation=RG8SBZ8A))

“This policy is easy to implement and unlike adaptive learning rate methods, incurs essentially no additional computational expense.” ([Smith, 2017, p. 8](zotero://select/library/items/7HMX8QTU)) ([pdf](zotero://open-pdf/library/items/KYLZPI9D?page=8&annotation=LMJQ5BP8))