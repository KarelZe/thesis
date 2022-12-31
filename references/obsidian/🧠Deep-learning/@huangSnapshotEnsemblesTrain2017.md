*title:* Snapshot ensembles: train 1, get $M$ for free
*authors:* Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, Kilian Q. Weinberger
*year:* 2016
*tags:* #snapshot #ensembles #deep-learning #cylic-learning-rate #lr 
*status:* #📦 
*related:*
- [[@izmailovAveragingWeightsLeads2019]]
- [[@loshchilovSGDRStochasticGradient2017]] 
# Notes 
- Snapshot ensembles make heavy use of cylic learning rates (see [[@loshchilovSGDRStochasticGradient2017]]  and [[@smithCyclicalLearningRates2017]]), and take a snapshot every time a local optima is reached.
- At inference all snapshots are averaged.
- Works if all models do not classify the same samples wrongly and if all ensemble members have a low test error.
- It takes roughly the same time to train an ensemble as it would take to train a single neural network.
- Paper contains good references to other techniques such as #dropout

![[visualization-snapshot-ensembles 1.png]]

# Annotations

“Ensembles of neural networks are known to be much more robust and accurate than individual networks. However, training multiple deep networks for model averaging is computationally expensive. In this paper, we propose a method to obtain the seemingly contradictory goal of ensembling multiple neural networks at no additional training cost. We achieve this goal by training a single neural network, converging to several local minima along its optimization path and saving the model parameters. To obtain repeated rapid convergence, we leverage recent work on cyclic learning rate schedules” ([Huang et al., 2017, p. 1](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=1&annotation=GDV9NFBZ))

“Although deep networks typically never converge to a global minimum, there is a notion of “good” and “bad” local minima with respect to generalization. Keskar et al. (2016) argue that local minima with flat basins tend to generalize better. SGD tends to avoid sharper local minima because gradients are computed from small mini-batches and are therefore inexact (Keskar et al., 2016). If the learningrate is sufficiently large, the intrinsic random motion across gradient steps prevents the optimizer from reaching any of the sharp basins along its optimization path. However, if the learning rate is small, the model tends to converge into the closest local minimum.” ([Huang et al., 2017, p. 1](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=1&annotation=5LS58JYU))

“Although different local minima often have very similar error rates, the corresponding neural networks tend to make different mistakes.” ([Huang et al., 2017, p. 1](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=1&annotation=7EQW2RIB))

“diversity can be exploited through ensembling, in which multiple neural networks are trained from different initializations and then combined with majority voting or averaging (Caruana et al., 2004)” ([Huang et al., 2017, p. 2](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=2&annotation=6QT8F7PJ))

“Our approach leverages the non-convex nature of neural networks and the ability of SGD to converge to and escape from local minima on demand. Instead of training M neural networks independently from scratch, we let SGD converge M times to local minima along its optimization path. Each time the model converges, we save the weights and add the corresponding network to our ensemble. We then restart the optimization with a large learning rate to escape the current local minimum. More specifically, we adopt the cycling procedure suggested by Loshchilov & Hutter (2016), in which the learning rate is abruptly raised and then quickly lowered to follow a cosine function. Because our final ensemble consists of snapshots of the optimization path, we refer to our approach as Snapshot Ensembling.” ([Huang et al., 2017, p. 2](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=2&annotation=MJ29RQZ3))

“During testing time, one can evaluate and average the last (and therefore most accurate) m out of M models.” ([Huang et al., 2017, p. 2](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=2&annotation=TNP9CKM2))

“The Dropout (Srivastava et al., 2014) technique creates an ensemble out of a single model by “dropping” — or zeroing — random sets of hidden nodes during each mini-batch. At test time, no nodes are dropped, and each node is scaled by the probability of surviving during training. Srivastava et al. claim that Dropout reduces overfitting by preventing the co-adaptation of nodes. An alternative explanation is that this mechanism creates an exponential number of networks with shared weights during training, which are then implicitly ensembled at test time.” ([Huang et al., 2017, p. 3](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=3&annotation=5KPNFFXH))

“At the heart of Snapshot Ensembling is an optimization process which visits several local minima before converging to a final solution. We take model snapshots at these various minima, and average their predictions at test time.” ([Huang et al., 2017, p. 3](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=3&annotation=AI6ZNRW8))

“Ensembles work best if the individual models (1) have low test error and (2) do not overlap in the set of examples they misclassify. Along most of the optimization path, the weight assignments of a neural network tend not to correspond to low test error. In fact, it is commonly observed that the validation error drops significantly only after the learning rate has been reduced, which is typically done after several hundred epochs.” ([Huang et al., 2017, p. 4](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=4&annotation=QJ5PKM44))

“We lower the learning rate at a very fast pace, encouraging the model to converge towards its first local minimum after as few as 50 epochs. The optimization is then continued at a larger learning rate, which perturbs the model and dislodges it from the minimum. We repeat this process several times to obtain multiple convergences. Formally, the learning rate α has the form: α(t) = f (mod (t − 1, dT /M e)) , (1) where t is the iteration number, T is the total number of training iterations, and f is a monotonically decreasing function. In other words, we split the training process into M cycles, each of which starts with a large learning rate, which is annealed to a smaller learning rate.” ([Huang et al., 2017, p. 4](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=4&annotation=TW9QLJ6J))

“At the end of each training cycle, it is apparent that the model reaches a local minimum with respect to the training loss. Thus, before raising the learning rate, we take a “snapshot” of the model weights (indicated as vertical dashed black lines). After training M cycles, we have M model snapshots, f1 . . . fM , each of which will be used in the final ensemble.” ([Huang et al., 2017, p. 4](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=4&annotation=U7UZM43T))

“The ensemble prediction at test time is the average of the last m (m ≤ M ) model’s softmax outputs. Let x be a test sample and let hi (x) be the softmax score of snapshot i. The output of the ensemble is a simple average of the last m models: hEnsemble = 1 m ∑m−1 0 hM−i (x) . We always ensemble the last m models, as these models tend to have the lowest test error.” ([Huang et al., 2017, p. 4](zotero://select/library/items/876NJ97B)) ([pdf](zotero://open-pdf/library/items/XPH6QHRL?page=4&annotation=KYGLHHHX))