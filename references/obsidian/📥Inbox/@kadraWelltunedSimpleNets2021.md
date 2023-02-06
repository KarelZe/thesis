*title:* Well-Tuned Simple Nets Excel on Tabular Datasets
*authors:* Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka
*year:* 2020
*tags:* #tabular #deep-learning #neural_network #lr-scheduling #dropout #sgd #adam #batch-normalization #gbm  #cyclic 
*status:* #📥
*related:*
- [[@borisovDeepNeuralNetworks2022]]
- [[@prokhorenkovaCatBoostUnbiasedBoosting2018]] (used as benchmark)
- [[@arikTabNetAttentiveInterpretable2020]] (used as benchmark)
- [[@hintonImprovingNeuralNetworks2012]] (used for regularization)
- [[@loshchilovSGDRStochasticGradient2017]] (used for regularization)
*code:* [https://github.com/releaunifreiburg/WellTunedSimpleNets](https://github.com/releaunifreiburg/WellTunedSimpleNets)
*review:* https://openreview.net/forum?id=2d34y5bRWxB

# Notes Sebastian Raschka
-   Using a combination of modern regularization techniques, the authors find that simple multi-layer perceptron (MLP) can outperform both specialized neural network architectures (TabNet) and gradient boosting machines (XGBoost and Catboost)
-   The MLP base architecture consisted of 9-layers with 512 units each (excluding the output layer) and was tuned with a cosine annealing scheduler.
-   The following 13 regularization techniques were considered in this study. Implicit: (1) BatchNorm, (2) stochastic weight averaging, (3) Look-ahead optimizer. (4) Weight decay. Ensemble: (5) Dropout, (6) snapshot ensembles. Structural: (7) skip connections, (8) Shake-Drop, (9) Shake-Shake. Augmentation: (10) Mix-Up, (11) Cut-Mix, (12) Cut-Out, (13) FGSM adversarial learning
-   The comparison spans 40 tabular datasets for classification, ranging from 452 to 416,188 examples. In 19 out of the 40 datasets, an MLP with a mix regularization techniques outperformed any other method evaluated in this study.
-   Caveats: the authors included NODE in the comparison but used did not tune its hyperparameters. Also, the regularization techniques were not applied to the other neural network architectures (TabNet and NODE).

## Notes
- Authors study different regularization cocktails 🍸 and can thereby outperform specialized neural nets for tabular data and gradient-boosted trees.
- Authors speculate that specialized architectures could also profit from regularization. 
- Authors are transparent about how the results were generated e. g., scaling, train-test-split etc.
- **caveats:** While regularization achieves remarkable achievements with regard to outperformance of the gradient-boosted trees, no TabTransformer has been covered. To TabNet no regularization techniques were applied due to the effort of implementation.
- Paper doesn't look into imbalanced data sets or semi-supervised learning. -> Could used to relativize results
- Paper was rejected from ICLR. see https://openreview.net/forum?id=2d34y5bRWxB for reasons.

## Annotations

“we hypothesize that the key to boosting the performance of neural networks lies in rethinking the joint and simultaneous application of a large set of modern regularization techniques” ([Kadra et al., 2021, p. 1](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=1&annotation=756AGEE8))

“The extensive experiments on 40 datasets we report indeed confirm that recent neural networks do not outperform GBDT when the hyperparameters of all methods are thoroughly tuned” ([Kadra et al., 2021, p. 1](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=1&annotation=IKY7UMAC))

“lies in exploiting the recent DL advances on regularization techniques (reviewed in Section 3), such as data augmentation, decoupled weight decay, residual blocks and model averaging (e.g., dropout or snapshot ensembles), or on learning dynamics (e.g., look-ahead optimizer or stochastic weight averaging)” ([Kadra et al., 2021, p. 1](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=1&annotation=EQBJUBAS))

“In fact, the performance improvements are quite pronounced and highly significant.1” ([Kadra et al., 2021, p. 2](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=2&annotation=NIQMLKXM))

“In contrast, we do not propose a new kind of neural architecture, but a novel paradigm for learning a combination of regularization methods” ([Kadra et al., 2021, p. 2](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=2&annotation=A6MIUAI5))

“Weight decay: The most classical approaches of regularization focused on minimizing the norms of the parameter values, e.g., either the L1  the L2 ], or a combination of L1 and L2 known as the Elastic Net [63].” ([Kadra et al., 2021, p. 2](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=2&annotation=FRLZSB3J))

“Data Augmentation: Among the augmentation regularizers, Cut-Out proposes to mask a subset of input features (e.g., pixel patches for images) for ensuring that the predictions remain invariant to distortions in the input space” ([Kadra et al., 2021, p. 3](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=3&annotation=A7IG7QQT))

“Ensemble methods: Ensembled machine learning models have been shown to reduce variance and act as regularizers” ([Kadra et al., 2021, p. 3](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=3&annotation=CKVE9MIM))

“Structural and Linearization: In terms of structural regularization, ResNet adds skip connections across layers ., while the Inception model computes latent representations by aggregating diverse convolutional filter sizes.” ([Kadra et al., 2021, p. 3](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=3&annotation=VVTR34CT))

“Implicit: The last family of regularizers broadly encapsulates methods that do not directly propose novel regularization techniques but have an implicit regularization effect as a virtue of their ‘modus operandi’ .. The simplest such implicit regularization is Early Stopping ., which limits overfitting by tracking validation performance over time and stopping training when validation performance no longer improves. Another implicit regularization method is Batch Normalization, which improves generalization by reducing internal covariate shift. The scaled exponential linear units (SELU) represent an alternative to batch-normalization through self-normalizing activation functions. On the other hand, stabilizing the convergence of the training routine is another implicit regularization, for instance by introducing learning rate scheduling schemes. The recent strategy of stochastic weight averaging relies on averaging parameter values from the local optima encountered along the sequence of optimization steps, while another approach conducts updates in the direction of a few ‘lookahead’ steps.” ([Kadra et al., 2021, p. 3](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=3&annotation=C3QLF75B))

“While we can in principle use any hyperparameter optimization method, we decided to use the multi-fidelity Bayesian optimization method BOHB  since it achieves strong performance across a wide range of computing budgets by combining Hyperband and Bayesian Optimization, and since BOHB can deal with the categorical hyperparameters we use for enabling or disabling regularization techniques and the corresponding conditional structures.” ([Kadra et al., 2021, p. 4](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=4&annotation=5ZMMQIHH))

“The datasets are retrieved from the OpenML repository using the OpenML-Python connector and split as 60% training, 20% validation, and 20% testing sets. The data is standardized to have zero mean and unit variance where the statistics for the standardization are calculated on the training split.” ([Kadra et al., 2021, p. 5](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=5&annotation=ULLVLKGX))

“Taking into account the dimensions D of the considered configuration spaces, we ran BOHB for at most 4 days, or at most 40 × D hyperparameter configurations, whichever came first. During the training phase, each configuration was run for 105 epochs, in accordance with the cosine learning rate annealing with restarts (described in the following subsection).” ([Kadra et al., 2021, p. 5](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=5&annotation=Z4BGUMBV))

“For the sake of studying the effect on more datasets, we only evaluated a single train-val-test split. After the training phase is completed, we report the results of the best hyperparameter configuration found, retrained on the joint train and validation set.” ([Kadra et al., 2021, p. 5](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=5&annotation=MG4ETHAP))

“We use a 9-layer feed-forward neural network with 512 units for each layer, a choice motivated by previous work” ([Kadra et al., 2021, p. 6](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=6&annotation=QLDTQ9PI))

“Moreover, we set a low learning rate of 10−3 after performing a grid search for finding the best value across datasets. We use AdamW, which implements decoupled weight decay, and cosine annealing with restarts as a learning rate scheduler. Using a learning rate scheduler with restarts helps in our case because we keep a fixed initial learning rate.” ([Kadra et al., 2021, p. 6](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=6&annotation=IY4J5SP3))

“For the restarts, we use an initial budget of 15 epochs, with a budget multiplier of 2, following published practices” ([Kadra et al., 2021, p. 6](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=6&annotation=NFA2C4PC))

“We use the Critical Difference (CD) diagram of the ranks based on the Wilcoxon significance test, a standard metric for comparing classifiers across multiple datasets” ([Kadra et al., 2021, p. 8](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=8&annotation=D8CQBND9))

“well-regularized simple deep MLPs outperform specialized neural architectures.” ([Kadra et al., 2021, p. 8](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=8&annotation=E9W2PNZS))

“We conclude that well-regularized simple deep MLPs outperform GBDT, which validates Hypothesis 2 in Section 5.3.” ([Kadra et al., 2021, p. 9](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=9&annotation=6GGRLCUN))

“The final cumulative comparison in Figure 2c provides a further result: none of the specialized previous deep learning methods (TabNet, NODE, AutoGluon Tabular) outperforms GBDT significantly. To the best of our awareness, this paper is therefore the first to demonstrate that neural networks beat GBDT with a statistically significant margin over a large-scale experimental protocol that conducts a thorough hyperparameter optimization for all methods” ([Kadra et al., 2021, p. 9](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=9&annotation=QPX8XTCY))

“The grouping reveals that a cocktail for each dataset often has at least one ingredient from every regularization family (detailed in Section 3), highlighting the need for jointly applying diverse regularization methods” ([Kadra et al., 2021, p. 9](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=9&annotation=NBV7VJ54))

“Based on these results, we conclude that regularization cocktails are time-efficient and achieve strong anytime results, which validates Hypothesis 3 in Section 5.3” ([Kadra et al., 2021, p. 9](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=9&annotation=BXVKT5DH))

“Focusing on the important domain of tabular datasets, this paper studied improvements to deep learning (DL) by better regularization techniques. We presented regularization cocktails, per-dataset-optimized combinations of many regularization techniques, and demonstrated that these improve the performance of even simple neural networks enough to substantially and significantly” ([Kadra et al., 2021, p. 9](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=9&annotation=KC4WACU5))

“surpass XGBoost, the current state-of-the-art method for tabular datasets.” ([Kadra et al., 2021, p. 10](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=10&annotation=U879IT7M))

“empirically showed that (i) modern DL regularization methods developed in the context of raw data (e.g., vision, speech, text) substantially improve the performance of deep neural networks on tabular data; (ii) regularization cocktails significantly outperform recent neural networks architectures, and most importantly iii) regularization cocktails outperform GBDT on tabular datasets.” ([Kadra et al., 2021, p. 10](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=10&annotation=363GL74Y))

“We also did not study datasets with extreme outliers, missing labels, semi-supervised data, streaming data, and many more modalities in which tabular data arises.” ([Kadra et al., 2021, p. 10](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=10&annotation=T6QRTKIK))

“An important point worth noticing is that the recent neural network architectures (Section 5.4) could also benefit from our regularization cocktails, but integrating the regularizers into these baseline libraries requires considerable coding efforts.” ([Kadra et al., 2021, p. 10](zotero://select/library/items/Z6SF869T)) ([pdf](zotero://open-pdf/library/items/299VK59K?page=10&annotation=3NS92KGN))