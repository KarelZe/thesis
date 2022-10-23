*title:* Well-Tuned Simple Nets Excel on Tabular Datasets
*authors:* Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka
*year:* 2020
*tags:* #tabular-data #deep-learning #neural_network 
*status:* #📥
*related:*
- [[@borisovDeepNeuralNetworks2022]]
*code:* [https://github.com/releaunifreiburg/WellTunedSimpleNets](https://github.com/releaunifreiburg/WellTunedSimpleNets)
# Notes Sebastian Raschka
-   Using a combination of modern regularization techniques, the authors find that simple multi-layer perceptron (MLP) can outperform both specialized neural network architectures (TabNet) and gradient boosting machines (XGBoost and Catboost)
-   The MLP base architecture consisted of 9-layers with 512 units each (excluding the output layer) and was tuned with a cosine annealing scheduler.
-   The following 13 regularization techniques were considered in this study. Implicit: (1) BatchNorm, (2) stochastic weight averaging, (3) Look-ahead optimizer. (4) Weight decay. Ensemble: (5) Dropout, (6) snapshot ensembles. Structural: (7) skip connections, (8) Shake-Drop, (9) Shake-Shake. Augmentation: (10) Mix-Up, (11) Cut-Mix, (12) Cut-Out, (13) FGSM adversarial learning
-   The comparison spans 40 tabular datasets for classification, ranging from 452 to 416,188 examples. In 19 out of the 40 datasets, an MLP with a mix regularization techniques outperformed any other method evaluated in this study.
-   Caveats: the authors included NODE in the comparison but used did not tune its hyperparameters. Also, the regularization techniques were not applied to the other neural network architectures (TabNet and NODE).

# Annotations