*title:* Transfer Learning with Deep Tabular Models
*authors:* Roman Levin, Valeriia Cherepanova, Avi Schwarzschild, Arpit Bansal, C. Bayan Bruss, Tom Goldstein, Andrew Gordon Wilson, Micah Goldblum
*year:* 2022
*tags:* #deep-learning #gradient_boosting #semi-supervised
*status:* #📥
*related:* 
- [[@borisovDeepNeuralNetworks2022]]

# Notes  Sebastian Raschka
-   In contrast to gradient boosting, deep learning methods for tabular data can be pretrained on upstream data to increase performance on the target dataset.
-   Supervised pretraining is better than self-supervised pretraining in a tabular dataset context.
-   Multilayer perceptrons outperform transformer-based deep neural networks if target data is scarce.
-   Proposes a pseudo-feature method for cases where the upstream and target feature sets differ.
-   Medical diagnosis benchmark dataset; patient data with 11 diagnosis targets where features between upstream and target data are related but may differ.
# Annotations