
title: TabTransformer: Tabular Data Modeling Using Contextual Embeddings
authors: Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin
year: 2020
*tags:* #deep-learning #gradient_boosting #semi-supervised #tabular-data #supervised-learning 
*status:* #📥
*related:* 
- [[@borisovDeepNeuralNetworks2022]]
## Notes Sebastian Raschka
-   Several open-source implementations are available on GitHub, however, I could not find the official implementation, so the results from this paper must be taken with a grain of salt.
-   The paper proposes a transformer-based architecture based on self-attention that can be applied to tabular data.
-   In addition to the purely supervised regime, the authors propose a semi-supervised approach leveraging unsupervised pre-training.
-   Looking at the average AUC across 15 datasets, the proposed TabTransformer (82.8) is on par with gradient-boosted trees (82.9).


## Notes
- TabTransformer only learns contextual embeddings on categorical features. Continous features are concatenated with the embeddings and fed into a vanilla neural network. [[@somepalliSAINTImprovedNeural2021]] critize that information about correlations between categorical and continous features are lost, as continous features are not passed through the transformer block.

## Annotations

“The state-of-the-art for modeling tabular data is treebased ensemble methods such as the gradient boosted decision trees (GBDT)” (Huang et al., 2020, p. 1)

“The tree-based ensemble models can achieve competitive prediction accuracy, are fast to train and easy to interpret. These benefits make them highly favourable among machine learning practitioners. However, the tree-based models have several limitations in comparison to deep learning models. (a) They are not suitable for continual training from streaming data, and do not allow efficient end-to-end learning of image/text encoders in presence of multi-modality along with tabular data. (b) In their basic form they are not suitable for state-of-the-art. semi-supervised learning methods.” (Huang et al., 2020, p. 1)