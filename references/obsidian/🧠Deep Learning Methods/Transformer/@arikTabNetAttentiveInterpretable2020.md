
title: TabNet: Attentive Interpretable Tabular Learning
authors: Sercan O. Arik, Tomas Pfister
year: 2020
*code:* [https://github.com/google-research/google-research/tree/master/tabnet](https://github.com/google-research/google-research/tree/master/tabnet)
*tags:* #deep-learning #gradient_boosting #semi-supervised
*status:* #📥
*related:* 
- [[@gorishniyRevisitingDeepLearning2021]]
- [[@borisovDeepNeuralNetworks2022]]
- [[@huangTabTransformerTabularData2020]]
*code:*
 - See this post for some additional insights on Ghost BatchNorm. https://medium.com/deeplearningmadeeasy/ghost-batchnorm-explained-e0fa9d651e03
## Notes Sebastian Raschka
- Based on my personal experience, TabNet is the first deep learning architecture for tabular data that gained widespread attention (no pun intended).
- TabNet is based on a sequential attention mechanism, showing that self-supervised learning with unlabeled data can improve the performance over purely supervised training regimes in tabular settings.
- Across six synthetic datasets, TabNet outperforms other methods on 3 out of 6 cases. However, XGBoost was omitted, and the tree-based reference method is [extremely randomized trees](https://link.springer.com/article/10.1007/s10994-006-6226-1) rather than random forests.
- Across 4 KDD datasets, TabNet ties with CatBoost and XGboost on 1 dataset and performs almost as well as the gradient-boosted tree methods on the remaining three datasets.

## Notes
- TabNet uses GhostBatchNorm [[@hofferTrainLongerGeneralize2017]]
- TabNet mimics decision trees by placing importance only on a few features at each layer. The attentation layers replace dot-product self-attention with a sparse layer, that allows only certain features to traverse. (found in [[Semi-supervised Learning/@somepalliSAINTImprovedNeural2021]])

## Reasons for domination of ensemble of decision trees for tabular data
“First, because DT-based approaches have certain benefits: (i) they are representionally efficient for decision manifolds with approximately hyperplane boundaries which are common in tabular data; and (ii) they are highly interpretable in their basic form (e.g. by tracking decision nodes) and there are popular post-hoc explainability methods for their ensemble form, e.g. (Lundberg, Erion, and Lee 2018) – this is an important concern in many real-world applications; (iii) they are fast to train. Second, because previously-proposed DNN architectures are not well-suited for tabular data: e.g. stacked convolutional layers or multi-layer perceptrons (MLPs) are vastly overparametrized – the lack of appropriate inductive bias often causes them to fail to find optimal solutions for tabular decision manifolds (Goodfellow, Bengio, and Courville 2016; Shavitt and Segal 2018; Xu et al. 2019).” (Arik and Pfister, 2020, p. 1)


## Annotations
“First, because DT-based approaches have certain benefits: (i) they are representionally efficient for decision manifolds with approximately hyperplane boundaries which are common in tabular data; and (ii) they are highly interpretable in their basic form (e.g. by tracking decision nodes) and there are popular post-hoc explainability methods for their ensemble form, e.g. (Lundberg, Erion, and Lee 2018) – this is an important concern in many real-world applications; (iii) they are fast to train. Second, because previously-proposed DNN architectures are not well-suited for tabular data: e.g. stacked convolutional layers or multi-layer perceptrons (MLPs) are vastly overparametrized – the lack of appropriate inductive bias often causes them to fail to find optimal solutions for tabular decision manifolds (Goodfellow, Bengio, and Courville 2016; Shavitt and Segal 2018; Xu et al. 2019).” ([Arik and Pfister, 2020, p. 1](zotero://select/library/items/EH5DCRUW)) ([pdf](zotero://open-pdf/library/items/TPDKX93V?page=1&annotation=ZFASIUHV))

“Finally, for the first time for tabular data, we show significant performance improvements by using unsupervised pre-training to predict masked features (see Fig. 2” ([Arik and Pfister, 2020, p. 1](zotero://select/library/items/EH5DCRUW)) ([pdf](zotero://open-pdf/library/items/TPDKX93V?page=1&annotation=X5WBP7CA))

“Self-supervised learning: Unsupervised representation learning improves supervised learning especially in small data regime (Raina et al. 2007). Recent work for text (Devlin et al. 2018) and image (Trinh, Luong, and Le 2019) data has shown significant advances – driven by the judicious choice of the unsupervised learning objective (masked input prediction) and attention-based deep learning” ([Arik and Pfister, 2020, p. 3](zotero://select/library/items/EH5DCRUW)) ([pdf](zotero://open-pdf/library/items/TPDKX93V?page=3&annotation=B6TI27FJ))

“Tabular self-supervised learning: We propose a decoder architecture to reconstruct tabular features from the TabNet encoded representations. The decoder is composed of feature transformers, followed by FC layers at each decision step. The outputs are summed to obtain the reconstructed features.” ([Arik and Pfister, 2020, p. 5](zotero://select/library/items/EH5DCRUW)) ([pdf](zotero://open-pdf/library/items/TPDKX93V?page=5&annotation=3QTDAR2A))

“Table 10: Self-supervised tabular learning results. Mean and std. of accuracy (over 15 runs) on Forest Cover Type, varying the size of the training dataset for supervised fine-tunin” ([Arik and Pfister, 2020, p. 9](zotero://select/library/items/EH5DCRUW)) ([pdf](zotero://open-pdf/library/items/TPDKX93V?page=9&annotation=IAGMU4FT))
