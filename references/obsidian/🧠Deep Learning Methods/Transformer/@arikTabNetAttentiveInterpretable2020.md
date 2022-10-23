
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
## Notes Sebastian Raschka
- Based on my personal experience, TabNet is the first deep learning architecture for tabular data that gained widespread attention (no pun intended).
- TabNet is based on a sequential attention mechanism, showing that self-supervised learning with unlabeled data can improve the performance over purely supervised training regimes in tabular settings.
- Across six synthetic datasets, TabNet outperforms other methods on 3 out of 6 cases. However, XGBoost was omitted, and the tree-based reference method is [extremely randomized trees](https://link.springer.com/article/10.1007/s10994-006-6226-1) rather than random forests.
- Across 4 KDD datasets, TabNet ties with CatBoost and XGboost on 1 dataset and performs almost as well as the gradient-boosted tree methods on the remaining three datasets.

## Reasons for domination of ensemble of decision trees for tabular data
“First, because DT-based approaches have certain benefits: (i) they are representionally efficient for decision manifolds with approximately hyperplane boundaries which are common in tabular data; and (ii) they are highly interpretable in their basic form (e.g. by tracking decision nodes) and there are popular post-hoc explainability methods for their ensemble form, e.g. (Lundberg, Erion, and Lee 2018) – this is an important concern in many real-world applications; (iii) they are fast to train. Second, because previously-proposed DNN architectures are not well-suited for tabular data: e.g. stacked convolutional layers or multi-layer perceptrons (MLPs) are vastly overparametrized – the lack of appropriate inductive bias often causes them to fail to find optimal solutions for tabular decision manifolds (Goodfellow, Bengio, and Courville 2016; Shavitt and Segal 2018; Xu et al. 2019).” (Arik and Pfister, 2020, p. 1)
