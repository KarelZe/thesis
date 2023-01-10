*title:* Transfer Learning with Deep Tabular Models
*authors:* Roman Levin, Valeriia Cherepanova, Avi Schwarzschild, Arpit Bansal, C. Bayan Bruss, Tom Goldstein, Andrew Gordon Wilson, Micah Goldblum
*year:* 2022
*tags:* #pretraining #semi-supervised #transformer #tabular-data 
*status:* #📥
*related:*
- [[@huangTabTransformerTabularData2020]]
- [[@devlinBERTPretrainingDeep2019]]
- [[@somepalliSAINTImprovedNeural2021]]
## Notes 📍

## Annotations 📖
“Recent work on deep learning for tabular data demonstrates the strong performance of deep tabular models, often bridging the gap between gradient boosted decision trees and neural networks” ([Levin et al., 2022, p. 1](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=1&annotation=UJML3DMN))

“In this work, we demonstrate that upstream data gives tabular neural networks a decisive advantage over widely used GBDT models.” ([Levin et al., 2022, p. 1](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=1&annotation=DEM6R7YA))

“We propose a realistic medical diagnosis benchmark for tabular transfer learning, and we present a how-to guide for using upstream data to boost performance with a variety of tabular neural network architectures.” ([Levin et al., 2022, p. 1](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=1&annotation=A9NBJEWS))

“Leading methods in tabular deep learning ] now perform on par with the traditionally dominant gradient boosted decision trees (GBDT) .” ([Levin et al., 2022, p. 1](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=1&annotation=RI6XRLXP))

“In this work, we show that deep tabular models with transfer” ([Levin et al., 2022, p. 1](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=1&annotation=XNIVGKZG))

“learning definitively outperform their classical counterparts when auxiliary upstream pretraining data is available and the amount of downstream data is limited. Importantly, we find representation learning with tabular neural networks to be more powerful than gradient boosted decision trees with stacking – a strong baseline leveraging knowledge transfer from the upstream data with classical methods.” ([Levin et al., 2022, p. 2](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=2&annotation=6VW4EV4X))

“Finally, we propose a pseudo-feature method which enables transfer learning when upstream and downstream feature sets differ. As tabular data is highly heterogeneous, the problem of downstream tasks whose formats and features differ from those of upstream data is common and has been reported to complicate knowledge transfer.” ([Levin et al., 2022, p. 2](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=2&annotation=8L8MVRIK))

“In the case that upstream data is missing a column, we first pre-train a model on the upstream data without that feature. We then fine-tune the pre-trained model on downstream data to predict values in the column absent from the upstream data. Finally, after assigning pseudo-values of this feature to the upstream samples, we re-do the pre-training and transfer the feature extractor to the downstream task.” ([Levin et al., 2022, p. 2](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=2&annotation=DE2EITAR))

“We compare supervised and self-supervised pre-training strategies and find that the supervised pre-training leads to more transferable features in the tabular domain.” ([Levin et al., 2022, p. 2](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=2&annotation=PJ6YN4UD))

“We propose a pseudo-feature method for aligning the upstream and downstream feature sets in heterogeneous data, addressing a common obstacle in the tabular domain.” ([Levin et al., 2022, p. 2](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=2&annotation=D2VJVU36))

“An extensive line of work on tabular deep learning aims to challenge the dominance of GBDT models. Numerous tabular neural architectures have been introduced, based on the ideas of creating differentiable learner ensembles, incorporating attention mechanisms and transformer architectures , as well as a variety of other approaches .” ([Levin et al., 2022, p. 3](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=3&annotation=FQMTNV38))

“In the tabular data domain, a recent review paper finds that transfer learning is underexplored and that the question of how to perform knowledge transfer and leverage upstream data remains open.” ([Levin et al., 2022, p. 3](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=3&annotation=3YECMQHZ))

“Multiple works mention that transfer learning in the tabular domain is challenging due to the highly heterogeneous nature of tabular data” ([Levin et al., 2022, p. 3](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=3&annotation=GPDDEYPW))

“Stacking could also be seen as a form of leveraging upstream knowledge in classical methods ([Levin et al., 2022, p. 3](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=3&annotation=JQAR5YSM))

“Recently, SSL has been adopted in the tabular domain for semi-supervised learning (saint, tabtransformer etc.). Contrastive pre-training on auxilary unlabelled data (saint) and MLM-like approaches (tabtransformer) have been shown to provide gains over training from scratch for transformer tabular architectures in cases of limited labelled data.” ([Levin et al., 2022, p. 3](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=3&annotation=CYRD7A54))

“For neural networks, we use transformer-based architectures found to be the most competitive with GBDT tabular approaches . The specific implementations we consider include the recent FT-Transformer  and TabTransformer . We do not include implementations with inter-sample attention in our experiments since these do not lend themselves to scenarios with extremely limited downstream data.” ([Levin et al., 2022, p. 5](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=5&annotation=BDI8DSH3))

“To summarize, we implement four transfer learning setups for neural networks: • Linear head atop a frozen feature extractor • MLP head atop a frozen feature extractor • End-to-end fine-tuned feature extractor with a linear head • End-to-end fine-tuned feature extractor with an MLP head We compare the above setups to the following baselines: • Neural models trained from scratch on downstream data • CatBoost and XGBoost with and without stacking” ([Levin et al., 2022, p. 5](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=5&annotation=FG2HLIQV))

“We use stacking for GBDT models to build a stronger baseline which leverages the upstream data. To implement stacking, we first train upstream GBDT models to predict the 11 upstream targets and then concatenate their predictions to the downstream data features when training downstream GBDT models.” ([Levin et al., 2022, p. 5](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=5&annotation=AAA7L7LM))

“In particular, for GBDT models and neural baselines trained from scratch, we tune the hyperparameters on a single randomly chosen upstream target with the same number of training samples as available in the downstream task, since hyperparameters depend strongly on the sample size. The optimal hyperparameters are chosen based on the upstream validation set, where validation data is plentiful.” ([Levin et al., 2022, p. 5](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=5&annotation=JPW6T3C5))

“or deep models with transfer learning, we tune the hyperparameters on the full upstream data using the available large upstream validation set with the goal to obtain the best performing feature extractor for the pre-training multi-target task. We then fine-tune this feature extractor with a small learning rate on the downstream data.” ([Levin et al., 2022, p. 6](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=6&annotation=6X87ZTT9))

“Figure 2 presents average model ranks on the downstream tasks as a heatmap where the warmer colors represent better overall rank. The performance of every model is shown on the dedicated panel of the heatmap with the results for different transfer learning setups presented in columns.” ([Levin et al., 2022, p. 6](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=6&annotation=EE6GK8QU))

“We emphasize that knowledge transfer with stacking, while providing strong boosts compared to from-scratch GBDT training (see Stacking and FS columns of GBDT), still decisively falls behind the deep tabular models with transfer learning, suggesting that representation learning for tabular data is significantly more powerful and allows neural networks to transfer richer information than simple predictions learned on the upstream tasks.” ([Levin et al., 2022, p. 7](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=7&annotation=QI464QKU))

“Representation learning with deep tabular models provides significant gains over strong GBDT baselines leveraging knowledge transfer from the upstream data through stacking. The gains are especially pronounced in low data regimes.” ([Levin et al., 2022, p. 7](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=7&annotation=VJXRWKWK))

“We use the Masked Language Model (MLM) pre-training recently adapted to tabular data  and the tabular version of contrastive learning . Since both methods were proposed for tabular transformer architectures, we conduct the experiments with the FT-Transformer model.” ([Levin et al., 2022, p. 7](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=7&annotation=TWFACI28))

“Masked Language Modeling (MLM) was first proposed for language models by Devlin et al as a powerful unsupervised learning strategy. MLM involves training a model to predict tokens in text masked at random so that its learned representations contain information useful for reconstructing these masked tokens. In the tabular domain, instead of masking tokens, a random subset of features is masked for each sample, and the masked values are predicted in a multi-target classification manner.” ([Levin et al., 2022, p. 8](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=8&annotation=A3JDIAAT))

“Meanwhile, the network is also trained to map negative pairs, or augmented views of different base examples, far apart in feature space. We use the implementation of contrastive learning from Somepalli et al..” ([Levin et al., 2022, p. 8](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=8&annotation=P5Q73BEY))

“Contrastive pre-training produces better results than training from scratch on the downstream data when using a linear head, but it is still inferior to supervised pre-training. Tabular MLM pretraining also falls behind the supervised strategy and performs comparably to training from scratch in the lower data regimes but leads to a weaker downstream model in the higher data regimes.” ([Levin et al., 2022, p. 8](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=8&annotation=X725JIMF))

“While so far we have worked with upstream and downstream tasks which shared a common feature space, in the real world, tabular data is highly heterogeneous and upstream data having a different set of features from downstream data is a realistic problem .” ([Levin et al., 2022, p. 8](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=8&annotation=7EG4EW44))

“Suppose our upstream data (Xu, Yu) is missing a feature xnew present in the downstream data (Xd, Yd). We then use transfer learning in stages. As the diagram on the left panel of Figure 4 shows: 1. Pre-train feature extractor f : Xu → Yu on upstream data (Xu, Yu) without feature xnew” ([Levin et al., 2022, p. 8](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=8&annotation=J6LUQ443))

“2. Fine-tune the feature extractor f on the downstream samples Xd to predict xnew as the target and obtain a model ˆ f : Xd \ {xnew} → xnew. 3. Use the model ˆ f to assign pseudo-values ˆ xnew of the missing feature to the upstream samples: ˆ xnew = ˆ f (Xu) thus obtaining augmented upstream data (Xu ∪ {ˆ xnew}, Yu). 4. Finally, we can leverage the augmented upstream data (Xu ∪ {ˆ xnew}, Yu) to pre-train a feature extractor which we will fine-tune on the original downstream task (Xd, Yd) using all of the available downstream features.” ([Levin et al., 2022, p. 9](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=9&annotation=E95X3FWA))

“The bottom heatmap represents the opposite scenario of the upstream data having an additional feature not available in the downstream data. To ensure that the features we experiment with are meaningful and contain useful information, for each task we chose important features according to GBDT feature importances. We observe that in both experiments, using the pseudo feature is better than doing transfer learning with without the missing feature at all.” ([Levin et al., 2022, p. 9](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=9&annotation=MG84XCZ8))

“Pre-training data gives tabular neural networks a distinct advantage over decision tree baselines, which persists even when the XGBoost and CatBoost are allowed to transfer knowledge through stacking and hyperparameter transfer.” ([Levin et al., 2022, p. 10](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=10&annotation=BPVMPZSN))

“Knowledge transfer can still be exploited even when there is a mismatch between upstream and downstream feature sets by leveraging pseudo-feature methods.” ([Levin et al., 2022, p. 10](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=10&annotation=PK8ISXE9))

“Supervised pre-training is significantly more effective than self-supervised alternatives in the tabular domain where SSL methods are not thoroughly explored.” ([Levin et al., 2022, p. 10](zotero://select/library/items/GNKZPFYK)) ([pdf](zotero://open-pdf/library/items/TRH7QFZ2?page=10&annotation=Y3C7XSGL))