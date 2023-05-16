
An exception are ([[@levinTransferLearningDeep2022]]7), who find no improvements from pre-training the FT-Transformer.

Interesting resources on pre-training:
- https://arxiv.org/pdf/2109.07437.pdf
- https://phontron.com/class/anlp2022/assets/slides/anlp-07-pretraining.pdf

[[@dalche-bucSemisupervisedMarginBoost2001]]
*SSMBoost* requires semi-supervised base learners, which 
ASSEMBLE 

Boosting, as an ensemble learning framework, is one of the most powerful classification algorithms in supervised learning. Based on the gradient descent view of boosting [Mason et al., 2000], many semi-supervised boosting methods have been proposed, such as SMarginBoost [d’Alch ́ e-Buc et al., 2002], ASSEMBLE [Bennett et al., 2002], RegBoost [Chen and Wang, 2007; Chen and Wang, 2011], SemiBoost [Mallapragada et al., 2009], SERBoost [Saffari et al., 2008] and information theoretic regularization based boosting [Zheng et al., 2009], where a margin loss function is minimized over both labeled and unlabeled data by the functional gradient descent method. The effectiveness of these methods can be ascribed to their tendency to produce large margin classifiers with a small classification error. However, these algorithms were not designed to directly maximize the margin (although some of them have the effects of margin enforcing), and the objective functions are not related to the margin in the sense that one can minimize these loss functions while simultaneously achieving a bad margin [Rudin et al., 2004]. Therefore, a natural goal is to construct classifiers that directly optimize margins as measured on both labeled and unlabeled data.

See [[@vanengelenSurveySemisupervisedLearning2020]] for different approaches.
For the existing semi-supervised boosting methods [Bennett et al., 2002; Chen and Wang, 2007; d’Alche-Buc ´ et al., 2002; Mallapragada et al., 2009; Saffari et al., 2008; Zheng et al., 2009],A Direct Boosting Approach for Semi-Supervised Classification ([[@zhaiDirectBoostingApproach]]).

[[@mallapragadaSemiBoostBoostingSemiSupervised2009]] [[@bennettExploitingUnlabeledData2002]][[@dalche-bucSemisupervisedMarginBoost2001]]




- We discuss approaches in cref-[[🍪Selection Of Semisupervised Approaches]].
- Problems of tree-based approaches and neural networks in semi-supervised learning. See [[@huangTabTransformerTabularData2020]] or [[@arikTabnetAttentiveInterpretable2020]]and [[@tanhaSemisupervisedSelftrainingDecision2017]]

(- *(R2) interpretability:* The method must interpretable. -> every classifier is somewhat interpretable. Better just mention attention mechanism / transparent.)

“The paper closest to our work is Gorishniy et al. [2021], benchmarking novel algorithms, on 11 tabular datasets. We provide a more comprehensive benchmark, with 45 datasets, split across different settings (medium-sized / large-size, with/without categorical features), accounting for the hyperparameter tuning cost, to establish a standard benchmark.” ([Grinsztajn et al., 2022, p. 2](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=2&annotation=YXJLM6JN)) “FT_Transformer : a simple Transformer model combined with a module embedding categorical and numerical features, created in Gorishniy et al. [2021]. We choose this model because it was benchmarked in a convincing way against tree-based models and other tabular-specific models. It can thus be considered a “best case” for Deep learning models on tabular data.” ([Grinsztajn et al., 2022, p. 5](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=5&annotation=AHYUCL2P))


“Creating tabular-specific deep learning architectures is a very active area of research (see section 2). One motivation is that tree-based models are not differentiable, and thus cannot be easily composed and jointly trained with other deep learning blocks. Most tabular deep learning publications claim to beat or match tree-based models, but their claims have been put into question: a simple Resnet seems to be competitive with some of these new models [Gorishniy et al., 2021], and most of these methods seem to fail on new datasets [Shwartz-Ziv and Armon, 2021]. Indeed, the lack of an established benchmark for tabular data learning provides additional degrees of freedom to researchers when evaluating their method. Furthermore, most tabular datasets available online are small compared to benchmarks in other machine learning subdomains, such as ImageNet [Ima], making evaluation noisier. These issues add up to other sources of unreplicability across machine learning, such as unequal hyperparameters tuning efforts [Lipton and Steinhardt, 2019] or failure to account for statistical uncertainty in benchmarks [Bouthillier et al., 2021].” (Grinsztajn et al., 2022, p. 1)


“MLP-like architectures are not robust to uninformative features In the two experiments shown in Fig. 4, we can see that removing uninformative features (4a) reduces the performance gap between MLPs (Resnet) and the other models (FT Transformers and tree-based models), while adding uninformative features widens the gap. This shows that MLPs are less robust to uninformative features, and, given the frequency of such features in tabular datasets, partly explain the results from Sec. 4.2.” ([Grinsztajn et al., 2022, p. 7](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=7&annotation=TQSG939L))

“Why are MLPs much more hindered by uninformative features, compared to other models? One answer is that this learner is rotationally invariant in the sense of Ng [2004]: the learning procedure which learns an MLP on a training set and evaluate it on a testing set is unchanged when applying a rotation (unitary matrix) to the features on both the training and testing set. On the contrary, tree-based models are not rotationally invariant, as they attend to each feature separately, and neither are FT Transformers, because of the initial FT Tokenizer, which implements a pointwise operation theoretical link between this concept and uninformative features is provided by Ng [2004], which shows that any rotationallly invariant learning procedure has a worst-case sample complexity that grows at least linearly in the number of irrelevant features. Intuitively, to remove uninformative features, a rotationaly invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost.” ([Grinsztajn et al., 2022, p. 8](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=8&annotation=W6LGGVAC))


“An extensive line of work on tabular deep learning aims to challenge the dominance of GBDT models. Numerous tabular neural architectures have been introduced, based on the ideas of creating differentiable learner ensembles [55, 29, 77, 43, 8], incorporating attention mechanisms and transformer architectures [64, 26, 6, 34, 65, 44], as well as a variety of other approaches [70, 71, 10, 42, 23, 61]. However, recent systematic benchmarking of deep tabular models [26, 63] shows that while these models are competitive with GBDT on some tasks, there is still no universal best method. Gorishniy et al. [26] show that transformer-based models are the strongest alternative to GBDT and that ResNet and MLP models coupled with a strong hyperparameter tuning routine [2] offer competitive baselines. Similarly, Kadra et al. [40] find that carefully regularized MLPs are competitive. In a follow-up work, Gorishniy et al. [27] show that transformer architectures equipped with advanced embedding schemes for numerical features bridge the performance gap between deep tabular models and GBDT” (Levin et al., 2022, p. 3)


**Why tabular data is hard:**
- “Tabular data is a database that is structured in a tabular form. It arranges data elements in vertical columns (features) and horizontal rows (samples)” ([Yoon et al., 2020, p. 1](zotero://select/library/items/XSYUS7JZ)) ([pdf](zotero://open-pdf/library/items/78GQQ36U?page=1&annotation=8MAKL2B9))
- Challenges of learning of tabular data can be found in [[@borisovDeepNeuralNetworks2022]] e. g. both 


**Coarse grained selection:**
- Show that there is a general concensus, that gradient boosted trees and neural networks work best. Show that there is a great bandwith of opinions and its most promising to try both. Papers: [[@shwartz-zivTabularDataDeep2021]]
- selection is hard e. g., in deep learning, as there are no universal benchmarks and robust, battle tested approaches for tabular data compared to other data sources. (see [[@gorishniyRevisitingDeepLearning2021]])
- reasons why deep learning on tabular data is challenging [[@shavittRegularizationLearningNetworks2018]] (use more as background citation)
- Taxonomy of approaches can be found in [[@borisovDeepNeuralNetworks2022]] 
![[tabular-learning-architectures.png]]

- Perform a wide (ensemble) vs. deep (neural net) comparison. This is commonly done in literature. Possible papers include:
	- [[@gorishniyRevisitingDeepLearning2021]] compare DL models with Gradient Boosted Decision Trees and conclude that there is still no universally superior solution.
	- For "shallow" state-of-the-art are ensembles such as GBMs. (see [[@gorishniyRevisitingDeepLearning2021]])
	- Deep learning for tabular data could potentially yield a higher performance and allow to combine tbular data with non-tabular data such as images, audio or other data that can be easily processed with deep learning. [[@gorishniyRevisitingDeepLearning2021]]
	- Despite growing number of novel (neural net) architectures, there is still no simple, yet reliable solution that achieves stable performance across many tasks. 
	- [[@arikTabnetAttentiveInterpretable2020]] Discuss a number of reasons why decisiion tree esembles dominate neural networks for tabular data.
	- [[@huangTabTransformerTabularData2020]] argue that tree-based ensembles are the leading approach for tabular data. The base this on the prediction accuracy, the speed of training and the ability to interpret the models. However, they list sevre limitations. As such they are not suitabl efor streaming data, multi-modality with tabular data e. g. additional image date and do not support semi-supervised learning by default.
- Choose neural network architectures, that are tailored towards tabular data.

**Camparison:**
- large number of datapoints -> Transformers are data hungry (must be stated in the [[@vaswaniAttentionAllYou2017]] paper)
- Nice formulation and overview of the dominance of GBT and deep learning is given in [[@levinTransferLearningDeep2022]]
- for use of transformer-based models in finance see[[@zouStockMarketPrediction2022]]
- Non-parametric model of [[@kossenSelfAttentionDatapointsGoing2021]]

- Sophisticated neural network architectures might not be required, but rather a mix of regularization approaches to regularize MLPs [[@kadraWelltunedSimpleNets2021]].
- See [[@huangTabTransformerTabularData2020]] that point out common problems of comparsions between gbts and dl.


**Regularization:** “Why are MLPs much more hindered by uninformative features, compared to other models? One answer is that this learner is rotationally invariant in the sense of Ng [2004]: the learning procedure which learns an MLP on a training set and evaluate it on a testing set is unchanged when applying a rotation (unitary matrix) to the features on both the training and testing set. On the contrary, tree-based models are not rotationally invariant, as they attend to each feature separately, and neither are FT Transformers, because of the initial FT Tokenizer, which implements a pointwise operation theoretical link between this concept and uninformative features is provided by Ng [2004], which shows that any rotationallly invariant learning procedure has a worst-case sample complexity that grows at least linearly in the number of irrelevant features. Intuitively, to remove uninformative features, a rotationaly invariant algorithm has to first find the original orientation of the features, and then select the least informative ones: the information contained in the orientation of the data is lost.” ([Grinsztajn et al., 2022, p. 8](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=8&annotation=W6LGGVAC))
“The paper closest to our work is Gorishniy et al. [2021], benchmarking novel algorithms, on 11 tabular datasets. We provide a more comprehensive benchmark, with 45 datasets, split across different settings (medium-sized / large-size, with/without categorical features), accounting for the hyperparameter tuning cost, to establish a standard benchmark.” ([Grinsztajn et al., 2022, p. 2](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=2&annotation=YXJLM6JN)) “FT_Transformer : a simple Transformer model combined with a module embedding categorical and numerical features, created in Gorishniy et al. [2021]. We choose this model because it was benchmarked in a convincing way against tree-based models and other tabular-specific models. It can thus be considered a “best case” for Deep learning models on tabular data.” ([Grinsztajn et al., 2022, p. 5](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=5&annotation=AHYUCL2P))

“MLP-like architectures are not robust to uninformative features In the two experiments shown in Fig. 4, we can see that removing uninformative features (4a) reduces the performance gap between MLPs (Resnet) and the other models (FT Transformers and tree-based models), while adding uninformative features widens the gap. This shows that MLPs are less robust to uninformative features, and, given the frequency of such features in tabular datasets, partly explain the results from Sec. 4.2.” ([Grinsztajn et al., 2022, p. 7](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=7&annotation=TQSG939L))

“Tuning hyperparameters does not make neural networks state-of-the-art Tree-based models are superior for every random search budget, and the performance gap stays wide even after a large number of random search iterations. This does not take into account that each random search iteration is generally slower for neural networks than for tree-based models (see A.2).” ([Grinsztajn et al., 2022, p. 6](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=6&annotation=K2FYJND8)) [[@grinsztajnWhyTreebasedModels2022]]

“Our contributions are as follow: 1. We create a new benchmark for tabular data, with a precise methodology for choosing and preprocessing a large number of representative datasets. We share these datasets through OpenML [Vanschoren et al., 2014], which makes them easy to use. 2. We extensively compare deep learning models and tree-based models on generic tabular datasets in multiple settings, accounting for the cost of choosing hyperparameters.” (Grinsztajn et al., 2022, p. 2)

with several works claiming to outperform tree-based learners for tabular classification tasks

corrobates

*Differentiable Trees*
the ideas of creating differentiable learner ensembles [55, 29, 77, 43, 8], i

corroborates this

Tabluar specific deep learning architectures


-> NODE poor performance? (in [[@gorishniyRevisitingDeepLearning2021]]) fairly reasonable / poor in ([[@kadraWelltunedSimpleNets2021]]) / ([[@borisovDeepNeuralNetworks2022]])
Differentiable trees take inspiration from tree-based learners. 

A class of of networks that takes inspiration from tree-based learners are differentiable trees.
to warrants
By the virtue
take inspiration from 
Differentiable trees try to adapt the success of 
These architectures are often based 
Recently, several deep learning approaches claim to excel the performance of gradient boosting. 
Why do they make sense?
Why gradient-boosting? Why not random forests?
Classical neural networks 
Parts of this success lies in the robustness to noise, 
A particularily promising strand of research are attention-based models.
Standard-MLPs. 
A line of research, including 
A fair comparison betw
These results, contradict 

(🚧short discussion what attention is)

(What can be seen? General overview for neural nets in [[@melisStateArtEvaluation2017]]. Also, [[@kadraWelltunedSimpleNets2021]])


Neural Oblivious Decision Ensembles (NODE). The NODE network [Popov et al., 2020] contains equal-depth oblivious decision trees (ODTs), which are differentiable such that error gradients can backpropagate through them. Like classical decision trees, ODTs split data according to selected features and compare each with a learned threshold. However, only one feature is chosen at each level, resulting in a balanced ODT that can be differentiated. Thus, the complete model provides an ensemble of differentiable trees.

“In its essence, the proposed NODE architecture generalizes CatBoost, making the splitting feature choice and decision tree routing differentiable. As a result, the NODE architecture is fully differentiable and could be incorporated in any computational graph of existing DL packages,” (Popov et al., 2019, p. 2)

“We introduce the Neural Oblivious Decision Ensemble (NODE) architecture with a layer-wise structure similar to existing deep learning models. In a nutshell, our architecture consists of differentiable oblivious decision trees (ODT) that are trained end-to-end by backpropagation. We describe our implementation of the differentiable NODE layer in Section 3.1, the full model architecture in Section 3.2, and the training and inference procedures in section 3.3.” (Popov et al., 2019, p. 3)
“Differentiable trees. The significant drawback of tree-based approaches is that they usually do not allow end-to-end optimization and employ greedy, local optimization procedures for tree construction.” (Popov et al., 2019, p. 2)

“Thus, they cannot be used as a component for pipelines, trained in an end-to-end fashion. To address this issue, several works (Kontschieder et al., 2015; Yang et al., 2018; Lay et al., 2018) 1https://github.com/Qwicen/node” (Popov et al., 2019, p. 2)

“propose to ”soften” decision functions in the internal tree nodes to make the overall tree function and tree routing differentiable. In our work, we advocate the usage of the recent entmax transformation (Peters et al., 2019) to ”soften” decision trees. We confirm its advantages over the previously proposed approaches in the experimental section” (Popov et al., 2019, p. 3)
“To address this issue, several works (Hazimeh et al., 2020; Kontschieder et al., 2015; Popov et al., 2020; Yang et al., 2018) propose to “smooth” decision functions in the internal tree nodes to make the overall tree function and tree routing differentiable. While the methods of this family can outperform GBDT on some tasks (Popov et al., 2020), in our experiments, they do not consistently outperform ResNet.” (Gorishniy et al., 2021, p. 2)

hat combines.... In large-scale,  