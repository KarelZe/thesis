The selection of methods in previous works is arbitrary and guided computational constraints (cp. cref-[[👪Related Work]]). Furthermore, the selection leaves out advancements in machine learning on tabular data.
This paper provides a succinct discussion of these gaps and selects a set of supervised classifiers based on empirical evidence. 

We impose the following requirements a classifier must fulfill:
*performance:* The approach must deliver state-of-the-art performance in tabular classification tasks. Typically, trades are provided as tabular datasets, that consist of rows and columns. The classifier must be well-suited for classification on tabular data.
*scalability:* The approach must scale to datasets with $>$ 10 Mio. samples. Due to the high trading activity and the long data history, datasets can contain many samples. Classifiers must scale to large to large data volumes.
*extensibility:* The approach must be extendable to train on partially-labelled trades. Most definitions of the trade initiator apply only to a subset of trades e.g., certain order types. The so-excluded trades can still be valuable in training a classifier. The classifier, however, must support it.

These results, contradict 


Thus, our study considers gradient boosting and the FT-Transformer. This comparison is particularly appealing, as it compares tree-based learners against neural nets, as well as wide ensembles against deep neural networks.

#gbm #transformer #supervised-learning #deep-learning 

- We discuss approaches in cref-[[🍪Selection Of Semisupervised Approaches]].
- Problems of tree-based approaches and neural networks in semi-supervised learning. See [[@huangTabTransformerTabularData2020]] or [[@arikTabNetAttentiveInterpretable2020]]and [[@tanhaSemisupervisedSelftrainingDecision2017]]

(- *(R2) interpretability:* The method must interpretable. -> every classifier is somewhat interpretable. Better just mention attention mechanism / transparent.)

“The paper closest to our work is Gorishniy et al. [2021], benchmarking novel algorithms, on 11 tabular datasets. We provide a more comprehensive benchmark, with 45 datasets, split across different settings (medium-sized / large-size, with/without categorical features), accounting for the hyperparameter tuning cost, to establish a standard benchmark.” ([Grinsztajn et al., 2022, p. 2](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=2&annotation=YXJLM6JN)) “FT_Transformer : a simple Transformer model combined with a module embedding categorical and numerical features, created in Gorishniy et al. [2021]. We choose this model because it was benchmarked in a convincing way against tree-based models and other tabular-specific models. It can thus be considered a “best case” for Deep learning models on tabular data.” ([Grinsztajn et al., 2022, p. 5](zotero://select/library/items/G3KP2Z9W)) ([pdf](zotero://open-pdf/library/items/A3KU4A43?page=5&annotation=AHYUCL2P))

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
	- [[@arikTabNetAttentiveInterpretable2020]] Discuss a number of reasons why decisiion tree esembles dominate neural networks for tabular data.
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