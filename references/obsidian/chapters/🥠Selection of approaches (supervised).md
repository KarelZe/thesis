

<mark style="background: #CACFD9A6;">“An extensive line of work on tabular deep learning aims to challenge the dominance of GBDT models. Numerous tabular neural architectures have been introduced, based on the ideas of creating differentiable learner ensembles [55, 29, 77, 43, 8], incorporating attention mechanisms and transformer architectures [64, 26, 6, 34, 65, 44], as well as a variety of other approaches [70, 71, 10, 42, 23, 61]. However, recent systematic benchmarking of deep tabular models [26, 63] shows that while these models are competitive with GBDT on some tasks, there is still no universal best method. Gorishniy et al. [26] show that transformer-based models are the strongest alternative to GBDT and that ResNet and MLP models coupled with a strong hyperparameter tuning routine [2] offer competitive baselines. Similarly, Kadra et al. [40] find that carefully regularized MLPs are competitive. In a follow-up work, Gorishniy et al. [27] show that transformer architectures equipped with advanced embedding schemes for numerical features bridge the performance gap between deep tabular models and GBDT” (Levin et al., 2022, p. 3)</mark>

<mark style="background: #FFB86CA6;">“Semi-supervised boosting methods have been studied extensively over the past two decades. The success achieved by supervised boosting methods, such as AdaBoost (Freund and Schapire 1997), gradient boosting, and XGBoost (Chen and Guestrin 2016), provides ample motivation for bringing boosting to the semi-supervised setting. Furthermore, the pseudo-labelling approach of self-training and co-training can be easily extended to boosting methods.” (Engelen and Hoos, 2020, p. 391)</mark> [[@vanengelenSurveySemisupervisedLearning2020]]

<mark style="background: #BBFABBA6;">“Tabular data is a database that is structured in a tabular form. It arranges data elements in vertical columns (features) and horizontal rows (samples)” ([Yoon et al., 2020, p. 1](zotero://select/library/items/XSYUS7JZ)) ([pdf](zotero://open-pdf/library/items/78GQQ36U?page=1&annotation=8MAKL2B9))</mark>

Due to a unsatisfactory research situation, for trade calssification (see chapter [[👪Related work]]) we base

We define the following criteria a model should meet:

- for comparability 

- large number of datapoints -> Transformers are data hungry (must be stated in the [[@vaswaniAttentionAllYou2017]] paper)

- Use classification methods (*probabilistic classifier*) that can return probabilities instead of class-only for better analysis. Using probabilistic trade classification rules might have been studied in [[@easleyDiscerningInformationTrade2016]]
- nice overview also in [[@levinTransferLearningDeep2022]]
- survey on transformers for tabular data [[@badaroTransformersTabularData]]
- Introduce a classifcation that differentiates between supervised, unsupervised, reenforcement learning and semi-supervised learning. 
- Introduce the concept of classification as a variant of supervised learning. 
- Could be supervised if all labels are known
- Could be semi-supervised if only some of the labels are known. Cover later.
- Use probabilistic classifier to allow for more in-depth analysis. Similar to BVC paper. Also explain why probabilistic clf makes sense -> Opens up new chances for evaluations and extensions. But comes with its own problems (see e. g., decision trees)
- Search for paper that performed a comparsion between Gradient Boosted Trees and Neural Net on large set of data sets....
- What works in similar use cases? What are similar use cases?
- Establish criteria for choosing an architecture:
	- **performance** That is, approach must deliver state-of-the-art performance in similar problems.
	- **interpretability** Classical approaches are transparent in a sense that we know how the decision was derived. In the best case try to aim for local and global interpretability. Think about how interpretability can be narrowed down? Note supervisor wants to see if her features are also important to the model. 
- Perform a model discussion on results from similar domains. Most broadly it's a classification problem on tabular data. Thus, architectures for tabular data should be considered.
- Perform a wide (ensemble) vs. deep (neural net) comparison. This is commonly done in literature. Possible papers include:
	- [[@gorishniyRevisitingDeepLearning2021]] compare DL models with Gradient Boosted Decision Trees and conclude that there is still no universally superior solution.
	- For "shallow" state-of-the-art are ensembles such as GBMs. (see [[@gorishniyRevisitingDeepLearning2021]])
	- Deep learning for tabular data could potentially yield a higher performance and allow to combine tbular data with non-tabular data such as images, audio or other data that can be easily processed with deep learning. [[@gorishniyRevisitingDeepLearning2021]]
	- Despite growing number of novel (neural net) architectures, there is still no simple, yet reliable solution that achieves stable performance across many tasks. 
	- Show that there is a general concensus, that gradient boosted trees and neural networks work best. Show that there is a great bandwith of opinions and its most promising to try both. Papers: [[@shwartz-zivTabularDataDeep2021]]
	- [[@arikTabNetAttentiveInterpretable2020]] Discuss a number of reasons why decisiion tree esembles dominate neural networks for tabular data.
	- [[@huangTabTransformerTabularData2020]] argue that tree-based esnembles are the leading approach for tabular data. The base this on the prediction accuracy, the speed of training and the ability to interpret the models. However, they list sever limitations. As such they are not suitabl efor streaming data, multi-modality with tabular data e. g. additional image date and do not support semi-supervised learning by default.
- Choose neural network architectures, that are tailored towards tabular data.
- Challenges of learning of tabular data can be found in [[@borisovDeepNeuralNetworks2022]] e. g. both 
- Taxonomy of approaches can be found in [[@borisovDeepNeuralNetworks2022]] 
![[tabular-learning-architectures.png]]
- Nice formulation and overview of the dominance of GBT and deep learning is given in [[@levinTransferLearningDeep2022]]

- Sophisticated neural network architectures might not be required, but rather a mix of regularization approaches to regularize MLPs [[@kadraWelltunedSimpleNets2021]].

- semi-supervised learning with pre-training for tabular data improves feature transfer. Also possible if features differ between the upstream and downstream task. [[@levinTransferLearningDeep2022]] 
- reasons why deep learning on tabular data is challenging [[@shavittRegularizationLearningNetworks2018]] (use more as background citation)
- selection is hard e. g., in deep learning, as there are no universal benchmarks and robust, battle tested approaches for tabular data compared to other data sources. (see [[@gorishniyRevisitingDeepLearning2021]])

- There are several established libraries such as catboost, XGBoost and LightGBM, (that differ in e. g., the growing policy of trees, handling missing values or the calculation of gradients. (see papers also see [[@josseConsistencySupervisedLearning2020]]))  Their performance however, doesn't differ much. (found in [[@gorishniyRevisitingDeepLearning2021]] and cited [[@prokhorenkovaCatBoostUnbiasedBoosting2018]])
- See [[@huangTabTransformerTabularData2020]] that point out common problems of comparsions between gbts and dl.
- Why to formulate problem as probabilistic classification problem: https://www.youtube.com/watch?v=RXMu96RJj_s
- for use of transformer-based models in finance see [[@zouStockMarketPrediction2022]]