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