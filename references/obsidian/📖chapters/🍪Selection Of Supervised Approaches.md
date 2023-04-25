The selection of methods in previous works is arbitrary and guided computational constraints (cp. cref-[[👪Related Work]]). Furthermore, the selection leaves out advancements in machine learning on tabular data.
This paper provides a succinct discussion of these gaps and selects a set of supervised classifiers based on empirical evidence. 

We impose the following requirements a classifier must fulfill:
*performance:* The approach must deliver state-of-the-art performance in tabular classification tasks. Typically, trades are provided as tabular datasets, that consist of rows and columns / vectors of heterogeneous features (What is in row? 🚧). Individual features may be both numerical or categorical. The classifier must be well-suited for classification on tabular data.
*scalability:* The approach must scale to datasets with $>$ 10 Mio. samples. Due to the high trading activity and the long data history, datasets can contain many samples. Classifiers must scale to large to large data volumes.
*extensibility:* The approach must be extendable to train on partially-labelled trades. Most definitions of the trade initiator apply only to a subset of trades e.g., certain order types. The so-excluded trades can still be valuable in training a classifier. The classifier, however, must support it.

Learning on tabular data has long been dominated by tree-based ensembles, in particular *gradient boosting* (cp. ). Tree-based ensemble combine individual decision trees into an ensemble to obtain an improved ensemble estimate. Popular variants include random forests ([[@breimanRandomForests2001]]), an extension to bagging, and gradient-boosting ([[@friedmanStochasticGradientBoosting2002]]), a variant of *boosting*. 

Despite the good performance of random forests in ([[@ronenMachineLearningTrade2022]]13--14), we do not consider *random forests*, which is a direct consequence of our problem framing as a *probabilistic* classification task. Decision trees and by extension random forests,yield probability estimates. They are generally derived from few samples in the leaf nodes, making them unreliable (). Also, the procedure contradicts the tree construction, which aims the purity of splits. This problem is less severe for gradient-boosted decision trees. As splitting procedures can be efficiently approximated, gradient-boosting is highly scalable to large quantities of data. It can also be extended to incorporate unlabelled trades. This makes gradient-boosting a compelling choice for trade classification. 

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

These includes architectures like, *TabNet* ([[@arikTabNetAttentiveInterpretable2020]]), *TabTransformer* ([[@huangTabTransformerTabularData2020]]), *SAINT* ([[@somepalliSAINTImprovedNeural2021]]), and *FT-Transformer* ([[@gorishniyRevisitingDeepLearning2021]]). TabNet* [[@arikTabNetAttentiveInterpretable2020]]}, fuses the concept of decision trees and attention. Similar to growing a decision tree, several subnetworks are used to process the input in a sequential, hierarchical fashion. Sequential attention, a variant of attention, is used to decide which features to use in each step. The output of *TabNet* is the aggregate of all sub-networks. Its poor performance in independent comparsion e. g., ([[@kadraWelltunedSimpleNets2021]]7) and ([[@gorishniyRevisitingDeepLearning2021]]7), raises concerns.  *SAINT* uses a specialized attention mechanism, the *intersample attention*, which performs attention over both columns and rows ([[@somepalliSAINTImprovedNeural2021]]4--5). Applied to our setting, the model could contextualise information from the trade itself, but also from neighbouring trades, which would be an unfair advantage over classical trade classification rules. Differently, *TabTransformer* ([[@huangTabTransformerTabularData2020]]2--3) performs attention per sample on categorical features-only. All numerical features are processed in a separate stream, which breaks correlations between categorical and numerical features ([[@somepalliSAINTImprovedNeural2021]]2). Most importantly though, most features in trade datasets are numerical. As such, trade classification would hardly profit from the Transformer architecture, causing the model to collapse to a vanilla gls-MLP. A more comprehensive approach is provided by ([[@gorishniyRevisitingDeepLearning2021]]4--5) in the form *FT-Transformer*, which is a Transformer-based architecture, that processes both numerical inputs and categorical input in the Transformer blocks. Since it achieved competitive performance in several empirical studies, we further consider FT-Transformer for our empirical study ([[@grinsztajnWhyTreebasedModels2022]]5). Being based on the Transformer architecture, FT-Transformer is *naturally* scales to large amounts of data and can utilize unlabelled data through self-training procedures.

To summarize, our study considers gradient boosting and the FT-Transformer, each trained on labelled or partially labelled trades. This comparison is particularly appealing, as it enables a multi-faceted comparison of wide tree based ensembles versus deep neural networks, as well as supervised versus semi-supervised methods.

#gbm #transformer #supervised-learning #deep-learning 

**Notes:**
[[🍪Selection of semisupervised Approaches notes]]