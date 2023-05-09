Previous studies on trade classification with machine learning made arbitrary selections of methods, failing to account for the latest advancements in the field (cp. cref-[[👪Related Work]]).  In this thesis, we perform a succinct discussion to select a set of supervised classifiers based on empirical evidence. To guide our discussion, we establish the following requirements a classifier must met:
-  *performance:* The approach must deliver state-of-the-art performance in tabular classification tasks. Trades are typically provided as tabular datasets, consisting of rows representing instances and columns representing features. The classifier must be well-suited for probabilistic classification on tabular data.
-  *scalability:* The approach must be able to scale to datasets with $>$ 10 Mio. samples. Due to the high trading activity and long data history, datasets may contain millions of samples, so classifiers must be able to handle large quantities of trades.
- *extensibility:* The approach must be extendable to train on partially-labelled trades. Most definitions of the trade initiator apply only to a subset of trades (e.g., certain order types), but excluded trades can still provide be valuable in training a classifier. The classifier must support training on unlabelled and labelled trades.

Tree-based ensembles have long dominated learning on tabular data, such as *gradient boosting* or *random forest* (cp.[[@grinsztajnWhyTreebasedModels2022]]). Tree-based ensembles combine the estimate of individual decision trees into an ensemble to obtain a more accurate prediction. Popular variants include random forests ([[@breimanRandomForests2001]]), which are built on the bagging principle and use deep trees on random subsets of data, and gradient-boosting ([[@friedmanStochasticGradientBoosting2002]]) sequentially combines small-sized trees, that improve upon the residual of the previous trees.

Despite the convincing performance of random forests in ([[@ronenMachineLearningTrade2022]]13--14), we do not consider *random forests*, due to our problem framing as a *probabilistic* classification task. Decision trees and by extension random forests, yield probability estimates. They are generally derived from few samples in the leaf nodes, making them unreliable (). Also, the procedure contradicts the tree construction, which aims the purity of splits. 

This problem is less severe non-existent for gradient-boosted decision trees, as... Since the splitting procedures can be efficiently approximated, gradient-boosting is highly scalable to large quantities of data. It can also be extended to incorporate unlabelled trades. This makes gradient-boosting a compelling choice for trade classification. 

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

These includes architectures like, *TabNet* ([[@arikTabnetAttentiveInterpretable2020]]), *TabTransformer* ([[@huangTabTransformerTabularData2020]]2--3), *SAINT* ([[@somepalliSaintImprovedNeural2021]]), *Non-Parametric Transformer* ([[@kossenSelfAttentionDatapointsGoing2021]]3--4)  and *FT-Transformer* ([[@gorishniyRevisitingDeepLearning2021]]4--5). *TabNet* ([[@arikTabnetAttentiveInterpretable2020]]3--5), fuses the concept of decision trees and attention. Similar to growing a decision tree, several sub-networks are used to process the input in a sequential, hierarchical fashion. Sequential attention, a variant of attention, is used to decide which features to select in each step. The output of *TabNet* is the aggregate of all sub-networks. Its poor performance in independent comparisons e.g., ([[@kadraWelltunedSimpleNets2021]]7) and ([[@gorishniyRevisitingDeepLearning2021]]7), raises doubts about its usefulness. *SAINT* uses a specialized attention mechanism, the *intersample attention*, which performs attention over both columns and rows ([[@somepalliSaintImprovedNeural2021]]4--5). Applied to our setting, the model would contextualize information from the trade itself, but also from neighbouring trades, which would be an unfair advantage over classical trade classification rules. Similarly, the *Non-Parametric Transformer* of ([[@kossenSelfAttentionDatapointsGoing2021]]3--4) uses the entire data set as a context, which rules out the application in our work. Differently, *TabTransformer* ([[@huangTabTransformerTabularData2020]]2--3) performs attention per sample on categorical features-only. All numerical features are processed in a separate stream, which breaks correlations between categorical and numerical features ([[@somepalliSaintImprovedNeural2021]]2). Most importantly though, most features in trade datasets are numerical. As such, trade classification would hardly profit from the Transformer architecture, causing the model to collapse to a vanilla gls-MLP. A more comprehensive approach is provided by ([[@gorishniyRevisitingDeepLearning2021]]4--5) in the form *FT-Transformer*, which is a Transformer-based architecture, that processes both numerical inputs and categorical input in the Transformer blocks. Since it achieved competitive performance in several empirical studies, we further consider FT-Transformer for our empirical study ([[@grinsztajnWhyTreebasedModels2022]]5). Being based on the Transformer architecture, FT-Transformer *naturally* scales to large amounts of data and can utilize unlabelled data through self-training procedures. 

To summarize, our study considers gradient boosting and the FT-Transformer, each trained on labelled or partially labelled trades. This comparison is particularly appealing, as it enables a multi-faceted comparison of wide tree based ensembles versus deep neural networks, as well as supervised versus semi-supervised methods.

#gbm #transformer #supervised-learning #deep-learning 

**Notes:**
[[🍪Selection of semisupervised Approaches notes]]