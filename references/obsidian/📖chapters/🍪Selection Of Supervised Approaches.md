1000 wörter

Previous studies on trade classification with machine learning made arbitrary selections of methods, failing to account for the latest advancements in the field (cp. cref-[[👪Related Work]]).  In this thesis, we perform a succinct discussion to select a set of supervised classifiers based on empirical evidence. To guide our discussion, we establish the following requirements a classifier must simulatenously meet:
-  *performance:* The approach must deliver state-of-the-art performance in tabular classification tasks. Trades are typically provided as tabular datasets, consisting of rows representing instances and columns representing features. The classifier must be suited for probabilistic classification on tabular data.
-  *scalability:* The approach must be able to scale to datasets with $>$ 10 Mio. samples. Due to the high trading activity and long data history, datasets may contain millions of samples, so classifiers must be able to handle large quantities of trades.
- *extensibility:* The approach must be extendable to train on partially-labelled trades. Most definitions of the trade initiator apply only to a subset of trades (e.g., certain order types), but excluded trades can still provide be valuable in training a classifier. The classifier must support training on unlabelled and labelled trades.

Trade classification, as we framed it, fits into supervised learning on tabular data, which is comprehensively covered by the research community with several studies reviewing and benchmarking newly proposed approaches against established machine learning methods.

**Shallow Tree-Based Ensembles**
Traditionally, tree-based ensembles, in particular gls-gbrt, have dominated modelling on tabular data with regard to predictive performance ([[@grinsztajnWhyTreebasedModels2022]]) and ([[@kadraWelltunedSimpleNets2021]]) and ([[@borisovDeepNeuralNetworks2022]]14). At its core, tree-based ensembles combine the estimates of individual decision trees into an ensemble to obtain a more accurate prediction. For gls-gbrt ([[@friedmanStochasticGradientBoosting2002]]) the ensemble is constructed by sequentially adding small-sized trees into the ensemble that improve upon the error of the previous trees.  Closely related to gradient-boosted trees are random forests (breiman / shappire paper?). Random forests ([[@breimanRandomForests2001]]) fuse decision trees with the bagging principle by growing deep decision trees on random subsets of data and aggregate the individual estimates. ([[@ronenMachineLearningTrade2022]]13--14) have unparalleled success in classifying trades through random forests. Due to the framing as a *probabilistic* classification task, random forests are not optimal. This is because decision trees yield poorly calibrated probability estimates, which carry into the ensemble ([[@tanhaSemisupervisedSelftrainingDecision2017]]356--360). Gradient boosting is unaffected by this problem, scales to large data set due to the availability of highly optimized implementations that approximate the construction of ensemble members, and is extensible to learn on unlabelled and labelled instances simultaneously. The state-of-the art performance in tabular classification tasks, together with its ability to scale and extend, makes it a candidate for trade classification.

**Deep Neural Networks**
The dominance of of gls-gbrt for tabular modelling is challenged by neural networks. In tabular deep learning one can identify three main lines of research:, regularized networks, differentiable learners and attention-based networks.

*Regularized Networks*
([[@kadraWelltunedSimpleNets2021]]9--10) advocate for the use of vanilla gls-ffn with an extensive mix of regularization techniques, such as dropout ([[@srivastavaDropoutSimpleWay]]), residual connections ([[@heDeepResidualLearning2015]]) or weight decay, over complex, tabular-specific deep learning architectures or gradient-boosted trees. Regularization is expected to enhance generalization performance, but the benefit is non-exclusive to gls-ffn. Conversely, when regularization is equally applied to tabular specific architectures, the effect reverses and several works including ([[@gorishniyRevisitingDeepLearning2021]]7 or [[@grinsztajnWhyTreebasedModels2022]]5) show that regularized gls-ffn actually trail the performance of specialized architectures. As these results violates our performance requirement, we focus on specialized architectures, in particular, differentiable learners and attention-based networks, but put emphasis on a careful regularization and optimization.

corrobates

*Differentiable Trees*


corroborates this

Tabluar specific deep learning architectures


-> NODE poor performance? (in [[@gorishniyRevisitingDeepLearning2021]]) fairly reasonable / poor in ([[@kadraWelltunedSimpleNets2021]]) / ([[@borisovDeepNeuralNetworks2022]])
Differentiable trees take inspiration from tree-based learners. 

A class of of networks that takes inspiration from tree-based learners are differentiable trees.
to warrants




“To our knowledge, this is the first empirical investigation of why tree-based models outperform neural networks on tabular data. Some speculative explanations, however, have been offered [Klambauer et al., 2017, Borisov et al., 2021]. Kadra et al. [2021a] claims that searching across 13 regularization techniques for MLPs to find a dataset-specific combination gives state-of-the-art performances.” (Grinsztajn et al., 2022, p. 3)
Recently, deep learning approaches excel the performance of gradient-boosted trees through sophisticated architectures. 


Claim that

While regularization expectedly improve

By the virtue

take inspiration from 

Differentiable trees try to adapt the success of 

These architectures are often based 

Networks, 



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

A successful line of research featThese includes architectures like, *TabNet* ([[@arikTabnetAttentiveInterpretable2020]]), *TabTransformer* ([[@huangTabTransformerTabularData2020]]2--3), *SAINT* ([[@somepalliSaintImprovedNeural2021]]), *Non-Parametric Transformer* ([[@kossenSelfAttentionDatapointsGoing2021]]3--4)  and *FT-Transformer* ([[@gorishniyRevisitingDeepLearning2021]]4--5). *TabNet* ([[@arikTabnetAttentiveInterpretable2020]]3--5), fuses the concept of decision trees and attention. Similar to growing a decision tree, several sub-networks are used to process the input in a sequential, hierarchical fashion. Sequential attention, a variant of attention, is used to decide which features to select in each step. The output of *TabNet* is the aggregate of all sub-networks. Its poor performance in independent comparisons e.g., ([[@kadraWelltunedSimpleNets2021]]7) and ([[@gorishniyRevisitingDeepLearning2021]]7), raises doubts about its usefulness. *SAINT* uses a specialized attention mechanism, the *intersample attention*, which performs attention over both columns and rows ([[@somepalliSaintImprovedNeural2021]]4--5). Applied to our setting, the model would contextualize information from the trade itself, but also from neighbouring trades, which would be an unfair advantage over classical trade classification rules. Similarly, the *Non-Parametric Transformer* of ([[@kossenSelfAttentionDatapointsGoing2021]]3--4) uses the entire data set as a context, which rules out the application in our work. Differently, *TabTransformer* ([[@huangTabTransformerTabularData2020]]2--3) performs attention per sample on categorical features-only. All numerical features are processed in a separate stream, which breaks correlations between categorical and numerical features ([[@somepalliSaintImprovedNeural2021]]2). Most importantly though, most features in trade datasets are numerical. As such, trade classification would hardly profit from the Transformer architecture, causing the model to collapse to a vanilla gls-MLP. A more comprehensive approach is provided by ([[@gorishniyRevisitingDeepLearning2021]]4--5) in the form *FT-Transformer*, which is a Transformer-based architecture, that processes both numerical inputs and categorical input in the Transformer blocks. Since it achieved competitive performance in several empirical studies, we further consider FT-Transformer for our empirical study ([[@grinsztajnWhyTreebasedModels2022]]5). Being based on the Transformer architecture, FT-Transformer *naturally* scales to large amounts of data and can utilize unlabelled data through self-training procedures. 

The findings ([[@ronenMachineLearningTrade2022]]50) do not support the use of neural networks in trade classification. But due to the lack of details regarding the model architecture, regularization techniques, and training insights, it is necessary to reevaluate these findings in the context of option markets.
To summarize, our study considers gradient boosting and the FT-Transformer, each trained on labelled or partially labelled trades. This comparison is particularly appealing, as it enables a multi-faceted comparison of wide tree based ensembles versus deep neural networks, as well as supervised versus semi-supervised methods.

#gbm #transformer #supervised-learning #deep-learning 

**Notes:**
[[🍪Selection of semisupervised Approaches notes]]