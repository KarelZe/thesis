
# Proposal

For a wide range of research questions in option markets, the side of the customer in a trade is of primary importance. Particularly, the trade direction is required to determine the information content of trades, the order imbalance and inventory accumulation of liquidity providers, the price impact of transactions, and to calculate many liquidity measures. Important examples are studies on option demand (Gârleanu, Pedersen, and Poteshman (2009); Muravyev and Ni (2020)), option order flow (Muravyev (2016)), and option price pressures (Goyenko and Zhang (2021)). Because most option datasets do not contain information on the side of a trade, empirical studies often rely on heuristics tested in stock markets to infer trade direction from prices and quotes. The most common of these classification rules are the tick test, the quote rule, the Lee and Ready (LR, 1991) algorithm, and the Ellis, Michaely, and O’Hara (EMO, 2000) rule. Notwithstanding their wide application, little is known on how stock trade classification rules perform in option markets.

establish an upper bound

- optionally we include Rosenthal

Based on a s

Both data sets are characterized as probablistic classification
Thus . The 

So far classical classification rules ... dominated 

neglect state-of-the-art algorithms and follow an unclear, 

The natural question is, can a predictor improve upon their results using the identical features? Approaching this concern with machine learning is a logical choice due to the ability to deal with high dimensional data 

The thesis follows the following structure:

In the introduction, we provide motivation and present our key findings. The contributions are four-fold:
1. We employ state-of-the-art machine learning algorithms i. e., gradient-boosted trees and transformer networks for trade classification. Both architectures haven't been applied before.
3. We study the impact on the trade classification accuracy of incorporating unlabeled trades into the training procedure. Hence, we compare supervised and semi-supervised model variants.  
3. We propose an approach based on kernel SHAP (...) to interpret classical trade classification rules and machine learning models consistently.
4. We add to the body of literature on trade classification in option markets

**Related Work**

While classical trade classification algorithms are tested in the stock markets e. g. (...) extensively, only few works exist, that evaluate trade classification rules in option markets. Notably, these include works (Savickas, Grauer, EMO, CLVN??).

Despite the general advent of machine learning, machine learning has hardly been applied to trade classification, none of which to option data. An early work of Rosenthal (...) incorporates classical trade classification rules into a logistic regression model and achieves outperformance in the stock market. Similarily, Ronen (....) and (....) improve upon classical rules with a random forest, a tree-based ensemble. Albeit their work considers a broad range of approaches, the selection leaves the latest advancements in artificial neural networks and ensemble learning aside. Even if the focus is on standard techniques, the unclear research agenda with regard to model selection, tuning and testing hampers the transferbility of their results to the option market. 

**Methodology**
We start by introducing the basic quote rule, the tick test, reverse tick test, depth rule and trade size rule and derive popular hybrids thereoff. Namely the LR-algorithm, the EMO algorithm and the CLVN method. We discuss deviations from the original algorithm like the offset in the LR-algorithms. Optionally, we include Rosenthal's method, [[@rosenthalModelingTradeDirection2012]] which incorporates the tick test, LR and EMO algorithm into a logistic regression model. Our focus is on the features used within the rules and their economic intuition. We also stress the link between hybrids like to ensembling techniques studied in machine learning. Classical trade classification rules serve as a benchmark ~~and optionally as a feature~~ in our study.

Data sets of options trades adhere to a tabular format.  Thus, machine learning-wise we begin with reviewing the state of the art for classification on tabular data with regard to accuracy. Possible models must support both categorical features e.g., exercise style of the option and continous features e. g. the option's $\Delta$. For option data set the true label i. e., indicator if trade is buyer-iniated, can only be inferred for fractions of the dataset ([[@grauerOptionTradeClassification2022]], [[@savickasInferringDirectionOption2003]]). Remaining trades are unlabelled. Leaving the unlabelled data aside, option trade classification is as a *supervised classification task*. Recent research ([[🧠Deep Learning Methods/Transformer/@arikTabNetAttentiveInterpretable2020]]) indicates, however, that leveraing unlabelled data can improve classifier performance and interpretability. Thus, we also frame the problem of trade classification in option markets as a *semi-supervised classification* task, where both unlabelled and labeled data is incorporated into the learning procedure. 

Our selection will likely consider *wide ensembles* in the form of gradient boosted trees and *deep, transformer-based neural networks*, such as TabNet [[@arikTabNetAttentiveInterpretable2020]] or *TabTransformer* [[@huangTabTransformerTabularData2020]]. Also, both model classes can naturally be enhanced to profit from partially-unlabelled data and are interpretable on a global and local level. 

Thereafter, we give a thorough introduction of the models for the supervised setting. We start with the notion of classical decision trees, as covered by [[@breimanRandomForests2001]] Decision trees are inherent to tree-based boosting approaches as weak learners. Thus, emphasis is put on the selection of features and the splitting process of the predictor space into disjoint regions. We motivate the use of ensemble approaches, such as gradient boosted trees, with the poor variance property of decision trees. The subsequent chapter draws on [[@hastietrevorElementsStatisticalLearning2009]] and [[@friedmanGreedyFunctionApproximation2001]]with a focus on gradient boosting for classification. Therein we introduce necessary enhancements to the boosting procedure to support probabilistic classification and discuss arising stability issues. Further adjustments are necessary for the treatment of categorical variables. Therefore, we draw on the *ordered boosting* by [[@prokhorenkovaCatBoostUnbiasedBoosting2018]], which enhances the classical gradient boosting algorithm.

Next we focus on transformer networks for tabular data. We start by introducing the classical transformer architecture of [[@vaswaniAttentionAllYou2017]]. We put our focus on introducing central concepts like the encoder-decoder structure, self-attention, embeddings or point-wise networks. These chapters lay the basis for the subsequent tabular-specific architectures like *TabNet* or *TabTransformer*, which follow thereafter. With a focus on sequence-to-sequence modelling the classic transformer is not directly applicable to tabular data.

The *TabTransformer* [[@huangTabTransformerTabularData2020]] utilizes the afore-mentioned transformers [[🧠Deep Learning Methods/Transformer/@vaswaniAttentionAllYou2017]] to learn contextual embeddings of categorical features, whereas continous features are directly input into a feed-forward network.

Another alternative is *Tabnet* ([[@arikTabNetAttentiveInterpretable2020]]), which fuses the concept of *decision trees* and *transformers*. Similar to growing a decision tree severals subnetworks are used to process the input in a sequential, hierarchical fashion. Sequential attention, a variant of attention, is used to decide which features to use in each step. The output of *tabnet* is the aggregate of all subnetworks like an ensemble. Despite its difference, concepts like the encoder or decoder or attention similar to the previous variants. 

Previous research (e. g., [[@arikTabNetAttentiveInterpretable2020]]) could show that both tree-based and neural-network-based approaches can profit from learning on additional, unlabelled data. Thus we demonstrate how the models from above can be enhanced for the semi-supervised setting. For gradient boosted trees, self-training [[@yarowskyUnsupervisedWordSense1995]]  is used to obtain pseudo labels for unlabeled parts of the data set. The ensemble itself is trained on both true and pseudo labels. For the neural networks the scope is limited to separate *pre-training procedures* to maintain consistency with the supervised counterparts. Thus, for *TabNet* we use unsupervised pretraining of the encoder as propagated in [[@arikTabNetAttentiveInterpretable2020]]. Equally, for the *TabTransformer* we pretrain the transformer layers and column embeddings through *masked language modeling* or *replaced token detection* as popularized in [[@devlinBERTPretrainingDeep2019]] and [[@clarkELECTRAPretrainingText2020]] respectively. 

**Empirical Study**

In our empirical analysis, we introduce the data sets, the generation of true labels and the applied pre-processing. The data sets contains option trades executed at either the ...(CBOE) or the ...(ISE) with additional intraday option price and quote data, end-of-day buy and sell trading volumes and characteristics of the option and its underlying. Yet our primary focus is on the classification of ISE trades, with secondary focus on the CBOE data set. 

Subsets of the CBOE and the ISE data set have been previously studied in [[@grauerOptionTradeClassification2022]]. Thus we align the data pre-processing with their work to maintain consistency. Despite that, some deviations are necessary for training the machine learning models. This includes the imputation of missing features, standardization, resampling, feature transformations, and feature subset selection. While all our models can theoretically handle raw tabular data without prior processing (Tabnet, Catboost etc.), we expect to improve the model's performance with theses additional steps. We derive features through feature transformations e. g., relative distance of the trade from the mid point found in the CLVN method to incorporate them into our models while not incorporating the rule directly. Doing so, provides insights on the relation of classical and machine learning based approaches. ~~A positional encoding is applied on temporal data.~~  Similar to [[👨‍👩‍👧‍👦Related Works/@ronenMachineLearningTrade2022]] we define different subsets of data i. e., one that includes only features found in the classical algorithms and another one incorporating option characteristics as well as price and trading data. Finally we do include unlabelled data for the training of semi-supervised models.

The data set is split into three disjoint sets for training, validation and testing. Similar to [[@ellisAccuracyTradeClassification2000]] and [[@ronenMachineLearningTrade2022]] we perform a classical train-test split, thereby maintaining the temporal ordering within the data. To assess the performance of trade classification rules we rely only on the true label of the trade initiator.  With statistical tests ~~e. g., adversarial validation~~ we verify that the distribution of the features and target is maintained on the test set. Due to the sheer number of model combinations considered and the computational demand of transformers and gradient boosted trees, we expect $k$-fold cross validation to be technically infeasable.

Next, we describe the implementation and training of the supervised, semi-supervised models and classical trade classification rules. 
For a consistent evaluation we opt to implement classical rules like the LR algorithm as a classifier conforming to the *Scikit learn* (...) API.
Gradient boosting is implemented using *CatBoost* by Par(...). The implementation of *TabNet* and *TabTransformer* is done in *PyTorch* based on the original papers. Deviations from the papers are reported.
For training we employ various model-agnostic deep learning practices like learning rate decay, drop out (...), early stopping, ensembling [[@huangSnapshotEnsemblesTrain2017]] or stochastic weight averaging (...) to speed up training or obtain better generalization. To test for the later, we report the loss curves to detect over- or underfitting. For unbiased estimates on the bias and variance properties of our model we discuss and report the learning curves.  

In contrast to Ronen (...) we emphasize a transparent hyperparameter tuning procedure. We tune using a novel Bayesian optimization based on the tree-structured parzen estimator algorithm. Compared to other approaches like a randomized search, unpromising search regions are omitted, thereby requiring fewer search trails. Bayesian search is reported to be superior over randomized search [[@turnerBayesianOptimizationSuperior2021]]. The search space for the parameters is based on the configurations in the corresponding papers. We use an implementation by [[@akibaOptunaNextgenerationHyperparameter2019]] for optimizing for the accuracy on the validation set. 

We report the optimisation metric on both the training, validation and test set to study the impact of different learning schemes and learning of generalisable features. Visualization-wise, the chapter may include a study of loss surfaces. The expectation is that pre-training improves both the training and validation loss, due to the larger sample size seen during training. A decline between the sets may be observed.

Subsequently, the model is evaluated. Firstly, a comparsion between the selected features is conducted. TabNet (...), TabTransformer, and gradient boosted trees (...) are interpretable by design, but rely on model specific techniques such as feature activation masks found only in transformer-based models rendering them useless for cross-model comparsions. We rely on acivation masks to study trades on the transaction level. ~~on an instance or global basis.~~ To compare *all* models we suggest kernel SHAP (...) or random feature permutation by Breiman (...) for local and global interpretability. Due to the proposed implemention of the classical rules as an estimator, we can perform a fair comparsion between classical and machine learning-based approaches. Our focus is to back the observed results with economic intuition.

Secondly, we benchmark TabNet, TabTransformer and gradient boosted trees against the classical, trade classification rules from above. Following a common track in literature, the decisive metric is the accuracy. ~~If the data is highly imbalanced, we replace the accuracy with the $F_1$-score.~~ We may back our analysis with additional metrics like the Receiver Operator characteristic, AUC curves or confusion matrices and report standard errors. We expect both semi-supervised and supervised algorithms to outperform the benchmarks with additional performance gains from learning on unlabelled data.

Based on preliminary tests, (...)

Despite serious counter efforts our models can still overfit the data. We use rigorous robustness checks to test whether the accuracy is maintained across time, trade sizes, underlyings and exchanges, among others. The procedure follows Grauer, Sacvickas...(...).

All in all, our empirical analysis aims for reproducability. As such, we implement sophisticated data set versioning and experiment tracking using *weights & biases*. The correctness of the code is verified with automated tests. 

**Discussion and Conclusion** 

A discussion and a conclusion follow the presentation of the results.


