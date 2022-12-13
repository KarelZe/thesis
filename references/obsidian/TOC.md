- [nature-summary-paragraph.pdf](https://www.nature.com/documents/nature-summary-paragraph.pdf)
- Guide on visualizations https://www.nature.com/articles/s41467-020-19160-7
- Guide on storytelling with data https://www.practicedataviz.com/pdv-evd-mvp#PDV-EVD-mvp-g
- see  `writing-a-good-introduction.pdf`

# Title
- Off being right: trade-side classification of options data with machine learning
- Trade side classifcation with machine learning: do or don't?
- Improving trade site classication with machine learning.
- Getting it right: trade side classifaction of options using machine learning.
- Getting it right: Improving trade side classification with gradient boosted trees...
- Done right: ...
- Do or do not, there is no try
- Be less wrong. Improving trade site classification with machine learning
- More than a nudge. Improving options trade site classification with machine learning
- Limit to view, yet theoretically promising techniques as derived in [[#^d8f019]]

and its consequences are an important, but understudied, cause for concern.

Commonly stock trade classifcation algorithms are used




# Introduction

- "The validity of many ecomnomic studies hinges on the ability to accuractely classify trades as buyer or seller-initiated." (found in [[@odders-whiteOccurrenceConsequencesInaccurate2000]])
- trade site classification matters for several reasons, market liqudity measures, short sells, study of bid-ask-spreads.
- Where is trade side classification applied? Why is it important? Do citation search.
- Repeat in short the motivation
- Outpeformance in similar / other domains
- Obtain probabilities for further analysis

- Crisp sentence of what ML is and why it is promising here. 

- goal is to outperform existing classical approaches

- [[@rosenthalModelingTradeDirection2012]] lists fields where trade classification is used and what the impact of wrongly classified trades is.
- The extent to which inaccurate trade classification biases empirical research dependes on whether misclassifications occur randomly or systematically [[@theissenTestAccuracyLee2000]].
- There is no common sense of who is the iniator of a trade. See discussion in [[@odders-whiteOccurrenceConsequencesInaccurate2000]]
- over time proposed methods applied more filters / got more sophisticated but didn't substainly improve im some cases. See e. g., [[@finucaneDirectTestMethods2000]] Time to switch to another paradigma and let the data speak?
- Works that require trade side classification in option markets:
	- [[@muravyevOrderFlowExpected2016]]
	- [[@huDoesOptionTrading2014]]

# 👨‍👩‍👧‍👦 Related Work
- [[@grauerOptionTradeClassification2022]] perform option trade classification
- [[@savickasInferringDirectionOption2003]] perform option trade classification
- [[@ronenMachineLearningTrade2022]] / [[@fedeniaMachineLearningCorporate2021]] They employ a machine learning-based approach for trade side classification. Selection of method follows no clear research agenda, so does sample selection or tuning. Also leaves out latest advancements in prediction of tabular data such as GBM or dedicated NN architectures. Data set only spans two days? General saying ML based predictor (random forest) outperforms tick rule and BVC. Still much human inutition is required for feature engineering. Treated as **supervised tasks**. More recent approaches and also ML approaches outperform classical approaches due to a higher trading frequency. Transfer learning not successful. **Note:** Tick rule has been among the poorest predictors in Grauer. **Note:** Check what the actual difference between the two papers are....
- Which works performed trade side classification for stocks, for options or other products.
- [[@rosenthalModelingTradeDirection2012]] incorporates different methods into a model for the likelihood a trade was buyer-initiated. It's a simple logistic regresssion. Performed on stocks. 
- [[@blazejewskiLocalNonparametricModel2005]] compare $k$-nn and logistic regression for trade-side classification. Performed for Australian stocks. Unclear how results compare to classical rules. 
- Similarily, [[@aitkenIntradayAnalysisProbability1995]] perform trade side classification with logistic regression.
1. Broader term is **trade site classification** = assign the side to a to a transaction and differentiate between buyer- and seller-initiated transactions
2. It's also sometimes called trade sign classification
- There is no single definition / understanding for the one who initiates trades. [[@olbrysEvaluatingTradeSide2018]] distinguish / discuss immediacy and initiator.
- Do not compare accuracies across different datasets. This won't work. Might mention [[@grauerOptionTradeClassification2022]] as it is calculated on (partly) the same data set.

- Results were very different for the option markets between the studies. Compare the frequency some literature (in the stock market) suggest, that  for higher frequencies classical approaches like the tick test deteriorate.
> Easley, O’Hara, and Srinivas (1998) use the Lee and Ready approach to test their game theoretic model of informed trading in stock and option markets. It is, therefore, important to determine whether the application of stock trade classification rules to derivatives is valid. [[@savickasInferringDirectionOption2003]]

- For classical rule-based approaches see some citations in [[@olbrysEvaluatingTradeSide2018]]. E. g., [[@chakrabartyTradeClassificationAlgorithms2012]] or [[@chakrabartyEvaluatingTradeClassification2015 1]]

# 🔗Rule-Based Approaches

The following section introduces common rules for signing option trades. We start by introducing the prevailing quote and tick rule and continue with the recently introduced depth and trade size rule. In section (...) we combine hybrids thereoff. We draw a connection to ensemble learning.

## Basic Rules


- See [Quantitative Finance Stack Exchange](https://quant.stackexchange.com/questions/8843/what-are-modern-algorithms-for-trade-classification) for most basic overview
- There are different views of what is considered as buyer / seller iniated i. e. [[@odders-whiteOccurrenceConsequencesInaccurate2000]] vs. [[@ellisAccuracyTradeClassification2000]]
(see [[@theissenTestAccuracyLee2000]] for more details)
- Different views on iniatiting party (i. e. [[@odders-whiteOccurrenceConsequencesInaccurate2000]] vs. [[@chakrabartyTradeClassificationAlgorithms2012]]) (see [[@aktasTradeClassificationAccuracy2014]] for more details)
- Submitters of market orders are called liquidity demanders, while submitters of limit orders stored stored in the book are liquidity providers.
- The BVC paper ([[@easleyDiscerningInformationTrade2016]]) treats trade classification as a probabilistic trade classificatio nproblem. Incorporate this idea into the classical rules section.
- BVC is illsuited for my task, as we require to sign each trade? (see [[@panayidesComparingTradeFlow2014]])
- Algorithms like LR, tick rule etc. are also available in bulked versions. See e. g., [[@chakrabartyEvaluatingTradeClassification2015 1]] for a comparsion in the stock market. 
### Quote-Rule
- The quote rule classifies a trade as buyer initiated if the trade price is above the midpoint of the buy and ask as buys and if it is below as seller-iniated. Can not classify at the midpoint of the quoted spread. (see e.g., [[@leeInferringTradeDirection1991]] or [[@finucaneDirectTestMethods2000]])
- See [[@hasbrouckTradesQuotesInventories1988]] . Might be one of the first to mention the quote rule. It is however not very clearly defined. Paper also discusses some (handwavy) approaches to treat midpoint transactions.
- ![[formula-quote-rule 1.png]]
	Adapted from [[@olbrysEvaluatingTradeSide2018]]. Rewrite to formula
- ![[quote-rule-alternative 1.png]]
(copied from [[@carrionTradeSigningFast2020]])

### Tick Test
- Tick tests use changes in trade prices and look at previous trade prices to infer trade direction. If the trade occurs at a higher price, hence uptick, as the previous trade its classified as as buyer-initiated. If the trade occurs at a lower price its seller-iniated. If the price change is zero, the last price is taken, that is different from the current price. (see e. g., [[@grauerOptionTradeClassification2022]] or [[@finucaneDirectTestMethods2000]] or [[@leeInferringTradeDirection1991]] for similar framing)
- Consider  for citation [[@leeInferringTradeDirection1991]] .
- One of the first works who mention the tick test is [[@holthausenEffectLargeBlock1987]] (referred to as tick classification rule) or [[@hasbrouckTradesQuotesInventories1988]] (referred to as transaction rule)
- ![[formula-tick-rule 1.png]]
	Adapted from [[@olbrysEvaluatingTradeSide2018]]
	![[tick-rule-formulae-alternative 1.png]]
Copied from [[@carrionTradeSigningFast2020]]
- Sources of error in the tick test, when quotes change.
- ![[missclassification-trade-rule 1.png]] [[@finucaneDirectTestMethods2000]]
- low data requirements, as only transaction data is needed. (see [[@theissenTestAccuracyLee2000]]) Could be good enough though.
- Tick test can not handle if two trades do not involve market orders e. g. two limit orders. In such a case the tick rule could be applied, but there is ambiguous. (see [[@finucaneDirectTestMethods2000]])
- [[@perlinPerformanceTickTest2014]] proved that the tick test performs better than random chance. 


The tick rule is very data efficient, as only transaction data ... However, in option markets

**Reverse Tick Test**
A variant of the tick test is the reverse tick test as popularized by [[@leeInferringTradeDirection1991]]. Instead of using the previous distinguishable trade price, the subsequent trade price, that is different from the current trade is used. 

- Instead of the previous trade, the reverse tick rule uses the subsequent trade price to classify the current trade. 
- If the next trade price that is differnet from the current price, is below the current price the trade (on a down tick or zero down tick) is classified as buyer-initiated. If the next distinguishable price is above the current price (up tick or zero up tick), the current price the trade is seller-initiated. (loosely adapted from [[@grauerOptionTradeClassification2022]]) (see also [[@leeInferringTradeDirection1991]])

### Depth Rule 🟢

[[@grauerOptionTradeClassification2022]] promote using trade size information to improve the classification performance of midspread trades. In their >>depth rule<<, they infer the trade initiator from the bid's depth and ask for quotes. Based on the observation that an exceeding bid or ask quote relates to higher liquidity on one side, trades are classified as buyer-iniated for a larger ask size and seller-iniated for a higher bid size.

As shown in Algorithm 2, the depth rule classifies midspread trades only, if the ask size is different from the bid size, as the ratio between the ask and bid size is the sole criterion for assigning the initiator. To sign the remaining trades, other rules must follow thereafter.

In a similar vain the subsequent >>trade size rule<< utilizes the ask and bid quote size to improve classification performance.

- classify midspread trades as buyer-initiated, if the ask size exceeds the bid size, and as seller-initiated, if the bid size is higher than the ask size (see [[@grauerOptionTradeClassification2022]])
- **Intuition:** trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely. (see [[@grauerOptionTradeClassification2022]])
- Alternative to handle midspread trades, that can not be classified using the quote rule.
- Improves LR algorithm by 0.8 %. Overall accuracy 75 %.
- Performance exceeds that of the LR algorithm, thus the authors assume that the depth rule outperforms the tick test and the reverse tick test, that are used in the LR algorithm for for classifying midspread trades.


### Trade Size Rule
Motivated by the deminishing performance of the classical algorithms (such as the previously introduced tick test and quote rule) for option trades, where the trade size matches the bid or ask size, [[@grauerOptionTradeClassification2022]] propose to 

Due to the restrictions on the trade size, this rule needs to be combined with other rules.

- classify trades for which the trade size is equal to the quoted bid size as customer buys and those with a trade size equal to the ask size as customer sells. (see [[@grauerOptionTradeClassification2022]])
- **Intuition:** trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely. (see [[@grauerOptionTradeClassification2022]])  
- Accuracy of 79.92 % on the 22.3 % of the trades that could classified, not all!. (see [[@grauerOptionTradeClassification2022]])
- Couple with other algorithms if trade sizes and quote sizes do not match / or if the trade size matches both the bid and ask size. For other 
- Requires other rules, similar to the quote rule, as only a small proportion can be matched.
- tested on option data / similar data set
## Hybrid Rules

^ce4ff0
The previous trade classification rules are applicable to certain trades or come with their own drawbacks. To mitigate 
these,... through ensembling

Naturally, 

- use the problems of the single tick test to motivate extended rules like EMO.
- that lead to a fine-grained  fragmentation 
- What are common extensions? How do new algorithms extend the classical ones? What is the intuition? How do they perform? How do the extensions relate? Why do they fail? In which cases do they fail?
- [[@savickasInferringDirectionOption2003]]
- [[@grauerOptionTradeClassification2022]]
- Which do I want to cover? What are their theoretical properties?
- What are common observations or reasons why authors suggested extensions? How do they integrate to the previous approaches? Could this be visualised for a streamlined overview / discussion. 
- What do we find, if we compare the rules 

**Interesting observations:**
![[visualization-of-quote-and-tick 1.png]]
(image copied from [[@poppeSensitivityVPINChoice2016]]) 
- Interestingly, researchers gradually segment the decision surface starting with quote and tick rule, continuing with LR, EMO and CLNV. This is very similar to what is done in a decision tree. Could be used to motivate decision trees.
- All the hybrid methods could be considered as an ensemble with some sophisticated weighting scheme (look up the correct term) -> In recommender the hybrid recommender is called switching.
- Current hybrid approaches use stacking ([[@grauerOptionTradeClassification2022]] p. 11). Also, due to technical limitations. Why not try out the majority vote/voting classifier with a final estimator? Show how this relates to ML.
- In stock markets applying those filters i. e. going from tick and quote rule did not always improve classification accuracies. The work of [[@finucaneDirectTestMethods2000]] raises critique about it in the stock market.
![[pseudocode-of-algorithms 1.png]]
(found in [[@jurkatisInferringTradeDirections2022]]). Overly complex description but helpful for implementation?
### Lee and Ready Algorithm

^370c50
- According to [[@bessembinderIssuesAssessingTrade2003]] the most widley used algorithm to categorize trades as buyer or seller-initiated.
- Accuracy has been tested in [[@odders-whiteOccurrenceConsequencesInaccurate2000]], [[@finucaneDirectTestMethods2000 1]] and [[@leeInferringInvestorBehavior2000]] on TORQ data set which contains the true label. (see [[@bessembinderIssuesAssessingTrade2003]])
- combination of quote and tick rule. Use tick rule to classify trades at midpoint and use the quote rule else where

- LR algorithm
![[lr-algorithm-formulae 1.png]]
- in the original paper the offset between transaction prices and quotes is set to 5 sec [[@leeInferringTradeDirection1991]]. Subsequent research like [[@bessembinderIssuesAssessingTrade2003]] drop the adjustment. Researchers like [[@carrionTradeSigningFast2020]] perform robustness checks with different, subsequent delays in the robustness checks.
- See [[@carrionTradeSigningFast2020]] for comparsions in the stock market at different frequencies. The higher the frequency, the better the performance of LR. Similar paper for stock market [[@easleyFlowToxicityLiquidity2012]]
- Also five second delay isn't universal and not even stated so in the paper. See the following comment from [[@rosenthalModelingTradeDirection2012]]
>Many studies note that trades are published with non-ignorable delays. Lee and Ready (1991) first suggested a five-second delay (now commonly used) for 1988 data, two seconds for 1987 data, and “a different delay . . . for other time periods”. Ellis et al. (2000) note (Section IV.C) that quotes are updated almost immediately while trades are published with delay2. Therefore, determining the quote prevailing at trade time requires finding quotes preceding the trade by some (unknown) delay. Important sources of this delay include time to notify traders of their executions, time to update quotes, and time to publish the executions. For example, an aggressive buy order may trade against sell orders and change the inventory (and quotes) available at one or more prices. Notice is then sent to the buyer and sellers; quotes are updated; and, the trade is made public. This final publishing timestamp is what researchers see in nonproprietary transaction databases. Erlang’s (1909) study of information delays forms the theory for modeling delays. Bessembinder (2003) and Vergote (2005) are probably the best prior studies on delays between trades and quotes.
- For short discussion timing offset in CBOE data see [[@easleyOptionVolumeStock1998]] . For reasoning behind offset (e. g., why it makes senses / is necessary) see [[@bessembinderIssuesAssessingTrade2003]], who study the offset for the NASDAQ. Their conclusion is, that there is no universal optimal offset.
- for LR on CBOE data set see [[@easleyOptionVolumeStock1998]]
- LR can not handle the simultanous arrival of market buy and sell orders. Thus, one side will always be wrongly classified. Equally, crossed limit orders are not handled correctly as both sides iniate a trade independent of each other (see [[@finucaneDirectTestMethods2000]]). 
- LR is also available in bulked 
**Reverse LR algorithm:**
- first introduced in [[@grauerOptionTradeClassification2022]] (p 12)
- combines the quote and reverse tick rule
- performs fairly well for options as shown in [[@grauerOptionTradeClassification2022]]

### Ellis-Michaely-O’Hara Rule
- combination of quote rule and tick rule. Use tick rule to classify all trades except trades at hte ask and bid at which points the quote rule is applied. A trade is classified as  abuy (sell) if it is executed at the ask (bid).
- turns the principle of LR up-side-down: apply the tick rule to all trades except those at the best bid and ask.
- EMO Rule
![[emo-rule-formulae 1.png]]
- classify trades by the quote rule first and then tick rule
- Based on the observation that trades inside the quotes are poorly classified. Proposed algorithm can improve
- They perform logistic regression to determin that e. g. , trade size, firm size etc. determines the proablity of correct classification most
- cite from [[@ellisAccuracyTradeClassification2000]]

The tick rule can be exchanged for the reverse tick rule, as previously studied in [[@grauerOptionTradeClassification2022]].

### Chakrabarty-Li-Nguyen-Van-Ness Method
CLNV-Method is a hybrid of tick and quote rules when transactions prices are closer to the ask and bid, and the the tick rule when transaction prices are closer to the midpoint [[@chakrabartyTradeClassificationAlgorithms2007]]
- show that CLNV, was invented after the ER and EMO. Thus the improvement, comes from a higher segmented decision surface. (also see graphics [[visualization-of-quote-and-tick 1.png]])

![[clnv-method-visualization 1.png]]
(image copied from [[@chakrabartyTradeClassificationAlgorithms2007]])
- replace with clear formula
- for success rate and motivation see  [[@chakrabartyTradeClassificationAlgorithms2007]]

### Rosenthal's Rule
- see [[@rosenthalModelingTradeDirection2012]]
- Seldomly used but ML-like. Would probably be sufficient to cover it under related works.


# 🧠 Supervised Approaches
- Introduce a classifcation that differentiates between supervised, unsupervised, reenforcement learning and semi-supervised learning. 
- Introduce the concept of classification as a variant of supervised learning. 
- Could be supervised if all labels are known
- Could be semi-supervised if only some of the labels are known. Cover later.
- Use probabilistic classifier to allow for more in-depth analysis. Similar to BVC paper. Also explain why probabilistic clf makes sense -> Opens up new chances for evaluations and extensions. But comes with its own problems (see e. g., decision trees)
- Search for paper that performed a comparsion between Gradient Boosted Trees and Neural Net on large set of data sets....

## Selection of Approaches

^d8f019

See also https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html

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
![[tabular-learning-architectures 1.png]]
- Nice formulation and overview of the dominance of GBT and deep learning is given in [[@levinTransferLearningDeep2022 1]]

- Sophisticated neural network architectures might not be required, but rather a mix of regularization approaches to regularize MLPs [[@kadraWelltunedSimpleNets2021]].

- semi-supervised learning with pre-training for tabular data improves feature transfer. Also possible if features differ between the upstream and downstream task. [[@levinTransferLearningDeep2022 1]] 
- reasons why deep learning on tabular data is challenging [[@shavittRegularizationLearningNetworks2018]] (use more as background citation)
- selection is hard e. g., in deep learning, as there are no universal benchmarks and robust, battle tested approaches for tabular data compared to other data sources. (see [[@gorishniyRevisitingDeepLearning2021]])

## Gradient Boosted Trees
- start with "wide" architectures.
- Include random forests, if too few models?
- https://github.com/LeoGrin/tabular-benchmark
- Develop deeper understanding of gradient boosting using papers from https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers
- There are several established libraries such as catboost, XGBoost and LightGBM, (that differ in e. g., the growing policy of trees, handling missing values or the calculation of gradients. (see papers also see [[@josseConsistencySupervisedLearning2020]]))  Their performance however, doesn't differ much. (found in [[@gorishniyRevisitingDeepLearning2021]] and cited [[@prokhorenkovaCatBoostUnbiasedBoosting2018]])
- See [[@huangTabTransformerTabularData2020]] that point out common problems of comparsions between gbts and dl.
### Decision Tree

^5db625

- commonly use decision trees as weak learnrs
- Compare how CatBoost, LightGBM and xgboost are different
- Variants of GBM, comparison: [CatBoost vs. LightGBM vs. XGBoost | by Kay Jan Wong | Towards Data Science](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924) (e. g., symmetric, balanced trees vs. asymetric trees) or see kaggle book for differences between lightgbm, catboost etc. [[@banachewiczKaggleBookData2022]]
- Describe details necessary to understand both Gradient Boosting and TabNet.
- How can missing values be handled in decision trees? (see [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]] as a primer)
  How can categorical data be handled in decision trees?
- See how weighting (`weight` in CatBoost) would be incorporated to the formula. Where does `timestamp` become relevant.
- Round off chapter
### Gradient Boosting Procedure
- Motivation for gradient boosted trees
- Introduce notion of tree-based ensemble. Why are sequentially built trees better than parallel ensembles?
- Start of with gradient boosted trees for regression. Gradient boosted trees for classification are derived from that principle.
- cover desirable properties of gradient boosted trees
- for handling of missing values see [[@twalaGoodMethodsCoping2008]]. Send missing value to whether side, that leads to the largest information gain (Found in [[@josseConsistencySupervisedLearning2020]])
- [[@chenXGBoostScalableTree2016]] use second order methods for optimization.
### Adaptions for Probablistic Classification
- Explain how the Gradient Boosting Procedure for the regression case, can be extended to the classifcation case
- Discuss the problem of obtainining good probability estimates from a boosted decision tree. See e. g., [[@caruanaObtainingCalibratedProbabilities]] or [[@friedmanAdditiveLogisticRegression2000]] (Note paper is commenting about boosting, gradient boosting has not been published at the time)
- Observations in [[@tanhaSemisupervisedSelftrainingDecision2017]] on poor probability estimates are equally applicable.
- See how it solved in [[@prokhorenkovaCatBoostUnbiasedBoosting2018]]
- Look into gradient boosting papers that adress the problem. Look in this list: https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers
## Transformer Networks
- Go "deep" instead of wide
- Explain how neural networks can be adjusted to perform binary classification.
- use feed-forward networks to discuss central concepts like loss function, back propagation etc.
- Discuss why plain vanilla feed-forward networks are not suitable for tabular data. Why do the perform poorly?
- How does the chosen layer and loss function to problem framing
- How are neural networks optimized?
- Motivation for Transformers
- For formal algorithms on Transformers see [[@phuongFormalAlgorithmsTransformers2022]]
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://www.youtube.com/watch?v=EixI6t5oif0
- https://transformer-circuits.pub/2021/framework/index.html
- On efficiency of transformers see: https://arxiv.org/pdf/2009.06732.pdf
- Mathematical foundation of the transformer architecture: https://transformer-circuits.pub/2021/framework/index.html
- Detailed explanation and implementation. Check my understanding against it: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- On implementation aspects see: https://arxiv.org/pdf/2007.00072.pdf
### Network Architecture

### Attention
- cover dot-product attention and sequential attention
- multi-headed attention
- self-attention
### Positional Encoding
### Embeddings
- Could include interesting ideas: [[@gorishniyEmbeddingsNumericalFeatures2022]]
### Extensions in TabNet
- TODO: Check if TabNet can actually be considered a Transformer or if it is just attention-based?
- See paper [[@arikTabNetAttentiveInterpretable2020]]
- cover only transformer for tabular data. Explain why.
- Are there other architectures, that I do not cover? Why is this the case?
- TabNet uses neural networks to mimic decision trees by placing importance on only a few features at each layer. The attention layers in that model replace the dot-product self-attention with a type of sparse layer that allows only certain features to pass through.
- Draw on chapter decision trees [[#^5db625]]
- Visualize decision tree-like behaviour


### Extensions in TabTransformer
- See paper [[@huangTabTransformerTabularData2020]]
- TabTransformer can't capture correlations between categorical and continous features. See [[🧠Deep Learning Methods/Transformer/@somepalliSAINTImprovedNeural2021]]
- Investigate whether my dataset even profits from this type of architecture?
- See about embedding continous features in [[@somepalliSAINTImprovedNeural2021]]
### Extensions in FT-Transformer

[[draft_ft_transformer]]
# Semi-Supervised Approaches

## Selection of Approaches

^c77130
- Start with what the ultimate goal of semi-supervised classification is? See [[@chapelleSemiSupervisedClassificationLow2005]] for catchy introduction. -> We want to find a decision boundary that lies in a low-density region / want to create models that generalize. -> Obtain models that generalize. -> In ultimate consequence avoid overfitting of the model.
- Instead of covering all possible semi-supervised approaches, do it the other way around. Take all the approaches from the supervised chapter and show how they can be extended to the semi-supervised setting. This will result in a shorter, yet more streamlined thesis, as results of supervised and semi-supervised learning can be directly compared. 
- Motivation for semi-supervised learning
- Differentiate semi-supervised learning into its subtypes transductive and inductive learning.
- Insert a graphics. Use it as a guidance through the thesis.
- Explain the limitations of transductive learning.
- General we observe performance improvements
- Labelling of data is costly, sometimes impossible (my case).
- For overview see [[@zhuSemiSupervisedLearningLiterature]]
- for problems / success of semi-supervised learning in tabular data see [[@yoonVIMEExtendingSuccess2020]]
- **pseudo-labelling:** e. g., [How To Build an Efficient NLP Model – Weights & Biases (wandb.ai)](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) and [[@leePseudolabelSimpleEfficient]]. Requires solving the issue of obtaining soft probablities. Reason why pseudo-labels are a poor idea -> requires change in neural network architecture -> we aim for comparability.
- For pretraining ofneural net architectures also see argumentation in [[🧠Deep Learning Methods/Transformer/@somepalliSAINTImprovedNeural2021]]
- See also [[@bahriSCARFSelfSupervisedContrastive2022]], while the paper is not 100 % relevant, it contains interesting citations.
- How does regularization relate to semi-supervised learning? Don't both aim for the same objective?
## Extensions to Gradient Boosted Trees
- Introduce the notion of probilistic classifiers
- Possible extension could be [[@yarowskyUnsupervisedWordSense1995]]. See also Sklearn Self-Training Classifier.
- Discuss why probabilities of gradient boosted trees might be missleading [[@arikTabNetAttentiveInterpretable2020]]
- Problems of tree-based approaches and neural networks in semi-supervised learning. See [[@huangTabTransformerTabularData2020]] or [[@arikTabNetAttentiveInterpretable2020]]and [[@tanhaSemisupervisedSelftrainingDecision2017]]

- See [[@tanhaSemisupervisedSelftrainingDecision2017]] for discussion of self-training in conjunction with decision trees and random forests.
- pseudocode for self-supervised algorithm can be found in [[@tanhaSemisupervisedSelftrainingDecision2017]].![[pseudocode-selftraining.png]]

## Extensions to TabNet
- See [[@huangTabTransformerTabularData2020]] for extensions.
- For pratical implementation see [Self Supervised Pretraining - pytorch_widedeep (pytorch-widedeep.readthedocs.io)](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/self_supervised_pretraining.html)
## Extensions to TabTransformer
- See [[@huangTabTransformerTabularData2020]] for extensions.
- Authors use unsupervised pretraining and supervised finetuning. They also try out techniques like pseudo labelling from [[@leePseudolabelSimpleEfficient 1]] for semi supervised learning among others.
- For pratical implementation see: - For pratical implementation see [Self Supervised Pretraining - pytorch_widedeep](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/self_supervised_pretraining.html)
- single output neuron, fused loss describe what loss function is used
# Empirical Study
- In the subsequent sections we apply methods from (...) in an empirical setting.
## Environment 🟡
As we aim for reproducability and configuration. We give a detailed description about the environment in which the experiments were conducted.

> We recommend that authors state the name of the software and specific release version they used, and also cite the paper describing the tool. In cases when an updated version of a tool used has been published, it should be cited to ensure that the precise version used is clear. And when there is doubt, an e-mail to the developers could resolve the issue.
(found in [[@GivingSoftwareIts2019]])

**Data:** Data set versioning using `SHA256` hash. Data is loaded by key using `wandb` library.
**Hardware:** Training and inference of our models is performed on the bwHPC cluster. Each node features ... cpus, ... gpus with cuda version ... and Ubuntu ... . If runpods (https://www.runpod.io/gpu-instance/pricing) or lambdalabs (https://lambdalabs.com/service/gpu-cloud) is used, cite it as well. If I use my docker image cite as well. Model training of gradient boosting approach and transformers is performed on accelerates. To guarantee deterministic behaviour (note gradient boosting may operate on samples, initializations of weights happens randomly, cuda runtime may perform non-deterministic optimizations) we fix the random seeds.
**Software:** Python version (...). Cite packages as "pandas (v1.5.1)". List software packages:
```python
# copied from repo at 12/08/2022
dependencies = [
  "click==8.1.3",
  "catboost==1.1",
  "einops==0.6.0",
  "fastparquet==2022.12.0",
  "gcsfs==2022.11.0",
  "google-auth==2.15.0", # needed by w&b in runpod
  "modin==0.17.0",
  "numpy==1.23.4",
  "optuna==3.0.3",
  "pandas==1.5.1",
  "pandas-datareader==0.10.0",
  "psutil==5.9.4", #needed by w&b
  "pydantic==1.10.2",
  "python-dotenv>=0.5.1",
  "pyyaml==6.0",
  "requests==2.28.1",
  "scikit-learn==1.1.3",
  "seaborn==0.12.1",
  "shap==0.41.0",
  "torch==1.13.0",
  "tqdm==4.64.1",
  "typer==0.7.0",
  "wandb==0.13.5",
]
```

- source code of experiments and paper is available at https://github.com/KarelZe/thesis/
- Get some inspiration from https://madewithml.com/#mlops

## Data and Data Preparation 🟡

- present data sets for this study
- describe applied pre-processing
- describe and reason about applied feature engineering
- describe and reason about test and training split
### ISE Data Set 🟡
- We construct datasets that suffice (...) and serve as an input to our machine learning models. 
- Refer to chapters (...) for which algorithm requires quote data (e. g., quote rule) and which requires price data (e. g., tick test).
- the ISE data set is the primary target for the study. Use CBOE data set as backup, if needed.
- data comes at a intra-day frequency (which resolution?)
- Data spans from May 2, 2005 to May 31, 2017 + (new samples until 2020). The dates are chosen non-arbitrarily: May 2, 2005 is first day of ISE Open/ Close. May, 31 2017 is last day of availability. We adhere to the data ranges to maintain consistency with [[@grauerOptionTradeClassification2022]]
- **Data sources from [[@grauerOptionTradeClassification2022]]:**
	- intraday option price data from `LiveVol` at a transaction level resolution
	- intraday option quote data from `LiveVol`
	- end-of-day buy and sell trading volumes from `ISE Open/Close Trade Profile` (daily resolution)
	- option and underlying characteristics from `Ivy DB OptionMetrics`
- `LiveVol` dataset contains Specifially, the trade price and trade size / volume, nbbo quotes, quotes and quote sizes of the exchanges where the option is quoted and information on the exchange where the trade is executed (see [[@grauerOptionTradeClassification2022]])
- Provide summary statistics, but wait until pre-processing and generation of true labels are introduced. May add pre-processing and generation of true labels in this chapter.

TODO: Is the delay between trades and quotes relevant here? Probably not, due to how matching is performed in [[@grauerOptionTradeClassification2022]] (See discussion in [[@rosenthalModelingTradeDirection2012]]) 


### CBOE Data Set 🟡
describe if data set is actually used. Write similarily to 

### Pre-processing 🟡
- infer minimimal data types to minimize memory requirements. No data loss happening as required resolution for e. g., mantisse is considered.  (see [here](https://github.com/KarelZe/thesis/blob/main/notebooks/1.0-mb-data_preprocessing_mem_reduce.ipynb) and [here](https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage/notebook)(not used) or [here](https://www.kaggle.com/code/wkirgsn/fail-safe-parallel-memory-reduction) (not used))

**Filter:**
- What preprocessing have been applied. Minimal set of filters. See [[@grauerOptionTradeClassification2022]]
**Scaling:**
- TODO: Why do we perform feature scaling at all?
- TODO: Try out robust scaler, as data contains outliers. Robust scaler uses the median which is robust to outliers and iqr for scaling. 
- TODO: Try out different IQR thresholds and report impact. Similarily done here: https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/
- We scale / normalize features to a $\left[-1,1\right]$  scale using statistics estimated on the training set to avoid data leakage. This is also recommended in [[@huyenDesigningMachineLearning]]. Interestingly, she also writes that empirically the interval $\left[-1,1\right]$ works better than $\left[0,1\right]$. Also read about this on stackoverflow for neural networks, which has to do with gradient calculation.
- Scale to an arbitrary range $\left[a,b\right]$ using the formula from [[@huyenDesigningMachineLearning]]:
$$
x^{\prime}=a+\frac{(x-\min (x))(b-a)}{\max (x)-\min (x)}
$$
- Feature scaling theoretically shouldn't be relevant for gradient boosting due to the way gbms select split points / not based on distributions. Also in my tests it didn't make much of a difference for gbms but for transformers. (see https://github.com/KarelZe/thesis/blob/main/notebooks/3.0b-mb-comparsion-transformations.ipynb) 
- [[@ronenMachineLearningTrade2022]] performed no feature scaling.
- [[@borisovDeepNeuralNetworks2022]] standardize numerical features and apply ordinal encoding to categorical features, but pass to the model which ones are categorical features. 
- [[@gorishniyRevisitingDeepLearning2021]] (p. 6) use quantile transformation, which is similar to the robust scaler, see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-pyf) Note that [[@grinsztajnWhyTreebasedModels2022]] only applied quantile transformations to all features, thus not utilize special implementations for categorical variables.
**Class imbalances:**
- Data set is slightly imbalanced. Would not treat, as difference is only minor and makes it harder. Could make my final decision based on [[@japkowiczClassImbalanceProblem2002]] [[@johnsonSurveyDeepLearning2019]]. Some nice background is also in [[@huyenDesigningMachineLearning]]
- [[@huyenDesigningMachineLearning]] discusses different sampling strategies. -> Could make sense to use stratified sampling or weighted sampling. With weighted sampling we could give classes that are notoriously hard to learn e. g., large trades or index options.

**imputation:**
- Best do a large-scale comparsion as already done for some transformations in  https://github.com/KarelZe/thesis/blob/main/notebooks/3.0b-mb-comparsion-transformations.ipynb. My feeling is that differences are neglectable. 
- Different imputation appraoches are listed in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]]. Basic observation use approaches that can inherently handle missing values. This is a good tradeoff between performance and prediction quality.
- simple overview for imputation https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779
- Refer to previous chapters e. g., on gradient boosting, where handling of missing values should be captured.

### Generation of True Labels
- To evaluate the performance of trade classification algoriths the true side of the trade needs to be known. To match LiveVol data, the total customer sell volume or total or total customer buy volume has to match with the transactions in LiveVol. Use unique key of trade date, expiration date, strike price, option type, and root symbol to match the samples. (see [[@grauerOptionTradeClassification2022]]) Notice, that this leads to an imperfect reconstruction!
- The approach of [[@grauerOptionTradeClassification2022]] matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate
- **fuzzy matching:** e. g., match volumes, even if there are small deviations in the volumes e. g. 5 contracts. Similar technique used for time stamps in [[@savickasInferringDirectionOption2003]]. Why might this be a bad idea?
- Discuss that only a portion of data can be reconstructed. Previous works neglected the unlabeled part. 
- Discuss how previously unused data could be used. This maps to the notion of supervised and semi-supervised learning
- Pseudo Labeling?
- We assign sells the label `0` and buys the label `1`. This has the advantage that the calculation from the logloss doesn't require any mapping. Easy interpretation as probability.

### Exploratory Data Analysis

![[summary_statistics.png]]


- Describe interesting properties of the data set. How are values distributed?
- Examine the position of trade's prices relative to the quotes. This is of major importance in classical algorithms like LR, EMO or CLNV.
- Study if classes are imbalanced and require further treatmeant. The work of [[@grauerOptionTradeClassification2022]] suggests that classes are rather balanced.
- Study correlations between variables
- Remove highly correlated features as they also pose problems for feature importance calculation (e. g. feature permutation)
- Plot KDE plot of tick test, quote test...
![[kde-tick-rule 1.png]]
Perform EDA e. g., [AutoViML/AutoViz: Automatically Visualize any dataset, any size with a single line of code. Created by Ram Seshadri. Collaborators Welcome. Permission Granted upon Request. (github.com)](https://github.com/AutoViML/AutoViz) and [lmcinnes/umap: Uniform Manifold Approximation and Projection (github.com)](https://github.com/lmcinnes/umap)
- The approach of [[@grauerOptionTradeClassification2022]] matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate [[@grauerOptionTradeClassification2022]](p. 9). 
- In [[@easleyOptionVolumeStock1998]] CBOE options are more often actively bought than sold (53 %). Also, the number of trades at the midpoints is decreasing over time [[@easleyOptionVolumeStock1998]]. Thus the authors reason, that classification with quote data should be sufficient. Compare this with my sample!
- In adversarial validation it became obvious, that time plays a huge role. There are multiple options how to go from here:
	- Drop old data. Probably not the way to go. Would cause a few questions. Also it's hard to say, where to make the cut-off.
	- Dynamic retraining. Problematic i. e., in conjunction with pretrained models.
	- Use Weighting. Yes! Exponentially or linearily or date-based. Weights could be used in all models, as a feature or through penelization. CatBoost supports this through `Pool(weight=...)`. For PyTorch one could construct a weight tensor and used it when calculating the loss (https://stackoverflow.com/questions/66374709/adding-custom-weights-to-training-data-in-pytorch).
### Feature Engineering
- Some features are more difficult to learn for decision trees and neural nets. Provide aid. https://www.kaggle.com/code/jeffheaton/generate-feature-engineering-dataset/notebook
- Two aspects should drive the feature selection / creation:
	- Use features, that are used in classical rules
	- Apply transformers, that are best suited for the models.
	- Create larger feature sets to find the very best feature set.
- Which features are very different in the training set and the validation set?
- Which features are most important in adversarial validation?
- Plot distributions of features from training and validation set. Could also test using https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test test if samples are drawn from the same distribution.
- See https://neptune.ai/blog/tabular-data-binary-classification-tips-and-tricks-from-5-kaggle-competitions for more ideas
- Previously not done due to use of simple rules only. 
- Try different encondings e. g., of the spread.
- Which architectures require what preprocessing? Derive?
- Perform search?
- Standardization is necessary for algorithms that are sensitive to the scale of features, such as neural net. [[@banachewiczKaggleBookData2022]]
- Use simple method to impute missing data like mean. We seldomly require more sophisticated methods. [[@banachewiczKaggleBookData2022]]
- Look which of the models can handle missing data inherently. How would TabNet or TabTransformer do it?
- Can some of the features be economically motivated?
- Apply feature transformations that are economically motivated.
- It might be wise to limit the transformations to ones that are present in the classical rules. Would help with reasoning.
- Try out features that are inherently used in the depth rule or the trade rule. 
- For imputation look into [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]]
- [[@josseConsistencySupervisedLearning2020]] also compare different imputation methods and handling approaches of missing values in tree-based methods.
- for visualizations and approaches see [[@zhengFeatureEngineeringMachine]] and [[@butcherFeatureEngineeringSelection2020]]
- Positional encoding was achieved using $\sin()$ and $\cos()$ transformation.
- ![[sine_cosine_transform 1.png]]
- [[@ronenMachineLearningTrade2022]] suggest to use models that can handle time series components. This would limit our choices. Thus we use feature engineering to induce a notion of time into our models.

- Implementation pattern https://www.linkedin.com/posts/sarah-floris_python-pythonprogramming-cleancode-activity-6990302724584087552-6lzF?utm_source=share&utm_medium=member_android
- Think about using a frequency of trade feature or so. Also use order imbalances as features. Came up with this idea when reading [[@aitkenIntradayAnalysisProbability1995]]
- Some feature ideas like order imbalance could be adapted from [[@aitkenIntradayAnalysisProbability1995]].
- Positional encode trading time throughout the day.
- Explain why it is necessary to include lagged data as column -> most ml models for tabular data only read rowise. No notion of previous observations etc. Some approaches however exist like specialized attention mechanisms to develop a notion of proximity.
- min-max scaling and $z$ scaling preserve the distribution of the variables  (see [here.](https://stats.stackexchange.com/a/562204/351242)). Applying both cancels out each other (see proof [here.](https://stats.stackexchange.com/a/562204/351242)). 
- zero imputation might be a poor choice for neural networks. (see practical and theoretical explanation in [[@yiWhyNotUse2020]]).
- [[@yiWhyNotUse2020]] and [[@smiejaProcessingMissingData2018]] contain various references to papers to impute missing data in neural networks. 
- Cite [[@rubinInferenceMissingData1976]] for different patterns in missing data.
- [[@lemorvanWhatGoodImputation2021]] for theoretical work on imputation.
- For patterns and analysis of imputed data see https://stefvanbuuren.name/fimd/ch-analysis.html

- What are the drawbacks of feature engineering?
- Why standardize the data at all?
- How is the definition of feature sets be motivated?
- Why does positional encoding make sense?
- Differentiate between categorical and continous variables?
- How to handle high number of categorical variables in data set? How does this relate to gradient boosted trees and transformers?
	- What does it mean for the number of parameters in a transformer model to have one more category?
	- Use a linear projection: https://www.kaggle.com/code/limerobot/dsb2019-v77-tr-dt-aug0-5-3tta/notebook
	- https://en.wikipedia.org/wiki/Additive_smoothing
	- How is the training of the gradient boosted tree affected?
	- For explosion in parameters also see [[@tunstallNaturalLanguageProcessing2022]]. Could apply their reasoning (calculate no. of parameters) for my work. 
	- KISS. Dimensionality is probably not so high, that it can not be handled. It's much smaller than common corpi sizes. Mapping to 'UKNWN' character. -> Think how this can be done using the current `sklearn` implementation.
	- The problem of high number of categories is called a high cardinality problem of categoricals see e. g., [[@huangTabTransformerTabularData2020]]
- Why do we need standardized inputs for neural nets?
- Why does standardization not affect learning of gbms
- Motivation for use of $\log$ is turn lognormal distribution into normal distribution or to reduce variability coming from outliers. 
	- https://datascience.stackexchange.com/questions/40089/what-is-the-reason-behind-taking-log-transformation-of-few-continuous-variables
	- Test log-normality visually with qq-plots (https://stackoverflow.com/questions/46935289/quantile-quantile-plot-using-seaborn-and-scipy) or using statistical tests e. g.,  log-transform + normality test. https://stats.stackexchange.com/questions/134924/tests-for-lognormal-distribution
	- Verify that my observation that log transform works only prices but not so much for size features. Could map to the observation that trade prices are log-normally distributed. https://financetrain.com/why-lognormal-distribution-is-used-to-describe-stock-prices
	- For references to tests for log normality see [[@antoniouLognormalDistributionStock2004]]
	- handle variables with high cardinality
- How do to reduce the number of categorical variables?
- strict assumption as we have out-of-vocabulary tokens e. g., unseen symbols like "TSLA".  (see done differently here https://keras.io/examples/structured_data/tabtransformer/)
- Idea: Instead of assign an unknown token it could help assign to map the token to random vector. https://stackoverflow.com/questions/45495190/initializing-out-of-vocabulary-oov-tokens
- Idea: reduce the least frequent root symbols.
- Apply an idea similar to sentence piece. Here, the number of words in vocabulary is fixed https://github.com/google/sentencepiece. See repo for paper / algorithm.
- redundant features:
	- [[@huangSnapshotEnsemblesTrain2017]] argue, that for continous features both quantized, normalized and log scaled can be kept. The say, that this redundant encoding shouldn't lead to overfitting.
- for final feature set see [[🧃Feature Sets]]
- combine size features and price features into a ratio. e. g., "normalize" price with volume. Found this idea here [[@antoniouLognormalDistributionStock2004]]
- log-transform can hamper interpretability [[@fengLogtransformationItsImplications2014]]
- The right word for testing different settings e. g., scalings or imputation approaches is https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence) 
- In my dataset the previous or subsequent trade price is already added as feature and thus does not have to be searched recursively.
- Motivation for scaling features to $[-1,1]$ range or zero mean. https://stats.stackexchange.com/questions/249378/is-scaling-data-0-1-necessary-when-batch-normalization-is-used
- If needed tokenization support: https://github.com/google/sentencepiece
- 

### Train-Test Split 🟡

^d50f5d
- https://www.coursera.org/learn/machine-learning-projects#syllabus

**How:**
We perform a split into three disjoint sets.
**Sets:**
- Training set is used to fit the model to the data
- Validation set is there for tuning the hyperparameters. [[@hastietrevorElementsStatisticalLearning2009]] (p. 222) write "to estimate prediction error for model selection"
- Test set for unbiased, out-of-sample performance estimates. [[@hastietrevorElementsStatisticalLearning2009]] write "estimate generalization error of the model" (p. 222)
- Common splitting strategy should be dependent on the training sample size and signal-to-noise ratio in the data. [[@hastietrevorElementsStatisticalLearning2009]] (p. 222)
- A common split is e g. 50-25-25. [[@hastietrevorElementsStatisticalLearning2009]] (p. 222)
- We use a 60-20-20 split, and assign dates to be either in one set to simplify evaluation.

**Why:**
The split is required to get unbiased performance estimates of our models. It is not required for classical rules, as these rules have no parameters to estimates or hyperparameters to tune.
To facilitate a fair comparsion we compare both classical rules and our machine learning approches on the common test set and neglect training and validation data for classical rules.

**Classical split over random split:**
A classical train test split is advantegous for a number of reasons:
- We maintain the temporal ordering within the data and avoid data leakage: e. g., from unknown `ROOT`s, as only trailing observations are used. (see similar reasoning in [[@lopezdepradoAdvancesFinancialMachine2018]] for trading strategies).
- The train set holds the most recent and thus most relevant observations.
- Work of [[@grauerOptionTradeClassification2022]] showed that the classification performance deterioriates over time. Thus, most recent data poses the most rigorous test conditions due to the identical data basis.
- Splitting time-correlated data randomly can bias the results and correlations are often non-obvious e. g., `ROOT`s, advent of etfs. She advocates to split data *by time* to avoid leakage (See [[@huyenDesigningMachineLearning]] (p. 137)).
- [[@ronenMachineLearningTrade2022]] performed a 70-30 % random split. This can be problematic for obvious reasons.
**Classical split over CV:**
- computational complexity
- Observations in finance are often not iid. The test set is used multiple times during model development resulting in a testing and selection bias [[@lopezdepradoAdvancesFinancialMachine2018]]. Serial correlation might be less of an issue here.
- use $k$ fold cross validation if possible (see motivation in e. g. [[@banachewiczKaggleBookData2022]] or [[@batesCrossvalidationWhatDoes2022]])
- A nice way to visualize that the models do not overfit is to show how much errors vary across the test folds.
- On cross-validation cite [[@batesCrossvalidationWhatDoes2022]]
**Moving window:**
- Why no moving window. Reason about computational complexity.
**Evaluate similarity of train, test and validation set:**
- Perform [[adversarial_validation]] or https://medium.com/mlearning-ai/adversarial-validation-battling-overfitting-334372b950ba. More of a practioner's approach than a scientific approach though. 
- discuss how split is chosen? Try to align with other works.
- compare distributions of data as part of the data analysis?
- Think about using a $\chi^2$ test to estimate the similarity between train and test set. Came up with this idea while reading [[@aitkenIntradayAnalysisProbability1995]]. Could help finding features or feature transformations that yield a similar train and test set.
- Write how target variable is distributed in each set. 
 Show that a stratified train-test-split is likely not necessary to maintain the distribution of the target variable.
- Plot learning curves to estimate whether performance will increase with the number of samples. Use it to motivate semi-supervised learning.  [Plotting Learning Curves — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) and [Tutorial: Learning Curves for Machine Learning in Python for Data Science (dataquest.io)](https://www.dataquest.io/blog/learning-curves-machine-learning/)
![[learning-curves-samples 1.png]]

## Training and Tuning
- Do less alchemy and more understanding [Ali Rahimi's talk at NIPS(NIPS 2017 Test-of-time award presentation) - YouTube](https://www.youtube.com/watch?v=Qi1Yry33TQE)
- Keep algorithms / ideas simple. Add complexity only where needed! 
- Do rigorous testing.
- Don't chase the benchmark, but aim for explainability of the results.
- compare against https://github.com/jktis/Trade-Classification-Algorithms
- Classical rules could be implemented using https://github.com/jktis/Trade-Classification-Algorithms
- Motivate the importance of regularized neural nets with [[@kadraWelltunedSimpleNets2021]] papers. Authors state, that the improvements from regualrization of neural nets are very pronounced and highly significant. Discuss which regularization approaches are applied and why.  
- Similarily, [[@heBagTricksImage2018]] show how they can improve the performance of neural nets for computer vision through "tricks" like lr scheduling.
- Also see [[@shavittRegularizationLearningNetworks2018]] for regularization in neural networks for tabular data.
- On activation function see [[@shazeerGLUVariantsImprove2020]]

### Training of Supervised Models
- Start with something simple e. g., Logistic Regression or Gradient Boosted Trees, due to being well suited for tabular data. Implement robustness checks (as in [[@grauerOptionTradeClassification2022]]) early on.
- Use classification methods (*probabilistic classifier*) that can return probabilities instead of class-only for better analysis. Using probabilistic trade classification rules might have been studied in [[@easleyDiscerningInformationTrade2016]]
- Interesting notebook about TabNet [Introduction to TabNet - Kfold 10 [TRAINING] | Kaggle](https://www.kaggle.com/code/ludovick/introduction-to-tabnet-kfold-10-training/notebook)
- Use [Captum · Model Interpretability for PyTorch](https://captum.ai/) to learn what the model picks up as a relevant feature.
- Try out Stochastic weight averaging for neural net as done [here.](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) or here [Stochastic Weight Averaging in PyTorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
- Try out adverserial weight perturbation as done [here.][feedback-nn-train | Kaggle](https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook)
- Try out ensembling as in [[@huangSnapshotEnsemblesTrain2017]]
- Try ADAM optimizer first, try out Adan by [[@xieAdanAdaptiveNesterov2022]] for fun. 
- Get inspiration for code from https://github.com/kathrinse/TabSurvey e. g., on saving results.
- use cyclic learning rates as done in [[@huangSnapshotEnsemblesTrain2017]]
- try cyclic learning rates https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html
- cycling procedure was proposed in [[@loshchilovSGDRStochasticGradient2017]] and [[@smithCyclicalLearningRates2017]]
- Test if buys and sells are *really* imbalanced, as indicated by [[@easleyOptionVolumeStock1998]]. Might require up-or downsampling.
- For implementation see https://www.learnpytorch.io/07_pytorch_experiment_tracking/
- Visualize learned embeddings for categorical data as done in [[@huangTabTransformerTabularData2020]]. Also see attention masks in [[@borisovDeepNeuralNetworks2022]] code.
- For improvement of TabTransformer
	- also see https://keras.io/examples/structured_data/tabtransformer/
	- use einsum that is part of torch already instead of external libary as done in  https://github.com/radi-cho/GatedTabTransformer/blob/master/gated_tab_transformer/gated_tab_transformer.py
	- Alternatively see https://github.com/timeseriesAI/tsai/blob/be3c787d6e6d0e41839faa3e62d74145c851ef9c/tsai/models/TabTransformer.py#L133 or original implementation https://github.com/autogluon/autogluon/blob/master/tabular/src/autogluon/tabular/models/tab_transformer/tab_transformer.py
	- We implement the classical rules as a classifier conforming to the sklearn api
- Compare TabTransformer implementations with:
	- https://github.com/aruberts/TabTransformerTF/blob/main/tabtransformertf/models/tabtransformer.py
	- https://github.com/manujosephv/pytorch_tabular/blob/main/pytorch_tabular/models/tab_transformer/tab_transformer.py
	- Simplify / cross-validate implementation of TabTransfromer and FTTransformer against https://pytorch.org/tutorials/beginner/transformer_tutorial
	- Can use to gather some ideas for TabNet: https://www.kaggle.com/code/medali1992/amex-tabnetclassifier-feature-eng-0-791/notebook and TabTransformer https://www.kaggle.com/code/yekenot/amex-pytorch-tabtransformer
	- 
### Training of Semi-Supervised Models
- Justify training of semi-supervised model from theoretical perspective with findings in chapter [[#^c77130]] . 
- Use learning curves from [[#^d50f5d]].

### Hyperparameter Tuning

- we set a time budget and hyperparameter constraints
- repeat with different random initializations e. g., use first https://de.wikipedia.org/wiki/Mersenne-Zahl as seeds
- there is a trade-off between robustness of results and the computational effort / search space

- Explain the importance why hyperparam tuning deserves its own chapter. - > even simple architectures can obtain SOTA-results with proper hyperparameter settings. -> See in-depth analysis in [[@melisStateArtEvaluation2017]] (found in [[@kadraWelltunedSimpleNets2021]])
- [[@melisStateArtEvaluation2017]] investigate hyperparam tuning by plotting validation losses against the hyperparams. 
- ![[validation-loss-vs-hyperparam 1.png]]
- [[@melisStateArtEvaluation2017]] also they try out different seeds. Follow their recommendations.
- See e. g., [[@olbrysEvaluatingTradeSide2018]][[@owenHyperparameterTuningPython2022]] for ideas / most adequate application.
- What optimizer is chosen? Why? Could try out Adam or Adan?
- Start with something simple like GridSearch. Implement in Optuna, so that one can easily switch between grid search, randomized search, Bayesian search etc. [09_Hyperparameter-Tuning-via-Optuna.ipynb - Colaboratory (google.com)](https://colab.research.google.com/github/PacktPublishing/Hyperparameter-Tuning-with-Python/blob/main/09_Hyperparameter-Tuning-via-Optuna.ipynb#scrollTo=580226e9-cc08-4dc7-846b-914876343071) 
- For optuna integration into weights and biases see [this article.](https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893)
- Perform comparsion between different samplers to study how sampler effects parameter search. e. g. see best estimate after $n$ trials.
- Also possible to optimize for multiple objectives e. g., accuracy and ... [optuna.visualization.plot_pareto_front — Optuna 3.0.2 documentation](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html)
- See reasoning towards Bayesian search in my last paper. (see e. g., [[@shahriariTakingHumanOut2016]]) 
- For implementations on tab transformer, tabnet and tabmlp see: pytorch wide-deep package.
- for most important hyperparams in litegbm, catboost etc. (see [[@banachewiczKaggleBookData2022]])
- Visualize training and validation curves (seimilar to [3.4. Validation curves: plotting scores to evaluate models — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/modules/learning_curve.html))
![[sample-validation-curve 1.png]]
When using optuna draw a boxplot. optimal value should lie near the median. Some values should be outside the IQR.
![[optuna-as-boxplot 1.png]]

- compare results of untuned and tuned models. Similar to [[@gorishniyRevisitingDeepLearning2021]].

Repeat search with different random initializations:
![[random-searches-hyperparms 1.png]]
(found in [[@grinsztajnWhyTreebasedModels2022]])

Show differences from different initializations using a violin plot. (suggested in [[@melisStateArtEvaluation2017]])

- For tree-parzen estimator see: https://neptune.ai/blog/optuna-guide-how-to-monitor-hyper-parameter-optimization-runs
- Framing hyperparameter search as an optimization problem. https://www.h4pz.co/blog/2020/10/3/optuna-and-wandb
- perform ablation study (https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)) when making important changes to the architecture. This has been done in [[@gorishniyRevisitingDeepLearning2021]].
- For implementation of permutation importance see https://www.rasgoml.com/feature-engineering-tutorials/how-to-generate-feature-importance-plots-using-catboost

## Evaluation
### Feature Importance Measure
- Feature Importance of Gradient Boosted Trees
	- Possibilities to calculate feature importances in GBMs [here.](https://blog.tensorflow.org/2019/03/how-to-train-boosted-trees-models-in-tensorflow.html)
- Feature Importance of TabNet
	- allows to obtain both local / and global importance
- Feature Importance of TabTransformer
- Unified Approach for Feature Importance
	- Make feature importances comparable across models.
	- For simple methods see permutation importance, ice and coutnerfactuals (see [8.5 Permutation Feature Importance | Interpretable Machine Learning (christophm.github.io)](https://christophm.github.io/interpretable-ml-book/feature-importance.html))
	- Open question: How are correlations handled in SHAP?
	- Think about using kernel-shap. Could work. See e. g., [Feature importance in deep learning - Deep Learning - Deep Learning Course Forums (fast.ai)](https://forums.fast.ai/t/feature-importance-in-deep-learning/42026/91?page=4) and [Census income classification with Keras — SHAP latest documentation](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/neural_networks/Census%20income%20classification%20with%20Keras.html)
	- If SHAP is to complex, one could just zero-out features like in [[@guEmpiricalAssetPricing2020]], but be aware of drawbacks. Yet similar method is to randomly permutate features "within a column" and see how to prediction changes" (see [[@banachewiczKaggleBookData2022]]) also comes at the advantage that no retraining is needed, but artificially breaks correlations etc. (see my previous seminar paper).
	- [[@ronenMachineLearningTrade2022]] study the feature importance only for random forests.
- For explanation on SHAP and difference between pre- and post-model explainability see [[@baptistaRelationPrognosticsPredictor2022]]
### Evaluation Metric
- Discuss what metrics are reasonable e. g., why is it reasonable to use the accuracy here? Dataset is likely balanced with a 50-50 distribution, metrics like accuracy are fine for this use case.
- Define the metrics.
- Accuracy, ROC-curve, area under the curve. Think about statistical Tests e. g., $\chi^2$-Test
- Introduce concept of a confusion matrix. Are all errors equally problematic?


# Results
- What are the findings? Find appropriate visualization (e. g., tables, charts)
-  For each tuned configuration, we run 15 experiments with different random seeds and report the performance on the test set. For some algorithms, we also report the performance of default configurations without hyperparameter tuning. [[@gorishniyRevisitingDeepLearning2021]]
- divide sample into zero ticks and non-zero ticks and see how the accuracy behaves. This was e. g. done in [[@finucaneDirectTestMethods2000]]. See also this paper for reasoning on zero tick and non-zero tick trades.
- Think about stuying the economic impact of false classification trough portfolio construction as done in [[@jurkatisInferringTradeDirections2022]]
- perform friedman test to compare algorithms. (see [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- See [[@odders-whiteOccurrenceConsequencesInaccurate2000]] she differentiates between a systematic and non-systematic error and studies the impact on the results in other studies. She uses the terms bias and noise. She also performs several robustness checks to see if the results can be maintained at different trade sizes etc.
- [[@huyenDesigningMachineLearning]] suggest to tet for fairness, calibration, robustness etc. through:
	- perturbation: change data slightly, add noise etc.
	- invariance: keep features the same, but change some sensitive information
	- Directional expectation tests. e. g. does a change in the feature has a logical impact on the prediction e. g. very high bid (**could be interesting!**)

## Results of Supervised Models
- Results for random classifier
- What would happen if the classical rules weren't stacked?
- Confusion matrix
- ROC curve. See e. g., [this thread](https://stackoverflow.com/a/38467407) for drawing ROC curves

![[visualize-classical-rules-vs-ml 1.png]]
(print heatmap with $y$ axis with ask, bid and mid, $x$-axis could be some other criteria e. g. the trade size or none. If LR rule was good fit for options, accuracy should be evenly distributed and green. Visualize accuracy a hue / color)
- calculate $z$-scores / $z$-statistic of classification accuracies to assess if the results are significant. (see e. g., [[@theissenTestAccuracyLee2000]])
- provide $p$-values. Compare twitter / linkedin posting of S. Raschka on deep learning paper.
- When ranking algorithms think about using the onesided Wilcoxon signed-rank test and the Friedman test. (see e. g. , code or practical application in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- Study removal of features with high degree of missing values with feature permutation. (see idea / code done in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]])
- How do classical rules compare to a zero rule baseline? Zero rule baseline predicts majority class. (variant of the simple heuristic). The later uses simple heuristics to perform a heuristic. 
- Compare against "existing solutions" e. g., LR algorithm, depth rule etc.
## Results of Semi-Supervised Models

Use $t$-SNE to assess the output of the supervised vs. the semi-supervised train models. See [[@leePseudolabelSimpleEfficient 1]] and [[@banachewiczKaggleBookData2022]] for how to use it.
See [[@vandermaatenVisualizingDataUsing2008]] for original paper.
![[t-sne-map 1.png]]

## Feature Importance
- local vs. global attention
- Visualize attention
- make models comparable. Find a notion of feature importance that can be shared across models.
 - compare feature importances between approachaes like in paper
 - How do they selected features relate to what is being used in classical formulas? (see [[#^ce4ff0]]) Could a hybrid formula be derived from the selection by the algorithm?
 - What is the economic intuition?

![[informative-uniformative-features 1.png]]
[[@grinsztajnWhyTreebasedModels2022]]
Interesting comments: https://openreview.net/forum?id=Fp7__phQszn
- Most finance papers e. g., [[@finucaneDirectTestMethods2000]] (+ other examples as reported in expose) use logistic regression to find features that affect the classification most. Poor choice due to linearity assumption? How would one handle categorical variables? If I opt to implement logistic regression, also report $\chi^2$.
- Think about approximating SHAP values on a sample or using some improved implementation like https://github.com/linkedin/FastTreeSHAP
- Do ablation studies. That is, the removal of one feature shouldn't cause the model to collapse. (idea found in [[@huyenDesigningMachineLearning]])
## Robustness Checks
- LR-algorithm (see [[#^370c50]]) require an offset between the trade and quote. How does the offset affect the results? Do I even have the metric at different offsets?
- Perform binning like in [[@grauerOptionTradeClassification2022]]
- Study results over time like in [[@olbrysEvaluatingTradeSide2018]]
- Are probabilities a good indicator reliability e. g., do high probablities lead to high accuracy.
- Are there certain types of options that perform esspecially poor?
- Confusion matrix
- create kde plots to investigate misclassified samples further
- ![[kde-plot-results 1.png]]
- What is called robustnesss checks is also refered as **slice-based evaluation**. The data is separated into subsets and your model's performance on each subset is evaluated. A reason why slice-based evaluation is crucial is Simpson's paradox. A trend can exist in several subgroups, but disappear or reverse when the groups are combined. Slicing could happen based on heuristics, errors or a slice finder (See [[@huyenDesigningMachineLearning]])
![[rankwise-correlations.png]]
(found in [[@hansenApplicationsMachineLearning]], but also other papers)
# 💣Discussion
- What does it mean? Point out limitations and e. g., managerial implications or future impact.
- How do wide models compare to deep models
- Study sources of missclassification. See e. g., [[@savickasInferringDirectionOption2003]]
- Would assembeling help here? As discussed in [[@huangSnapshotEnsemblesTrain2017]] ensembles can only improve the model, if individual models have a low test error and if models do not overlap in the samples they missclassify.
- The extent to which inaccurate trade classification biases empirical research dependes on whether misclassifications occur randomly or systematically [[@theissenTestAccuracyLee2000]]. This document also contains ideas how to study the impact of wrong classifications in stock markets. Might different in option markets.
- Ceveat is that we don't know the true labels, but rather subsets. Could be biased?
# Conclusion
- Repeat the problem and its relevance, as well as the contribution (plus quantitative results).
# 🌄Outlook
- Provide an outlook for further research steps.