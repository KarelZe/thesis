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

# Abstract
- [nature-summary-paragraph.pdf](https://www.nature.com/documents/nature-summary-paragraph.pdf)


# Introduction

- "The validity of many ecomnomic studies hinges on the ability to accuractely classify trades as buyer or seller-initiated." (found in [[@odders-whiteOccurrenceConsequencesInaccurate2000]])

- see  `writing-a-good-introduction.pdf`
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

- For classical rule-based approaches see some citations in [[@olbrysEvaluatingTradeSide2018]]. E. g., [[@chakrabartyTradeClassificationAlgorithms2012]] or [[@chakrabartyEvaluatingTradeClassification2015]]

# 🔗Rule-Based Approaches
## Basic Rules
- See [Quantitative Finance Stack Exchange](https://quant.stackexchange.com/questions/8843/what-are-modern-algorithms-for-trade-classification) for most basic overview
- There are different views of what is considered as buyer / seller iniated i. e. [[@odders-whiteOccurrenceConsequencesInaccurate2000]] vs. [[@ellisAccuracyTradeClassification2000]]
(see [[@theissenTestAccuracyLee2000]] for more details)
- Different views on iniatiting party (i. e. [[@odders-whiteOccurrenceConsequencesInaccurate2000]] vs. [[@chakrabartyTradeClassificationAlgorithms2012]]) (see [[@aktasTradeClassificationAccuracy2014]] for more details)
- Submitters of market orders are called liquidity demanders, while submitters of limit orders stored stored in the book are liquidity providers.
- The BVC paper ([[@easleyDiscerningInformationTrade2016]]) treats trade classification as a probabilistic trade classificatio nproblem. Incorporate this idea into the classical rules section.
- BVC is illsuited for my task, as we require to sign each trade? (see [[@panayidesComparingTradeFlow2014]])
- Algorithms like LR, tick rule etc. are also available in bulked versions. See e. g., [[@chakrabartyEvaluatingTradeClassification2015]] for a comparsion in the stock market. 
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
### Reverse Tick Test
- Instead of the previous trade, the reverse tick rule uses the subsequent trade price to classify the current trade. 
- If the next trade price that is differnet from the current price, is below the current price the trade (on a down tick or zero down tick) is classified as buyer-initiated. If the next distinguishable price is above the current price (up tick or zero up tick), the current price the trade is seller-initiated. (loosely adapted from [[@grauerOptionTradeClassification2022]]) (see also [[@leeInferringTradeDirection1991]])

### Depth Rule
- classify midspread trades as buyer-initiated, if the ask size exceeds the bid size, and as seller-initiated, if the bid size is higher than the ask size (see [[@grauerOptionTradeClassification2022]])
- **Intuition:** trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely. (see [[@grauerOptionTradeClassification2022]])
- Alternative to handle midspread trades, that can not be classified using the quote rule.
- Improves LR algorithm by 0.8 %. Overall accuracy 75 %.
- Performance exceeds that of the LR algorithm, thus the authors assume that the depth rule outperforms the tick test and the reverse tick test, that are used in the LR algorithm for for classifying midspread trades.
### Trade Size Rule
- classify trades for which the trade size is equal to the quoted bid size as customer buys and those with a trade size equal to the ask size as customer sells. (see [[@grauerOptionTradeClassification2022]])
- **Intuition:** trade size matches exactly either the bid or ask quote size, it is likely that the quote came from a customer, the market maker found it attractive and, therefore, decided to fill it completely. (see [[@grauerOptionTradeClassification2022]])  
- Accuracy of 79.92 % on the 22.3 % of the trades that could classified, not all!. (see [[@grauerOptionTradeClassification2022]])
- Couple with other algorithms if trade sizes and quote sizes do not match / or if the trade size matches both the bid and ask size. For other 
- Requires other rules, similar to the quote rule, as only a small proportion can be matched.
- tested on option data / similar data set
## Hybrid Rules

^ce4ff0
- use the problems of the single tick test to motivate extended rules like EMO.
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
### Reverse Lee and Ready Algorithm
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

## Gradient Boosted Trees
- start with "wide" architectures.
- Include random forests, if too few models?
- https://github.com/LeoGrin/tabular-benchmark
- Develop deeper understanding of gradient boosting using papers from https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers
### Decision Tree

^5db625

- commonly use decision trees as weak learnrs
- Compare how CatBoost, LightGBM and xgboost are different
- Variants of GBM, comparison: [CatBoost vs. LightGBM vs. XGBoost | by Kay Jan Wong | Towards Data Science](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924) (e. g., symmetric, balanced trees vs. asymetric trees) or see kaggle book for differences between lightgbm, catboost etc. [[@banachewiczKaggleBookData2022]]
- Describe details necessary to understand both Gradient Boosting and TabNet.
- How can missing values be handled in decision trees? (see [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]] as a primer)
  How can categorical data be handled in decision trees? 
- Round off chapter
### Gradient Boosting Procedure
- Motivation for gradient boosted trees
- Introduce notion of tree-based ensemble. Why are sequentially built trees better than parallel ensembles?
- Start of with gradient boosted trees for regression. Gradient boosted trees for classification are derived from that principle.
- cover desirable properties of gradient boosted trees
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
- Variant of the classical transformer, but for tabular data. Published in [[@gorishniyRevisitingDeepLearning2021]]
- Firstly, Feature Tokenizer transforms features to embeddings. The embeddings are then processed by the Transformer module and the final representation of the (CLS) token is used for prediction.
- Very likely not interpretable...
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
# Empirical Study
## Environment
- Reproducability
- Configuration
- Get some inspiration from https://madewithml.com/#mlops
- Guide on visualizations https://www.nature.com/articles/s41467-020-19160-7

```python
def seed_everything(seed, tensorflow_init=True, pytorch_init=True): 
""" 
Seeds basic parameters for reproducibility of results 
""" 
random.seed(seed) 
os.environ["PYTHONHASHSEED"] = str(seed) np.random.seed(seed) 
if tensorflow_init is True: 
	tf.random.set_seed(seed) 
if pytorch_init is True: 
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
```

## Data and Data Preparation

- convert data to managable size (see [[Preprocessing]])
- https://github.com/aperezlebel/benchmark_mv_approaches
- https://www.runpod.io/gpu-instance/pricing
- https://lambdalabs.com/service/gpu-cloud

### ISE Data Set
- focus is on ISE data set
- focus on `nbbo`
- all kind of options are equally important
- data comes at a daily frequency
- Describe interesting properties of the data set. How are values distributed?
- What preprocessing have been applied. See [[@grauerOptionTradeClassification2022]]
- Data range from May 2, 2005 to May 31, 2017 + (new samples)
- Is the delay between trades and quotes relevant here? (See discussion in [[@rosenthalModelingTradeDirection2012]]) Probably not as daily data?
- Could I obtain access to the previously used data? What kind of features like volumes at certain exchanges would be available?
- How many features does the data set contain? `8 categories for buy and sell volumes  x 4 trader types by option series x trading day`. 
- What should be primary target e. g., classification on `NBBO`? -> all exchanges
- Should the data set remain static or include new samples since publication? Hence, 2017 onwards for the ISE set. -> Focus on ISE 
- merge data from ise + cboe -> needs features from stock exchanges
- standardize numerical features and apply ordinal encoding to categorical features, but pass to the model which ones are categorical features similar to [[@borisovDeepNeuralNetworks2022]]. Note that [[@grinsztajnWhyTreebasedModels2022]] only applied quantile transformations to all features, thus not utilize special implementations for categorical variables.
- Perform [[adversarial_validation]]
- Different imputation appraoches are listed in [[@perez-lebelBenchmarkingMissingvaluesApproaches2022]]. Basic observation use approaches that can inherently handle missing values. This is a good tradeoff between performance and prediction quality.
- For implementation of permutation importance see https://www.rasgoml.com/feature-engineering-tutorials/how-to-generate-feature-importance-plots-using-catboost
- Data set is slightly imbalanced. Would not treat, as difference is only minor and makes it harder. Could make my final decision based on [[@japkowiczClassImbalanceProblem2002]] [[@johnsonSurveyDeepLearning2019]]. Some nice background is also in [[@huyenDesigningMachineLearning]]
- [[@huyenDesigningMachineLearning]] discusses different sampling strategies. -> Could make sense to use stratified sampling or weighted sampling. With weighted sampling we could give classes that are notoriously hard to learn e. g., large trades or index options.
### CBOE Data Set
- data comes at a daily frequency

### Generation of True Labels
- To evaluate the performance of trade classification algoriths the true side of the trade needs to be known. To match LiveVol data, the total customer sell volume or total or total customer buy volume has to match with the transactions in LiveVol. Use unique key of trade date, expiration date, strike price, option type, and root symbol to match the samples. (see [[@grauerOptionTradeClassification2022]]) Notice, that this leads to an imperfect reconstruction!
- The approach of [[@grauerOptionTradeClassification2022]] matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate
- **fuzzy matching:** e. g., match volumes, even if there are small deviations in the volumes e. g. 5 contracts. Similar technique used for time stamps in [[@savickasInferringDirectionOption2003]]. Why might this be a bad idea?
- Discuss that only a portion of data can be reconstructed. Previous works neglected the unlabeled part. 
- Discuss how previously unused data could be used. This maps to the notion of supervised and semi-supervised learning
- Pseudo Labeling?

### Explanatory Data Analysis
- Examine the position of trade's prices relative to the quotes. This is of major importance in classical algorithms like LR, EMO or CLNV.
- Study if classes are imbalanced and require further treatmeant. The work of [[@grauerOptionTradeClassification2022]] suggests that classes are rather balanced.
- Study correlations between variables
- Remove highly correlated features as they also pose problems for feature importance calculation (e. g. feature permutation)
- Plot KDE plot of tick test, quote test...
![[kde-tick-rule 1.png]]
Perform EDA e. g., [AutoViML/AutoViz: Automatically Visualize any dataset, any size with a single line of code. Created by Ram Seshadri. Collaborators Welcome. Permission Granted upon Request. (github.com)](https://github.com/AutoViML/AutoViz) and [lmcinnes/umap: Uniform Manifold Approximation and Projection (github.com)](https://github.com/lmcinnes/umap)
- The approach of [[@grauerOptionTradeClassification2022]] matches the LiveVol data set, only if there is a matching volume on buyer or seller side. Results in 40 % reconstruction rate [[@grauerOptionTradeClassification2022]](p. 9). 
- In [[@easleyOptionVolumeStock1998]] CBOE options are more often actively bought than sold (53 %). Also, the number of trades at the midpoints is decreasing over time [[@easleyOptionVolumeStock1998]]. Thus the authors reason, that classification with quote data should be sufficient. Compare this with my sample!
### Feature Engineering
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
### Train-Test Split

^d50f5d

- discuss how split is chosen? Try to align with other works.
- Discuss / visualize how training, validation and test set compare.
- compare distributions of data as part of the data analysis?
- Use a stratified train-test-split to maintain the distribution of the target variable.
- use $k$ fold cross validation if possible (see motivation in e. g. [[@banachewiczKaggleBookData2022]] or [[@batesCrossvalidationWhatDoes2022]])
- A nice way to visualize that the models do not overfit is to show how much errors vary across the test folds.
- Think about using a $\chi^2$ test to estimate the similarity between train and test set. Came up with this idea while reading [[@aitkenIntradayAnalysisProbability1995]]. Could help finding features or feature transformations that yield a similar train and test set.
- Plot learning curves to estimate whether performance will increase with the number of samples. Use it to motivate semi-supervised learning.  [Plotting Learning Curves — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) and [Tutorial: Learning Curves for Machine Learning in Python for Data Science (dataquest.io)](https://www.dataquest.io/blog/learning-curves-machine-learning/)
- On cross-validation read [[@batesCrossvalidationWhatDoes2022]]
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
- Visualize learned embeddings for categorical data as done in [[@huangTabTransformerTabularData2020]]
### Training of Semi-Supervised Models
- Justify training of semi-supervised model from theoretical perspective with findings in chapter [[#^c77130]] . 
- Use learning curves from [[#^d50f5d]].

### Hyperparameter Tuning

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

Repeat search with different random initializations:
![[random-searches-hyperparms 1.png]]
(found in [[@grinsztajnWhyTreebasedModels2022]])

Show differences from different initializations using a violin plot. (suggested in [[@melisStateArtEvaluation2017]])

- For tree-parzen estimator see: https://neptune.ai/blog/optuna-guide-how-to-monitor-hyper-parameter-optimization-runs
- Framing hyperparameter search as an optimization problem. https://www.h4pz.co/blog/2020/10/3/optuna-and-wandb

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