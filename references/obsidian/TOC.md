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

# Abstract
- [nature-summary-paragraph.pdf](https://www.nature.com/documents/nature-summary-paragraph.pdf)

# Introduction
- see  `writing-a-good-introduction.pdf`
- trade site classification matters for several reasons, market liqudity measures, short sells, study of bid-ask-spreads.
- Where is trade side classification applied? Why is it important? Do citation search.
- Repeat in short the motivation
- Outpeformance in similar / other domains
- Obtain probabilities for further analysis

- Crisp sentence of what ML is and why it is promising here. 

- goal is to outperform existing classical approaches

- [[@rosenthalModelingTradeDirection2012]] lists fields where trade classification is used and what the impact of wrongly classified trades is.


# Related Work
- [[@grauerOptionTradeClassification2022]]
- [[@savickasInferringDirectionOption2003]]
- [[@olbrysEvaluatingTradeSide2018 1]]
- [[@ronenMachineLearningTrade2022]] / [[@fedeniaMachineLearningCorporate2021]] They employ a machine learning-based approach for trade side classification. Selection of method follows no clear research agenda, so does sample selection or tuning. Also leaves out latest advancements in prediction of tabular data such as GBM or dedicated NN architectures. Data set only spans two days? General saying ML based predictor (random forest) outperforms tick rule and BVC. Still much human inutition is required for feature engineering. Treated as **supervised tasks**. More recent approaches and also ML approaches outperform classical approaches due to a higher trading frequency. Transfer learning not successful. **Note:** Tick rule has been among the poorest predictors in Grauer. **Note:** Check what the actual difference between the two papers are....
- Which works performed trade side classification for stocks, for options or other products.
- [[@rosenthalModelingTradeDirection2012]] incorporates different methods into a model for the likelihood a trade was buyer-initiated. It's a simple logistic regresssion. Performed on stocks. 
- [[@blazejewskiLocalNonparametricModel2005]] compare $k$-nn and logistic regression for trade-side classification. Performed for Australian stocks. Unclear how results compare to classical rules. 
1. Broader term is **trade site classification** = assign the side to a to a transaction and differentiate between buyer- and seller-initiated transactions
2. It's also sometimes called trade sign classification
- There is no single definition / understanding for the one who initiates trades. [[@olbrysEvaluatingTradeSide2018 1]] distinguish / discuss immediacy and initiator
- Do not compare accuracies across different datasets. This won't work. Might mention [[@grauerOptionTradeClassification2022]] as it is calculated on (partly) the same data set.
- [[@blazejewskiLocalNonparametricModel2005]] use $k$-nn to infer the sign of a trade on the stock market.
- 

# Rule-Based Approaches
## Basic Rules
- See [Quantitative Finance Stack Exchange](https://quant.stackexchange.com/questions/8843/what-are-modern-algorithms-for-trade-classification) for most basic overview
### Tick Test
- Tick tests use changes in trade prices and look at previous trade prices to infer trade direction. If the trade occurs at a higher price, hence uptick, as the previous trade its classified as as buyer-initiated. If the trade occurs at a lower price its seller-iniated. If the price change is zero, the last price is taken, that is different from the current price. (see e. g., [[@grauerOptionTradeClassification2022]] or [[@finucaneDirectTestMethods2000]] or [[@leeInferringTradeDirection1991]] for similar framing)
- One of the first works who mention the tick test is [[@holthausenEffectLargeBlock1987]] (referred to as tick classification rule) or [[@hasbrouckTradesQuotesInventories1988]] (referred to as transaction rule)
- ![[formula-tick-rule.png]]
	Adapted from [[@olbrysEvaluatingTradeSide2018 1]]
- Sources of error in the tick test, when quotes change.
- ![[missclassification-trade-rule.png]] [[@finucaneDirectTestMethods2000]]
### Reverse Tick Test
- Instead of the previous trade, the reverse tick rule uses the subsequent trade price to classify the current trade. 
- If the next trade price that is differnet from the current price, is below the current price the trade (on a down tick or zero down tick) is classified as buyer-initiated. If the next distinguishable price is above the current price (up tick or zero up tick), the current price the trade is seller-initiated. (loosely adapted from [[@grauerOptionTradeClassification2022]]) (see also [[@leeInferringTradeDirection1991]])

### Quote-Rule
- The quote rule classifies a trade as buyer initiated if the trade price is above the midpoint of the buy and ask as buys and if it is below as seller-iniated. Can not classify at the midpoint of the quoted spread. (see e.g., [[@leeInferringTradeDirection1991]] or [[@finucaneDirectTestMethods2000]])

- ![[formula-quote-rule.png]]
	Adapted from [[@olbrysEvaluatingTradeSide2018 1]]. Rewrite to formula
## Extended Rules

^ce4ff0

- What are common extensions? How do new algorithms extend the classical ones? What is the intuition? How do they perform? How do the extensions relate? Why do they fail? In which cases do they fail?
- [[@savickasInferringDirectionOption2003]]
- [[@grauerOptionTradeClassification2022]]
- Which do I want to cover? What are their theoretical properties
- What are common observations or reasons why authors suggested extensions? How do they integrate to the previous approaches? Could this be visualised for a streamlined overview / discussion. 
- What do we find, if we compare the rules 
### Lee and Ready Algorithm
### Ellis-Michaely-O’Hara Rule
### CLNV Method
### Reverse Tick Rule 
### Trade Size Rule
### Rosenthal's Rule

# Supervised Approaches
- Introduce a classifcation that differentiates between supervised, unsupervised, reenforcement learning and semi-supervised learning. 
- Introduce the concept of classification as a variant of supervised learning. 
- Could be supervised if all labels are known
- Could be semi-supervised if only some of the labels are known. Cover later.
- Use probabilistic classifier to allow for more in-depth analysis.
- Search for paper that performed a comparsion between Gradient Boosted Trees and Neural Net on large set of data sets....

## Selection of Approaches

^d8f019

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
## Gradient Boosted Trees
- start with "wide" architectures.
### Decision Tree

^5db625

- commonly use decision trees as weak learnrs
- Compare how CatBoost, LightGBM and xgboost are different
- Variants of GBM, comparison: [CatBoost vs. LightGBM vs. XGBoost | by Kay Jan Wong | Towards Data Science](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924) (e. g., symmetric, balanced trees vs. asymetric trees) or see kaggle book for differences between lightgbm, catboost etc. [[@banachewiczKaggleBookData2022]]
- Describe details necessary to understand both Gradient Boosting and TabNet.
- Round off chapter
### Gradient Boosting Procedure
- Motivation for gradient boosted trees
- Introduce notion of tree-based ensemble. Why are sequentially built trees better than parallel ensembles?
- Start of with gradient boosted trees for regression. Gradient boosted trees for classification are derived from that principle.
- cover desirable properties of gradient boosted trees
### Adaptions for Probablistic Classification
- Explain how the Gradient Boosting Procedure for the regression case, can be extended to the classifcation case
- Discuss the problem of obtainining good probability estimates from a boosted decision tree. See e. g., [[@caruanaObtainingCalibratedProbabilities]] or [[@friedmanAdditiveLogisticRegression2000]] (Note paper is commenting about boosting, gradient boosting has not been published at the time)
## Transformer Networks
- Go "deep" instead of wide
- Explain how neural networks can be adjusted to perform binary classification.
- use feed-forward networks to discuss central concepts like loss function, back propagation etc.
- Discuss why plain vanilla feed-forward networks are not suitable for tabular data. Why do the perform poorly?
- How does the chosen layer and loss function to problem framing
- How are neural networks optimized?
- Motivation for Transformers
- For formal algorithms on Transformers see [[@phuongFormalAlgorithmsTransformers2022]]
### Network Architecture
### Attention
- cover dot-product attention and sequential attention
- multi-headed attention
- self-attention
### Positional Encoding
### Embeddings
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
- TabTransformer can't capture correlations between categorical and continous features. See [[@somepalliSAINTImprovedNeural2021]]
- Investigate whether my dataset even profits from this type of architecture?
### Extensions in FT-Transformer
- Variant of the classical transformer, but for tabular data. Published in [[@gorishniyRevisitingDeepLearning2021]]
- Firstly, Feature Tokenizer transforms features to embeddings. The embeddings are then processed by the Transformer module and the final representation of the (CLS) token is used for prediction.
- Very likely not interpretable...
# Semi-Supervised Approaches

^c77130

- Instead of covering all possible semi-supervised approaches, do it the other way around. Take all the approaches from the supervised chapter and show how they can be extended to the semi-supervised setting. This will result in a shorter, yet more streamlined thesis, as results of supervised and semi-supervised learning can be directly compared. 
- Motivation for semi-supervised learning
- Differentiate semi-supervised learning into its subtypes transductive and inductive learning.
- Insert a graphics. Use it as a guidance through the thesis.
- Explain the limitations of transductive learning.
- General we observe performance improvements
- Labelling of data is costly, sometimes impossible (my case).
- For overview see [[@zhuSemiSupervisedLearningLiterature]]
## Extensions to Gradient Boosted Trees
- Introduce the notion of probilistic classifiers
- Possible extension could be [[@yarowskyUnsupervisedWordSense1995]]. See also Sklearn Self-Training Classifier.
- Discuss why probabilities of gradient boosted trees might be missleading [[@arikTabNetAttentiveInterpretable2020]]
- Problems of tree-based approaches and neural networks in semi-supervised learning. See [[@huangTabTransformerTabularData2020]] or [[@arikTabNetAttentiveInterpretable2020]]and [[@tanhaSemisupervisedSelftrainingDecision2017]]
## Extensions to TabNet
- See [[@arikTabNetAttentiveInterpretable2020]] for extensions.
- For pratical implementation see [Self Supervised Pretraining - pytorch_widedeep (pytorch-widedeep.readthedocs.io)](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/self_supervised_pretraining.html)
## Extensions to TabTransformer
- See [[@huangTabTransformerTabularData2020]] for extensions.
- Authors use unsupervised pretraining and supervised finetuning. They also try out techniques like pseudo labelling from [[@leePseudolabelSimpleEfficient]] for semi supervised learning among others.
- For pratical implementation see: - For pratical implementation see [Self Supervised Pretraining - pytorch_widedeep](https://pytorch-widedeep.readthedocs.io/en/latest/pytorch-widedeep/self_supervised_pretraining.html)
# Empirical Study
## Environment
- Reproducability
- Configuration

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

### ISE Data Set
- focus is on ISE data set
- focus on `nbbo`
- all kind of options are equally important
- data comes at a daily frequency
- Describe interesting properties of the data set. How are values distributed?
- What preprocessing have been applied. See [[@grauerOptionTradeClassification2022]]
- Data range from May 2, 2005 to May 31, 2017 + (new samples)
- Is the delay between trades and quotes relevant here? (See discussion in [[@rosenthalModelingTradeDirection2012]]) Probably not as daily data?
### CBOE Data Set
- data comes at a daily frequency

### Generation of True Labels
- To evaluate the performance of trade classification algoriths the true side of the trade needs to be known. To match LiveVol data, the total customer sell volume or total or total customer buy volume has to match with the transactions in LiveVol. Use unique key of trade date, expiration date, strike price, option type, and root symbol to match the samples. (see [[@grauerOptionTradeClassification2022]]) Notice, that this leads to an imperfect reconstruction!
- Discuss that only a portion of data can be reconstructed. Previous works neglected the unlabeled part. 
- Discuss how previously unused data could be used. This maps to the notion of supervised and semi-supervised learning
- Pseudo Labeling?

### Explanatory Data Analysis
- Examine the position of trade's prices relative to the quotes. This is of major importance in classical algorithms like LR, EMO or CLNV.
- Study if classes are imbalanced and require further treatmeant. The work of [[@grauerOptionTradeClassification2022]] suggests that classes are rather balanced.
- Study correlations between variables
- Remove highly correlated features as they also pose problems for feature importance calculation (e. g. feature permutation)
- Plot KDE plot of tick test, quote test...
![[kde-tick-rule.png]]
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
### Train-Test Split

^d50f5d

- discuss how split is chosen? Try to align with other works.
- Discuss / visualize how training, validation and test set compare.
- compare distributions of data as part of the data analysis?
- Use a stratified train-test-split to maintain the distribution of the target variable.
- use $k$ fold cross validation if possible (see motivation in e. g. [[@banachewiczKaggleBookData2022]] or [[@batesCrossvalidationWhatDoes2022]])
- A nice way to visualize that the models do not overfit is to show how much errors vary across the test folds.
- Plot learning curves to estimate whether performance will increase with the number of samples. Use it to motivate semi-supervised learning.  [Plotting Learning Curves — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) and [Tutorial: Learning Curves for Machine Learning in Python for Data Science (dataquest.io)](https://www.dataquest.io/blog/learning-curves-machine-learning/)
![[learning-curves-samples.png]]

## Training and Tuning
### Training of Supervised Models
- Interesting notebook about TabNet [Introduction to TabNet - Kfold 10 [TRAINING] | Kaggle](https://www.kaggle.com/code/ludovick/introduction-to-tabnet-kfold-10-training/notebook)
- Use [Captum · Model Interpretability for PyTorch](https://captum.ai/) to learn what the model picks up as a relevant feature.
- Try out Stochastic weight averaging for neural net as done [here.](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx) or here [Stochastic Weight Averaging in PyTorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
- Try out adverserial weight perturbation as done [here.][feedback-nn-train | Kaggle](https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook)
- Try out ensembling as in [[@huangSnapshotEnsemblesTrain2017a]]
- Try ADAM optimizer first, try out Adan by [[@xieAdanAdaptiveNesterov2022]] for fun. 

### Training of Semi-Supervised Models
- Justify training of semi-supervised model from theoretical perspective with findings in chapter [[#^c77130]] . 
- Use learning curves from [[#^d50f5d]].

### Hyperparameter Tuning
- See e. g., [[@olbrysEvaluatingTradeSide2018 1]][[@owenHyperparameterTuningPython2022]] for ideas / most adequate application.
- What optimizer is chosen? Why? Could try out Adam or Adan?
- Start with something simple like GridSearch. Implement in Optuna, so that one can easily switch between grid search, randomized search, Bayesian search etc. [09_Hyperparameter-Tuning-via-Optuna.ipynb - Colaboratory (google.com)](https://colab.research.google.com/github/PacktPublishing/Hyperparameter-Tuning-with-Python/blob/main/09_Hyperparameter-Tuning-via-Optuna.ipynb#scrollTo=580226e9-cc08-4dc7-846b-914876343071) 
- For optuna integration into weights and biases see [this article.](https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893)
- Perform comparsion between different samplers to study how sampler effects parameter search. e. g. see best estimate after $n$ trials.
- Also possible to optimize for multiple objectives e. g., accuracy and ... [optuna.visualization.plot_pareto_front — Optuna 3.0.2 documentation](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_pareto_front.html)
- See reasoning towards Bayesian search in my last paper. (see e. g., [[@shahriariTakingHumanOut2016]]) 
- For implementations on tab transformer, tabnet and tabmlp see: pytorch wide-deep package.
- for most important hyperparams in litegbm, catboost etc. (see [[@banachewiczKaggleBookData2022]])
- Visualize training and validation curves (seimilar to [3.4. Validation curves: plotting scores to evaluate models — scikit-learn 1.1.2 documentation](https://scikit-learn.org/stable/modules/learning_curve.html))
![[sample-validation-curve.png]]
When using optuna draw a boxplot. optimal value should lie near the median. Some values should be outside the IQR.
![[optuna-as-boxplot.png]]
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
### Evaluation Metric
- Discuss what metrics are reasonable e. g., why is it reasonable to use the accuracy here? Dataset is likely balanced with a 50-50 distribution, metrics like accuracy are fine for this use case.
- Define the metrics.
- Accuracy, ROC-curve, area under the curve. Think about statistical Tests e. g., $\chi^2$-Test
- Introduce concept of a confusion matrix. Are all errors equally problematic?


# Results
- What are the findings? Find appropriate visualization (e. g., tables, charts)
## Results of Supervised Models
- Results for random classifier
- What would happen if the classical rules weren't stacked?
- Confusion matrix
- ROC curve. See e. g., [this thread](https://stackoverflow.com/a/38467407) for drawing ROC curves
- 
## Results of Semi-Supervised Models
## Robustness Checks
- Perform binning like in [[@grauerOptionTradeClassification2022]]
- Study results over time like in [[@olbrysEvaluatingTradeSide2018 1]]
- Are probabilities a good indicator reliability e. g., do high probablities lead to high accuracy.
- Are there certain types of options that perform esspecially poor?
- Confusion matrix
- create kde plots to investigate misclassified samples further
- ![[kde-plot-results.png]]
## Feature Importance
- local vs. global attention
- Visualize attention
- make models comparable. Find a notion of feature importance that can be shared across models.
 - compare feature importances between approachaes like in paper
 - How do they selected features relate to what is being used in classical formulas? (see [[#^ce4ff0]]) Could a hybrid formula be derived from the selection by the algorithm?
 - What is the economic intuition?
# Discussion
- What does it mean? Point out limitations and e. g., managerial implications or future impact.
- How do wide models compare to deep models
- Study sources of missclassification. See e. g., [[@savickasInferringDirectionOption2003]]
# Conclusion
- Repeat the problem and its relevance, as well as the contribution (plus quantitative results).
# Outlook
- Provide an outlook for further research steps.