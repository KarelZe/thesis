- Off being right: trade-side classification of options data with machine learning
- Trade side classifcation with machine learning: do or don't?
- Improving trade site classication with machine learning.
- Getting it right: trade side classifaction of options using machine learning.
- Getting it right: Improving trade side classification with gradient boosted trees...
- Done right: ...
- Do or do not, there is no try
- Be less wrong. Improving trade site classification with machine learning
- More than a nudge. Improving options trade site classification with machine learning


# Introduction
- see  `writing-a-good-introduction.pdf`
- trade site classification matters for several reasons, market liqudity measures, short sells, study of bid-ask-spreads.
- Where is trade side classification applied? Why is it important? Do citation search.
- Repeat in short the motivation
- Outpeformance in similar / other domains
- Obtain probabilities for further analysis

- Crisp sentence of what ML is and why it is promising here. 

# Related Work
- [[@grauerOptionTradeClassification2022]]
- [[@savickasInferringDirectionOption2003]]
- [[@olbrysEvaluatingTradeSide2018]]
- Which works performed trade side classification for stocks, for options or other products.
1. Broader term is **trade site classification** = assign the side to a to a transaction and differentiate between buyer- and seller-initiated transactions
- There is no single definition / understanding for the one who initiates trades. [[@olbrysEvaluatingTradeSide2018]] distinguish / discuss immediacy and initiator


# Rule-Based Approaches
## Basic Rules
### Tick Rule-Based Approach
### Quote-Based Approach
## Extended Rules
- What are common extensions? How do new algorithms extend the classical ones? What is the intuition? How do they perform? How do the extensions relate? Why do they fail? In which cases do they fail?
- [[@savickasInferringDirectionOption2003]]
- [[@grauerOptionTradeClassification2022]]
- Which do I want to cover? What are their theoretical properties
- What are common observations or reasons why authors suggested extensions? How do they integrate to the previous approaches? Could this be visualised for a streamlined overview / discussion. 
- What do we find, if we compare the rules 
### Lee and Ready Algorithm
### Ellis-Michaely-O’Hara Rule
### Reverse Tick Rule 
### Trade Size Rule
# Supervised Approaches
- Introduce a classifcation that differentiates between supervised, unsupervised, reenforcement learning and semi-supervised learning. 
- Introduce the concept of classification as a variant of supervised learning. 
- Could be supervised if all labels are known
- Could be semi-supervised if only some of the labels are known. Cover later.
- Use probabilistic classifier to allow for more in-depth analysis.

## Selection of Approaches
- What works in similar use cases? What are similar use cases?
- Establish criteria for choosing an architecture:
	- **performance** That is, approach must deliver state-of-the-art performance in similar problems.
	- **interpretability** Classical approaches are transparent in a sense that we know how the decision was derived. In the best case try to aim for local and global interpretability. Think about how interpretability can be narrowed down? Note supervisor wants to see if her features are also important to the model. 
	- Perform a model discussion on results from similar domains.
	- Perform a wide (ensemble) vs. deep (neural net) comparison. 
## Gradient Boosted Trees
- start with "wide" architectures.
### Weak Learner
- commonly use decision trees as weak lernrs
- Compare how CatBoost, LightGBM and xgboost are different
- Variants of GBM, comparison: [CatBoost vs. LightGBM vs. XGBoost | by Kay Jan Wong | Towards Data Science](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924) (e. g., symmetric, balanced trees vs. asymetric trees)
- Round off chapter
### Gradient Boosting Procedure
- Motivation for gradient boosted trees
- Introduce notion of tree-based ensemble. Why are sequentially built trees better than parallel ensembles?
- Start of with gradient boosted trees for regression. Gradient boosted trees for classification are derived from that principle.
- cover desirable properties of gradient boosted trees
### Adaptions for Classification
- Explain how the Gradient Boosting Procedure for the regression case, can be extended to the classifcation case
## Transformer Networks
- Go "deep" instead of wide
- Explain how neural networks can be adjusted to perform binary classification.
- use feed-forward networks to discuss central concepts like loss function, back propagation etc.
- Discuss why plain vanilla feed-forward networks are not suitable for tabular data. Why do the perform poorly?
- How does the chosen layer and loss function to problem framing
- How are neural networks optimized?
- Motivation for Transformers
### Network Architecture
### Attention
- cover dot-product attention and sequential attention
- multi-headed attention
- self-attention
### Positional Encoding
### Embeddings
### Extensions in TabNet
- cover only transformer for tabular data. Explain why.
- Are there other architectures, that I do not cover? Why is this the case?
### Extensions in TabTransformer
# Semi-Supervised Approaches
- Instead of covering all possible semi-supervised approaches, do it the other way around. Take all the approaches from the supervised chapter and show how they can be extended to the semi-supervised setting. This will result in a shorter, yet more streamlined thesis, as results of supervised and semi-supervised learning can be directly compared. 
- Motivation for semi-supervised learning
- Differentiate semi-supervised learning into its subtypes transductive and inductive learning.
- Insert a graphics. Use it as a guidance through the thesis.
- Explain the limitations of transductive learning.
- General we observe performance improvements
- Labelling of data is costly, sometimes impossible (my case).
## Extensions to Gradient Boosted Trees
- Introduce the notion of probilistic classifiers
- Possible extension could be [[@yarowskyUnsupervisedWordSense1995]]. See also Sklearn Self-Training Classifier.
- Discuss why probabilities of gradient boosted trees might be missleading [[@arikTabNetAttentiveInterpretable2020]]
- Problems of tree-based approaches and neural networks in semi-supervised learning. See [[@huangTabTransformerTabularData2020]] or [[@arikTabNetAttentiveInterpretable2020]]and [[@tanhaSemisupervisedSelftrainingDecision2017]]
## Extensions to TabNet
- See [[@arikTabNetAttentiveInterpretable2020]] for extensions.
## Extensions to TabTransformer
- See [[@huangTabTransformerTabularData2020]] for extensions.

# Empirical Study
## Data and Data Preparation
### CBOE Data Set
### ISE Data Set
- Describe interesting properties of the data set. How are values distributed?
- What preprocessing have been applied. See [[@grauerOptionTradeClassification2022]]
### Generation of True Labels
- Discuss that only a portion of data can be reconstructed. Previous works neglected the unlabeled part. 
- Discuss how this maps to the notion of supervised and semi-supervised learning

### Feature Engineering
- Previously not done due to use of simple rules only. 
- Try different encondings e. g., of the spread.
- Which architectures require what preprocessing? Derive?
- Perform search?
### Train-Test Split
- discuss how split is chosen? Try to align with other works.
## Training and Tuning
### Training of Supervised Models
### Training of Semi-Supervised Models
### Hyperparameter Tuning
- See e. g., [[@olbrysEvaluatingTradeSide2018]][[@owenHyperparameterTuningPython2022]] for ideas / most adequate application.
- What optimizer is chosen? Why? Could try out Adam or Adan?
- For implementations on tab transformer, tabnet and tabmlp see: pytorch wide-deep.

## Evaluation
### Feature Importance Measure
- Feature Importance of Gradient Boosted Trees
- Feature Importance of TabNet
- Feature Importance of TabTransformer
-  Unified Approach for Feature Importance
### Evaluation Metric
- Discuss what metrics are reasonable e. g., why is it reasonable to use the accuracy here? Dataset is likely balanced with a 50-50 distribution, metrics like accuracy are fine for this use case.
- Define the metrics.
- Accuracy, ROC-curve, area under the curve. Think about statistical Tests e. g., $\chi^2$-Test


# Results
## Results of Supervised Models
## Results of Semi-Supervised Models
## Robustness Checks
- Perform binning like in [[@grauerOptionTradeClassification2022]]
- Study results over time like in [[@olbrysEvaluatingTradeSide2018]]
- Are probabilities a good indicator reliability e. g., do high probablities lead to high accuracy.
## Feature Importance
- local vs. global attention
- Visualize attention
- make models comparable. Find a notion of feature importance that can be shared across models.
 - compare feature importances between approachaes like in paper

# Discussion
- How do wide models compare to deep models
- Study sources of missclassification. See e. g., [[@savickasInferringDirectionOption2003]]
# Conclusion