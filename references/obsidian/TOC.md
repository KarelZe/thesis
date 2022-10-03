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

# Related Work
- [[@grauerOptionTradeClassification2022]]
- [[@savickasInferringDirectionOption2003]]
- [[@olbrysEvaluatingTradeSide2018]]
- ...
- Which works performed trade side classification for stocks, for options or other products.

# Methodology
1. Broader term is **trade site classification** = assign the side to a to a transaction and differentiate between buyer- and seller-initiated transactions
- There is no single definition / understanding for the one who initiates trades. [[@olbrysEvaluatingTradeSide2018]] distinguish / discuss immediacy and initiator


3. Classical Approaches
		1. Basic Approaches
			1. Rule-based Approach
			2. Quote-based Approach
		2. Extensions
				- What are common extensions? How do new algorithms extend the classical ones? What is the intuition? How do they perform? How do the extensions relate? Why do they fail? In which cases do they fail?
				- [[@savickasInferringDirectionOption2003]]
				- [[@grauerOptionTradeClassification2022]]
			1. Reverse Tick Rule 
			2. Trade Size Rule
	2.  Machine Learning-based Approaches
		- Establish criteria for choosing an architecture. Performance and interpretability?
		- Where does supervised / semi-supervised learning fit into the picture?
		- What ML approaches have been applied? Why is there such a gap?
		- Distinguish between supervised / semisupervised learning
		- What works for similar problems?
		- Problems of tree-based approaches and neural networks in semi-supervised learning. See [[@huangTabTransformerTabularData2020]] or [[@arikTabNetAttentiveInterpretable2020]]and [[@tanhaSemisupervisedSelftrainingDecision2017]]

# Empirical Study
1. Data and Data Prepration
		1. Data Set
			- Describe interesting properties of the data set. How are values distributed?
			- What preprocessing have been applied. See [[@grauerOptionTradeClassification2022]]
		2. Feature Engineering
			- Previously not done due to use of simple rules only. 
			- Try different encondings e. g., of the spread.
		3. Labelling
			- Discuss problems of Tree-based approaches for semi-supervised learning
		1. Train-Test Split
	2. Model Selection and Evaluation
		1. Hyperparameter Tuning
			- See e. g., [[@olbrysEvaluatingTradeSide2018]][[@owenHyperparameterTuningPython2022]] for ideas / most adequate application.
	3. Evaluation Criteria
		1. Metrics
		2. Statistical Tests e. g., $\chi^2$-Test


# Results
1. Classification Results
	2. Sources of Missclassification
		- See e. g., [[@savickasInferringDirectionOption2003]]

# Discussion

# Conclusion