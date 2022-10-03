# Introduction
- see  `writing-a-bood-introduction.pdf`
- trade site classification matters for several reasons, market liqudity measures, short sells, study of bid-ask-spreads.
- Where is trade side classification applied? Why is it important? Do citation search.

# Related Work
- [[@grauerOptionTradeClassification2022]]
- [[@savickasInferringDirectionOption2003]]
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
				- What are common extensions? How do new algorithms extend the classical ones? What is the intuition? How do they perform?
				- [[@savickasInferringDirectionOption2003]]
				- [[@grauerOptionTradeClassification2022]]
			1. Reverse Tick Rule 
			2. Trade Size Rule
	2.  Machine Learning-based Approaches
		- Distinguish between supervised / semisupervised learning
		- What works for similar problems?

# Empirical Study
1. Data and Data Prepration
		1. Data Set
			- Describe interesting properties of the data set. How are values distributed?
			- What preprocessing have been applied. See [[@grauerOptionTradeClassification2022]]
		2. Feature Engineering
			- Previously not done due to use of simple rules only. 
			- Try different encondings e. g., of the spread.
		3. Train-Test Split
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