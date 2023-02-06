
title: An Introduction to Statistical Learning
authors: Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani
year: 2012
tags :  #supervised-learning #gbm #decision #gbm 
status : #📦 
related: 
- [[@friedmanGreedyFunctionApproximation2001]]
- [[@hastietrevorElementsStatisticalLearning2009]]

## Boosting

- Boosting approaches learn slowly. Instead of fitting trees hard to the data, the  model is fit to the residuals of the model. A new decission tree is added into the fitted function to update the residuals. (James et. al; 2013)
- The learning rate slows the learning process down, allowing more and ifferent shaped trees to attack the residuals.
- Unlike bagging, the construction of trees is **dependent** on trees already grown.

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