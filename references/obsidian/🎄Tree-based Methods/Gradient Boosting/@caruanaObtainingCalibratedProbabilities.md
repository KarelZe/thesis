
title: Obtaining Calibrated Probabilities from Boosting
authors: Alexandru Niculescu-Mizil Rich Caruana
year: 

“In a recent evaluation of learning algorithms (Caruana & Niculescu-Mizil 2006), boosted decision trees had excellent performance on metrics such as accuracy, lift, area under the ROC curve, average precision, and precision/recall break even point. However, boosted decision trees had poor squared error and cross-entropy because AdaBoost does not produce good probability estimates.” (Caruana, p. 28)

“Friedman, Hastie, and Tibshirani (2000) provide an explanation for why boosting makes poorly calibrated predictions. They show that boosting can be viewed as an additive logistic regression model. A consequence of this is that the predictions made by boosting are trying to fit a logit of the true probabilities, as opposed to the true probabilities themselves. To get back the probabilities, the logit transformation must be inverted.” (Caruana, p. 28)

“A consequence of this is that the predictions made by boosting are trying to fit a logit of the true probabilities, as opposed to the true probabilities themselves.” (Caruana, p. 28)
