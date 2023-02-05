
title: Obtaining Calibrated Probabilities from Boosting
authors: Alexandru Niculescu-Mizil Rich Caruana
year: 2012
tags :  #supervised-learning #gbm #decision #gbm #probabilistic-classification
status : #📦 
related: 
- [[🎄Tree-based Methods/Gradient Boosting/@friedmanGreedyFunctionApproximation2001]]
- [[@hastietrevorElementsStatisticalLearning2009]]
- [[@prokhorenkovaCatBoostUnbiasedBoosting2018]]

## Notes
- Boosted decision trees achieve a high accuracy but poor probability estimates.
- According to [[@hastietrevorElementsStatisticalLearning2009]] the reason for poorly calibrated predictions lies the fact that boosting tries to fit a logit of the true probabilities instead of the true probabilities themselves. Obtaining the true probabilities thus require a transformation back.

## Annotations
“In a recent evaluation of learning algorithms (Caruana & Niculescu-Mizil 2006), boosted decision trees had excellent performance on metrics such as accuracy, lift, area under the ROC curve, average precision, and precision/recall break even point. However, boosted decision trees had poor squared error and cross-entropy because AdaBoost does not produce good probability estimates.” ([Caruana, p. 28](zotero://select/library/items/ZGPCNYSL)) ([pdf](zotero://open-pdf/library/items/RA8HXVK6?page=1&annotation=MD8ZQKRE))

“Friedman, Hastie, and Tibshirani (2000) provide an explanation for why boosting makes poorly calibrated predictions. They show that boosting can be viewed as an additive logistic regression model. A consequence of this is that the predictions made by boosting are trying to fit a logit of the true probabilities, as opposed to the true probabilities themselves. To get back the probabilities, the logit transformation must be inverted.” ([Caruana, p. 28](zotero://select/library/items/ZGPCNYSL)) ([pdf](zotero://open-pdf/library/items/RA8HXVK6?page=1&annotation=WS7XC6XI))