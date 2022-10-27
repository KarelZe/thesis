
title: Cross-validation: what does it estimate and how well does it do it?
authors: Stephen Bates, Trevor Hastie, Robert Tibshirani
year: 2022
tags :  #supervised-learning #cross-validation #nested-cv
status : #📦 
related: 
- [[@banachewiczKaggleBookData2022]] (recommended this paper)
- [[@hastietrevorElementsStatisticalLearning2009]]

## Notes
- The result of the CV is not the accuracy of the model fit on the data but rather the average accuracy over hypthetical many data sets.
- The prediction error can be under estimated, as correlations between the error estimates in different folds are not taken into account. Points are however part of both in the training and test set. 
- In consequence the estimates for the variance are too small and intervals of the prediction error are too narrow.
- Authors suggest nested cross-validation

## Annotations
“In this work, we show that the the estimand of CV is not the accuracy of the model fit on the data at hand, but is instead the average accuracy over many hypothetical data sets.” ([Bates et al., 2022, p. 1](zotero://select/library/items/WUNDECNK)) ([pdf](zotero://open-pdf/library/items/R2B9JY8T?page=1&annotation=U2UEVUPI))

“Turning to confidence intervals for prediction error, we show that na ̈ıve intervals based on CV can fail badly, giving coverage far below the nominal level” ([Bates et al., 2022, p. 1](zotero://select/library/items/WUNDECNK)) ([pdf](zotero://open-pdf/library/items/R2B9JY8T?page=1&annotation=L69EDACZ))

“The source of this behavior is the estimation of the variance used to compute the width of the interval: it does not account for the correlation between the error estimates in different folds, which arises because each data point is used for both training and testing.” ([Bates et al., 2022, p. 1](zotero://select/library/items/WUNDECNK)) ([pdf](zotero://open-pdf/library/items/R2B9JY8T?page=1&annotation=S6BBE6TT))

“As a result, the estimate of variance is too small and the intervals are too narrow. To address this issue, we develop a modification of cross-validation, nested cross-validation (NCV), that achieves coverage near the nominal level, even in challenging cases where the usual cross-validation intervals have miscoverage rates two to three times larger than the nominal rate.” ([Bates et al., 2022, p. 1](zotero://select/library/items/WUNDECNK)) ([pdf](zotero://open-pdf/library/items/R2B9JY8T?page=1&annotation=9KNZEXN3))