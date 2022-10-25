title: A local non-parametric model for trade sign inference
authors: Adam Blazejewski, Richard Coggins
year: 2005
tags: #supervised-learning  #trade-classification #knn #logistic-regression #majority-vote #moving-window
status : #📦 
related: 
- [[@ronenMachineLearningTrade2022]] (also use $k$-nn, logistic regression)
- [[@fedeniaMachineLearningCorporate2021]] (see above)
- [[@rosenthalModelingTradeDirection2012]] (also uses logistic regression)
- [[@aitkenIntradayAnalysisProbability1995]] (also use logistic regression; similar dataset)

## Notes
- The classify trades in the Australian stock market using linear logistic regression, trade continuation, and majority vote. 
- We show that the k-NN classifier is superior to the other classifiers and can separate buyer-initiated and seller-initiated trades in our data set with an average accuracy of over 71%. The best results are achieved for larger neighbourhoods of e. g., $5$ or $9$.
- They do not use quote and trade prices.
- Models are trained using a moving window approach.
- Authors conclude that non-linear approach may produce a more parsimonious trade sign inference model with a higher out-of-sample classification accuracy.
- Paper is poorly executed with regards to statistical tests and is impacted by computational constraints. Note that publication lays long way back.🚨

## Annotations
“The k-nearest neighbor with three predictor variables achieves an average out-of-sample classification accuracy of 71.40%, compared to 63.32% for the linear logistic regression with seven predictor variables.” ([Blazejewski and Coggins, 2005, p. 481](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=1&annotation=SKICD63H))

“The result suggests that a non-linear approach may produce a more parsimonious trade sign inference model with a higher out-of-sample classification accuracy.” ([Blazejewski and Coggins, 2005, p. 481](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=1&annotation=I9P2NWE9))

“A buyer-initiated trade (a buy) is a trade triggered by a buy market order matched against one or more sell limit orders in the order book. The opposite holds for a seller-initiated trade (a sell), where a sell market order is matched against one or more buy limit orders in the order book. Submitters of market orders are called liquidity demanders, while submitters of limit orders stored in the book are called liquidity providers.” ([Blazejewski and Coggins, 2005, p. 482](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=2&annotation=T5G3YPWR))

“The trade initiator variable is alternatively referred to as a trade sign, trade direction, trade indicator, or buy/sell indicator. We will use the second term, trade sign, throughout the rest of this paper.” ([Blazejewski and Coggins, 2005, p. 482](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=2&annotation=8GIXTWB2))

“Aitken et al. analyze the intraday probability of trading at the asking price on the Australian Stock Exchange. They use limit order book and other data to build a logistic regression model for a set of over 3 million trades, and manage to correctly classify 53.3% of trades, while 51.58% of all trades in their data set are at the asking price.” ([Blazejewski and Coggins, 2005, p. 483](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=3&annotation=WLIDEQVF))

“Our study explores a regularity in market order submission strategies on the Australian Stock Exchange (ASX).” ([Blazejewski and Coggins, 2005, p. 483](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=3&annotation=9ZV28XP2))

“We demonstrate this predictability by developing an empirical trade sign inference model to classify trades into buyer-initiated and sellerinitiated. The model is based on a local non-parametric method, k-nearest-neighbor (k-NN).” ([Blazejewski and Coggins, 2005, p. 483](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=3&annotation=PS2PBQNE))

“Quote and trade prices are not used.” ([Blazejewski and Coggins, 2005, p. 483](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=3&annotation=L63QG24H))

“Classification accuracy is determined through out-of-sample testing. The classification performance of the kNN classifier is compared against the performance of three other classifiers: linear logistic regression, trade continuation, and majority vote. We show that the k-NN classifier is superior to the other classifiers and can separate buyer-initiated and seller-initiated trades in our data set with an average accuracy of over 71%.” ([Blazejewski and Coggins, 2005, p. 483](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=3&annotation=YYXYVB7V))

“Variable sets are ranked by classification accuracy across all stocks, and the best sets are selected for the logistic regression and the knearest-neighbor.” ([Blazejewski and Coggins, 2005, p. 484](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=4&annotation=8JRPKCHM))

“Two simple classifiers, a trade continuation and a majority vote, based on lagged values of the trade sign only, are used for performance comparison. The models are estimated and tested with a moving window method.” ([Blazejewski and Coggins, 2005, p. 484](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=4&annotation=VMNGXM3Y))

“Before model estimation all predictor variables are preprocessed by calculating their natural logarithms.” ([Blazejewski and Coggins, 2005, p. 486](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=6&annotation=Y7B9N9VT))

“The result of the testing procedure is a single classification accuracy value for the test day. The estimation and testing are then repeated for a new pair of training and test intervals, obtained by shifting the previous pair of intervals one day forward.” ([Blazejewski and Coggins, 2005, p. 486](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=6&annotation=4UIH4QEN))

“Statistical significance was not determined because it is not used by our selection procedure” ([Blazejewski and Coggins, 2005, p. 489](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=9&annotation=58YRHEMA))

“Depending on the value of k; the k-nearest-neighbor has the mean accuracy approximately 9% to 11% higher than the logistic regression, while its standard deviation varies from 2.89% to 3.28%. Furthermore, the higher the value of k the better the performance of the k-NN classifier, even though an improvement between k ¼ 5 and 9 is minimal.” ([Blazejewski and Coggins, 2005, p. 490](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=10&annotation=L4WRY4G5))

“(1) Among the k-NN classifiers, the higher the value of k the greater the mean accuracy. The difference between accuracies for k ¼ 9 and 5, however, can be minimal and sometimes negative, but on average k ¼ 9 is the best (12). (2) The mean accuracy of the k-NN classifier, where k ¼ 9; is a monotonically increasing function of the training interval length. The rate of the increase, however, rapidly declines. Small, negligible fluctuations are sometimes present (10). (3) The mean accuracy of the k-NN classifier, where k ¼ 9; is greater than the mean accuracy of the logistic regression classifier for all training timescales (8).” ([Blazejewski and Coggins, 2005, p. 491](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=11&annotation=PEISTE82))

“The mean accuracy of the k-NN classifier, where k ¼ 9; is greater than the mean accuracies of the trade continuation and the majority vote classifiers, for all training timescales (12). The total number of models constructed for each stock was 145: 2 3 16 k-NN, 2 16 logistic regression, 1 trade continuation, and 16 majority vote models. ” ([Blazejewski and Coggins, 2005, p. 493](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=13&annotation=TTJGZ7EW))

“the k-nearest-neighbor classifier as an alternative to the linear logistic regression. The average classification accuracy of the k-NN ðk ¼ 9Þ classifier, across all stocks and allowing contemporaneous predictor variables, was found to be 71.40% (SD ¼ 4:01%), or 8.08% higher than the corresponding accuracy of 63.32% (SD ¼ 4:27%) for the logistic regression. When compared with the trade continuation and the majority vote classifiers, the k-nearest-neighbor was 14.08% and 17.87% better, respectively.” ([Blazejewski and Coggins, 2005, p. 494](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=14&annotation=NQD5BK7G))

“These results suggest that a non-linear approach may produce a more parsimonious trade sign inference model with a higher out-of-sample classification accuracy. Furthermore, for most of our stocks the classification accuracy of the k-nearest-neighbor ðk ¼ 9Þ with contemporaneous predictor variables is a monotonically increasing function of the training interval length, with 30 days being the best interval.” ([Blazejewski and Coggins, 2005, p. 494](zotero://select/library/items/ULRH88UK)) ([pdf](zotero://open-pdf/library/items/2KMK55IH?page=14&annotation=MCA94DNA))