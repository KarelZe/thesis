*title:* A Test of the Accuracy of the Lee/Ready Trade Classification Algorithm
*authors:* Erik Theissen
*year:* 1999
*tags:* #trade-classification #stocks #lr 
*status:* #📥
*related:*
- [[@leeInferringTradeDirection1991]]
- [[@odders-whiteOccurrenceConsequencesInaccurate2000]]
- [[@ellisAccuracyTradeClassification2000]]
# Notes 
- As tick test requires only transaction data, data requirements are low.
- The extent to which inaccurate trade classification biases empirical research dependes on whether misclassifications occur randomly or systematically.
- There are different views on buyer-initiated. Either by the one who placed the buy order last (view of [[@odders-whiteOccurrenceConsequencesInaccurate2000]]) or based on the action of the makler (view of [[@ellisAccuracyTradeClassification2000]]).
- Assessing the performance requires true labels, which are often only available in some, proprietary results.
- The percentage of transactions classified correctly by the tick test is significantly lower than for the LR method.
- The fraction of correctly classified trades appears to be higher for more liquid stocks.
- Performance of tick test deterioriates for transactions on a zero tick. For zero tick trades the the correct classification rates are as low as 52.6 %.
- In the german stock market, where the empirical study was conducted, the application of the LR method is limited.
# Annotations
“As only transaction data is needed (for the tick test), data requirements for the application of the tick test are low” ([Theissen, 2000, p. 1](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=2&annotation=IATU5TDV))

“Aitken / Frino (1996) use data from the Australian Stock Exchange, an electronic open limit order book. They find that the tick rule classifies only 74% of the transactions correctly. Odders-White (1999) analyzes all steps of the algorithm using the TORQ database provided by the New York Stock Exchange. She finds that the algorithm on average correctly classifies 85% of the transactions. Ellis / Michaely / O’Hara (1999) provide an analysis of trade classification accuracy using NASDAQ data. In their sample, 81.4% of the transactions are classified correctly” ([Theissen, 2000, p. 2](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=3&annotation=BPWVC7SF))

“The extent to which inaccurate trade classification biases the results of empirical research partly depends on whether the misclassifications occur randomly or follow a systematic pattern. There are two ways to address this issue.” ([Theissen, 2000, p. 2](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=3&annotation=63B97J79))

“Ellis / Michaely / O’Hara (1999) estimate a logit model and find that the proximity of the transaction price to the quotes is the most important determinant of the probability of misclassification. The second method, chosen by Lightfoot et al. (1999) and Odders-White (1999), entails estimating empirical model” ([Theissen, 2000, p. 2](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=3&annotation=WQ6PTN7D))

“3 using the inferred trade classifications and comparing the results to those obtained when using the true classifications instead.” ([Theissen, 2000, p. 3](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=4&annotation=7NT3MGW3))

“Second, the present paper uses a different definition of the true trade classification than both Odders-White (1999) and Lightfoot et al. (1999). These authors consider a transaction to be buyer-initiated (seller-initated) if the buy order (sell order) was placed last, chronologically. In contrast, we use a definition based on the position taken by the Makler (the equivalent of the specialist). If the Makler sold bought shares the transaction is classified as being buyer-initiated seller-initiated. This is similar to the approach in Ellis / Michaely / O’Hara (1999).” ([Theissen, 2000, p. 3](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=4&annotation=Y34IZPYB))

“The true classification is based on the position taken by the Makler. We define a trade to be buyer-initiated when the Makler sold shares and to be seller-initiated when the Makler bought shares” ([Theissen, 2000, p. 7](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=8&annotation=DNSGZDRL))

“The percentage of transactions classified correctly by the tick test is only slightly lower than the corresponding percentage for the Lee / Ready method (72.22% compared to 72.77%).” ([Theissen, 2000, p. 8](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=9&annotation=QDKKVWJL))

“The fraction of correctly classified trades appears to be higher for more liquid stocks.” ([Theissen, 2000, p. 8](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=9&annotation=HMPK5LN2))

“The performance of the tick test deteriorates dramatically when transactions occurring on a zero tick are considered. For these trades, the classification obtained when using the tick test appears to be unrelated to the true classification. The percentage of correct classifications is only 52.6% and is not significantly different from 50%” ([Theissen, 2000, p. 10](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=11&annotation=Y5669YUH))

“Overall, the results indicate that, at least for the German stock market, the accuracy of the Lee / Ready trade classification method is limited. The misclassification probability of 27.23% is higher than the corresponding percentages reported for the NYSE and NASDAQ” ([Theissen, 2000, p. 10](zotero://select/library/items/ESEIBAMC)) ([pdf](zotero://open-pdf/library/items/2XMIU8NA?page=11&annotation=34PK6HSV))