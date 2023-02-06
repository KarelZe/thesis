
title: A Direct Test of Methods for Inferring Trade Direction from Intra-Day Data
authors: Thomas J. Finucane
year: 2000
tags : #trade-classification #tick-rule #lr #accuracy
status : #📦 
related:
- [[@leeInferringTradeDirection1991]] 

## Notes
- Author implements tick and quote rule as well as LR algorithm and assess the performance on the TORQ data set.
- **Problems of LR:** LR can not handle the simultanous arrival of market buy and sell orders. Thus, one side will always be wrongly classified. Equally, crossed limit orders are not handled correctly as both sides iniate a trade independent of each other.
- Interesting observations in their results:
	- One fourth of all trades are no market orders
	- Relative location to the quoted spread is relevant for the accuracy. The accuracy deterioriates for mid-spread trades.
	- Trick rule is only slightly worse than the LR algorithm at quoted bid and ask.
	- For both tick rule and LR algorithm's accuracy is not 100 % accurate for trades at the bid or ask.
	- The tick test to classify trades will achieve results that are close to the results that can be achieved using quote-based methods and, in at least some applications, the tick test may provide more accurate measures than quote-based methods.
- Author uses a logit model to indentify important features that affect the classification accuracy.

## Annotations

“The most commonly used methods of inferring trade direction are the tick tests, which use changes in trade prices to infer direction; the quote method, which infers trade direction by comparing trade prices to quote” ([Finucane, 2000, p. 554](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=3&annotation=B487338J))

“Other researchers, including Hasbrouck ((1991), (1993)), Hausman, Lo, and MacKinlay (1992), Foster and Viswanathan (1993), Hasbrouck and Sofianos (1993), and Harris, Mclnish, and Chakravarty (1995), use the quote method to sign trades: trades above the quoted bid-ask midpoint are classified as buys, trades below the midpoint are classified as sells, and trades at the midpoint are either omitted or classified as trades with indeterminate directio” ([Finucane, 2000, p. 554](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=3&annotation=6GL5AUXG))

“The tick test classifies trades using previous trade prices to infer trade direction. If the trade occurs at a higher price than the previous trade (an uptick), the trade is classified as a buy. If the trade occurs at a lower price than the previous trade (a downtick) it is classified as a sell. When the price change between trades is zero (a zero tick), the trade is classified using the last price that differs from the current price. The reverse tick test is similar, but uses the next trade price to classify the current trade. If the next trade occurs on an uptick or zero uptick, the current trade is classified as a sell. If the next trade occurs on a downtick or zero downtick, the current trade is classified as a buy.” ([Finucane, 2000, p. 557](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=6&annotation=QECAH6VU))

“A limitation of the quote method, which classifies trades above the midpoint of the spread as buys and trades below the midpoint as sells, is that trades that oc? cur at the midpoint of the quoted spread cannot be classifi” ([Finucane, 2000, p. 557](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=6&annotation=L3DGFE8C))

“Figure 1 illustrates how trades can be misclassified when quotes change. When quotes rise between trades, sales at the bid on upticks and zero upticks will be misclassified as buys by the tick test, but should be correctly classified using quote-based methods. Trade 6a in Panel A of Figure 1 illustrates the case of a sell (at the bid) being misclassified by the tick test on a zero uptick. If quotes are falling, as is the case for trade 6b in Panel B, buys at the ask on downticks and zero downticks will be misclassified as sells by the tick test, and should be correctly classified by quote-based methods. Figure 1 also illustrates the tendency for mid-spread trades to be misclassified by the tick test, even when quotes do not” ([Finucane, 2000, p. 557](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=6&annotation=HHGX5BZ9))

“change. Trade 4a, a sale that occurs on an uptick, is misclassified as a buy, and trade 5b, a buy that occurs on a downtick with constant quotes, is misclassified as a sell. FIGURE 1 lllustrative Trade Sequences for Mid-Spread Trades Panel A - - - Ask Bid ?\*? - ?X? - Sell Buy Sell Sell Cross Sell (1a) (2a) (3a) (4a) (5a) (6a) Panel B Bid ?\*? ?\*? - - - - Bid Sell Sell Buy Buy Buy Buy (1b) (2b) (3b) (4b) (5b) (6b) Solid lines represent ask and bid prices and Xs denote trades. Trade direction is indicated below each trade. Panel A illustrates the case where sells on upticks and zero upticks are misclassified by the tick test when quotes are increasing. Panel B illustrates the case where buys on downticks and zero downticks are misclassified when quotes are falling. Panel B also demonstrates how m” ([Finucane, 2000, p. 558](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=7&annotation=YHNR29NJ))

“LR assert that the simultaneous arrival of market buy and sell orders that are executed as a cross is an "extremely rare occurrence," but their data does not permit them to empirically verify this assertion. To the extent that such trades are present, the accuracy of all classification methods will be reduced; one side of the trade will always be incorrectly classified” ([Finucane, 2000, p. 558](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=7&annotation=T777NSTJ))

“Additionally, trade direction may not always be unambiguously determined. While LR assume that trades generally occur only when a market buy or sell order arrives, trades that do not involve market orders also can occur, such as when two limit orders are crossed. Although the trade can be classified by the tick test or LR's algorithm, the true direction of the trade is ambiguous. Classifying such trades as buys or sells may lead to erroneous conclusions in empirical studies.” ([Finucane, 2000, p. 559](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=8&annotation=FPXC9PYD))

“The data for this study is extracted from the NYSE's TORQ (trades, orders, quotes) database. The TORQ database covers a representative sample of 144 firms for the three-month period November 1990 through January 1991” ([Finucane, 2000, p. 559](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=8&annotation=V7KYTW6Y))

“Contrary to what is typically assumed, nearly one-fourth of all trades do not occur as the result of the arrival of market order” ([Finucane, 2000, p. 560](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=9&annotation=FUNQXPXK))

“An analysis of the 75.2% of the trades that do contain market orders shows that the orders on the opposite side of the trade are nearly equally split between system side limit orders and crowd side market order” ([Finucane, 2000, p. 560](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=9&annotation=YKL8ITYZ))

“hen the sample trades are classified by their location relative to the quoted spread, it becomes apparent that the accuracy of the tick test (and LR's algorithm that uses the tick test to classify mid-spread trades) for mid-spread trades is far lower than the 85% predicted by LR's model. T” ([Finucane, 2000, p. 562](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=11&annotation=ACWUUJ3C))

“Contrary to expectations, the performance of the tick test is only marginally worse than LR's algorithm for trades that occur at the quoted bid or ask.” ([Finucane, 2000, p. 562](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=11&annotation=PSL3G9PA))

“Further? more, LR's algorithm is less than 100% accurate for trades at the bid or ask; both methods incorrectly classify at least 10% of the trades at the bid or ask” ([Finucane, 2000, p. 562](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=11&annotation=922L2YI4))

“y to affect classification accuracy. To further explore the importance of zero ticks, the sample is divided into trades on zero and non-zero ticks. Table 4 shows that all three methods perform worse for trades on zero ticks than on non-zero ticks, but it also shows that trades on zero ticks are far more likely to be mid-spread trades, crosses, or trades that occur on quote changes, supporting the hypothesis that zero ticks proxy for other factors” ([Finucane, 2000, p. 563](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=12&annotation=37UHF8E3))

“f zero ticks are causing the classification errors for the tick test, the same errors should not be found when LR's method is applied to trades at the quoted bid or ask on zero ticks; the trade direction predicted by LR's method is independent of previous trade price movements for trades at the bid or ask. The results of the tests in column 5 do not support the hypothesis that zero ticks per se are responsible for classification error” ([Finucane, 2000, p. 563](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=12&annotation=NIAKKD9Y))

“Since it is clear that the factors affecting the accuracy of the classification algorithms are not independent and that each factor may affect the accuracy of LR's algorithm and the tick test differently, it is useful to consider the marginal” ([Finucane, 2000, p. 563](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=12&annotation=TFYBKLWL))

“impact of the factors on classification accuracy for the two algorithms. This is accomplished by estimating logit models for each of the algorithm” ([Finucane, 2000, p. 565](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=14&annotation=XM93MWAN))

“Furthermore, price improvement will tend to lead to classification errors, since buys will occur away from the ask and sells will occur away from the bid. When trades receive price improvement, buys will also be more likely to occur on downticks and sells will be more likely to occur on upticks. I” ([Finucane, 2000, p. 565](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=14&annotation=T573QMUX))

“Table 5 contains the maximum likelihood coefficient estimates and asso? ciated x2-statistics for the two models, together with estimates of the marginal change in the probability of correctly classifying an observation for a one unit change in each independent variable.” ([Finucane, 2000, p. 566](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=15&annotation=MH5DIL3R))

“show that efforts to filter data in an attempt to increase classification accuracy may further exacerbate these biases. Somewhat surprisingly, although the classification error rates are slightly smaller for LR's method than for the tick test, the biases for estimated effective spreads and signed volume are smaller for the tick test than for LR's method. These findings sug? gest that researchers using the tick test to classify trades will achieve results that are close to the results that can be achieved using quote-based methods and, in at least some applications, the tick test may provide more accurate measures than quote-based methods.” ([Finucane, 2000, p. 574](zotero://select/library/items/KKJY6E7W)) ([pdf](zotero://open-pdf/library/items/RQ8KUGBP?page=23&annotation=GZHPZHDJ))