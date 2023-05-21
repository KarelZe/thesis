
title: Inferring Trade Direction from Intraday Data
authors: Charles M. C. Lee, Mark J. Ready
year: 1991
tags : #trade-classification #lr #tick-rule #quote-rule #rule-based 
status : #📦 
related:
- [[@hasbrouckTradesQuotesInventories1988]]
- [[@holthausenEffectLargeBlock1987]]
- [[@ellisAccuracyTradeClassification2000]]
- [[@chakrabartyTradeClassificationAlgorithms2007]]
**code:**
- https://github.com/jblocher/sas_util/blob/master/LR_Trade_ID.sas


## Notes
- Most cited paper in journal of finance.
- Authors propose the LR algorithm to classify individual trades as market buys and sell orders using intraday trade / execution prices and quote data. The LR algorithm fuses two commonly used algorithms. Use quote rule in general (due to performance) but apply tick rule to trades at the midpoint spread.
- Authors show that the tick test performs alone performs remarkably well, if closer to the bid or ask, but so does the quote rule. Quote rule requires however the tick rule to assign midpoint trades.
- They identify the two problems with the common approach of comparing trade prices with quote prices at the time of trade. First, quotes are often recorded before the trade, that triggered them and second that trades are often inside the spread.
- To reduce the misclassifications they propose to use an offset of 5 sec. when comparing the trade and the quote. For trades inside the spread they suggest to use the tick test. They demonstrate that the effective spread is often on one side of the quoted spread due to standing order.
- The tick test is a techniques that infers the trade direction by comparing the trade price aginst the price of preceding trades. Four categories are possible: uptick, downtick, zero-uptick, and zero-downtick. Note that "zero-..." ticks can be problamtic, if trades didn't take place for a long time / are stale.
- Tick test incorporates less information than quote method, that utilises posted quotes instead. (found in [[@odders-whiteOccurrenceConsequencesInaccurate2000]])
- In theory all trades can be classified using the tick test. In practise certain trades are not classifiable due to being reported out-of-sequence or sold with special conditions. 
- Tick test is relatively imprecise when compared with the quote rule. This is esspecially evident if the prevailing quote has changed or if the quote is a long time back.
- An alternative is the reverse tick test ([[@hasbrouckTradesQuotesInventories1988]]), that compares the trade price against the prices of trades immediately following the trade. If the following price is higher, the reverse tick test classifies the current trade as a sell.
- I the trade is bracketed by price reversal (price change before the trade is the opposite to the price change after the trade) the reverse tick test and the tick test yield the same results.
![[tick-rule-reverse-tick-rule.png]]
- Authors study a sample of NYSE stocks. Authors do not know true labels. This limits their evaluation e. g. what is the true direction of a trade inside the spread. For trades at the ask 92.1 % are classified as buys using the tick test and 90.2 % at the bid are classified as sells. Thus there is a high degree of agreement between the tick test and quote rule, if the prevailing quote is unambiguous. Unambigous means that the quote reversion occured more than 5 sec before the trade. Quote reversions are generally triggered by trades.
- Trades and quotes can be out of their natural order depending on how they are entered into the system (Problem 1). Authors observe that quote reversions are clustered near the trade with a substantial portion of quote recorded ahead of the trade.  Authors suggest an adjustment. If the current quote is less than 5 sec old it was probably caused by the trade an so the previous quote should be used for classification. They note that a different delay might be appropriate for other markets and that the 5 sec rule was derived from the AMEX and NYSE sample.
- When a trade causes a quote reversion, the new quote tends to straddle the trade that triggered it. If new quotes are however recorded ahead of time, the current quote could cause a larger number of trades to appear inside the spread.
- Trading inside the spread (Problem 2) can happen due to the timing effects from above, but also due to standing market orders. As discussed in [[@hasbrouckTradesQuotesInventories1988]] trading inside the spread can happen if a market sell and market buy order arrive simultanousley or if a floor broker betters their qute. Solutions are neglecting them or use the quote rule which classifies based on the relative distance the ask or bid. Trades at the midpoint can however not be classified. 
- Given a mid-point trade on a down tick the next price change would likely be an up tick and vice versa (price reversal).
- Based on the observation that trades inside the spread are mainly due to standing orders and the price reversal patterns, they propose to classify with the tick test. 
- They provide some theoretical evidence why the tick test will classify at least 85 % of all trades at the midpoint of the spread correctly, albeit they have no true labels.

## Annotations


“This paper evaluates alternative methods for classifying individual trades as market buy or market sell orders using intraday trade and quote data.” ([Lee and Ready, 1991, p. 1](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=1&annotation=2RAXLL57))

“We then propose and test relatively simple procedures for improving trade classifications” ([Lee and Ready, 1991, p. 1](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=1&annotation=TQ78S73U))

“In this paper, we identify two serious potential problems with this method, namely, that quotes are often recorded ahead of the trade that triggered them, and that” ([Lee and Ready, 1991, p. 1](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=1&annotation=AUIJFYB8))

“trades are often (30%of the time) inside the spread” ([Lee and Ready, 1991, p. 2](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=2&annotation=L7SPPE42))

“We show that misclassificationscan be greatly reduced by comparing the trade to the quote in effect 5 seconds earlier. For trades inside of the spread, we provide evidence that the effective spread is often on one side of the quoted spread due to standing orders held by floor brokers. In these cases, we suggest that the “tick test” provides the best way to classify the trades as buys or sells. When only price data is available, we show that the “tick test” also performs remarkably well” ([Lee and Ready, 1991, p. 2](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=2&annotation=B29EE3HJ))

“The tick test is a technique which infers the direction of a trade by comparing its price to the price of the preceding trade@” ([Lee and Ready, 1991, p. 3](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=3&annotation=FBN84KX9))

“The test classifies each trade into four categories: an uptick, a downtick, a zero-uptick, and a zero-downtick. A trade is an uptick (downtick) if the price is higher (lower) than the price of the previous trade. When the price is the same as the previous trade (a zero tick), if the last price change was an uptick, then the trade is a zero-uptic” ([Lee and Ready, 1991, p. 3](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=3&annotation=4F3ZXZGT))

“A trade is classified as a buy if it occurs on an uptick or a zero-uptick;otherwise it is classified as a sell” ([Lee and Ready, 1991, p. 3](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=3&annotation=QC92KQHH))

“In theory, all trades can be classified as either a buy or a sell order by using a tick test.5In practise, certain trades are not classifiable because they are either reported out of sequence or are sold with special conditions attached” ([Lee and Ready, 1991, p. 3](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=3&annotation=96GXIDN8))

“The primary limitation of the tick test is its relative imprecision when compared to a quote-based approach, particularly if the prevailing quote has changed or it has been a long time since the last trade.” ([Lee and Ready, 1991, p. 3](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=3&annotation=C2QT5QSA))

“A possible alternative to the tick test is the “reverse tick test,” which classifies trades by comparing the trade price to prices of trades immediately” ([Lee and Ready, 1991, p. 3](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=3&annotation=8R8Q5Z6V))

“after the current trade. If the current trade is followed by a trade with a higher (lower)price, the reverse tick test classifies the current trade as a sell (buy). This method was used by Hasbrouck (1988) to classify trades at the midpoint of the bid-ask spread.” ([Lee and Ready, 1991, p. 4](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=4&annotation=ADTZ3VUB))

“The tick test and the reverse tick test yield the same classification when the current trade is bracketed by a price reversal (i.e., when the price change before the trade is the opposite of the price change after the trade).” ([Lee and Ready, 1991, p. 4](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=4&annotation=3R52GZRX))

“Thus, there is a high degree of agreement between the tick test and quote-based classifications when the identity ofthe prevailing quote is unambiguous.” ([Lee and Ready, 1991, p. 5](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=5&annotation=TBW25IEG))

“The quote revisions are clearly clustered near the trade, with a substantial portion (59.3percent) of the quotes recorded ahead of the trade” ([Lee and Ready, 1991, p. 6](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=6&annotation=ZHV8QUSS))

“The shape of the distribution suggests that these quote revisions were attributable to the trade in question.” ([Lee and Ready, 1991, p. 6](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=6&annotation=QP5UAWNL))

“These findings point to a data problem which will need to be addressed in future studies. The sharp drop in quotes between 5 and 6 seconds before the trade indicates that a simple procedure could mitigate this problem” ([Lee and Ready, 1991, p. 6](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=6&annotation=EJAYX35K))

“If the current quote is less than 5 seconds old, it was probably caused by the trade under consideration, so the previous quote should be used for classification” ([Lee and Ready, 1991, p. 6](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=6&annotation=CFQK4TXG))

“When a trade causes a quote revision, the new quote tends to straddle the trade that triggered it. If new quotes are recorded ahead of the trade, then naively using the current quote should cause a larger number of trades to appear inside the spread.” ([Lee and Ready, 1991, p. 6](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=6&annotation=CTX4C8SF))

“A different delay may be appropriate for other time periods. For example, tests we conducted using AMEX and NYSE data from September and October 1987 showed that during that period, most of the pretrade quotes occurred within 2 seconds of the trade” ([Lee and Ready, 1991, p. 6](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=6&annotation=Q89P84BM))

“Although some of the apparent trading inside the spread is actually due to the timing issue identified earlier, Table I1 indicates that 30 percent of all trades are inside the spread even after correcting the timing problem” ([Lee and Ready, 1991, p. 7](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=7&annotation=UR27L7CP))

“An alternative way to classify trades inside the spread, used by Harris (1989), is to call them buys (sells) if they are closer to the ask (bid). However, when the spread is an even number of eighths, trades at the midpoint of the spread will be unclassified” ([Lee and Ready, 1991, p. 8](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=8&annotation=4I53JF9M))

“In Subsection B a simple model is used to demonstrate the effectiveness of the tick test in classifying midpoint trades.” ([Lee and Ready, 1991, p. 8](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=8&annotation=4WR7T3RR))

“price change will be an increase or a decrease with equal probability. On the other hand, if there is an effective spread on one side of the quoted spread, we should observe reversals on midpoint trades.” ([Lee and Ready, 1991, p. 9](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=9&annotation=RIZP63IM))

“In other words, given a midpoint trade on a down (up) tick the next price change would likely be an up (down) tick” ([Lee and Ready, 1991, p. 9](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=9&annotation=7W3CWCSY))

“The evidence presented thus far implies trades inside the spread often arise as a result of standing orders” ([Lee and Ready, 1991, p. 10](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=10&annotation=NW32K2JD))

“Moreover, the pattern of price reversals reported in Figure 3 suggests it is possible to infer the direction of these standing orders by using the tick test. Empirically, it is difficult to quantif” ([Lee and Ready, 1991, p. 10](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=10&annotation=ZPUPS4V4))

“Inferring TradeDirection from Intraday Data 743 the expected improvement from using the tick test without knowing the actual direction of each trade inside the spread. However, the expected improvement can be evaluated analytically by means of a simple model” ([Lee and Ready, 1991, p. 11](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=11&annotation=BL6JCNG2))

“The tick test will only misclassify the second midpoint trade after the arrival of the standing order if it misclassifies the first midpoint trade and the second trade is in the same direction as the first trade (i.e., another buy).” ([Lee and Ready, 1991, p. 11](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=11&annotation=LUHKVG2B))

“In this paper, we show that the price-based trade classification method commonly known as the “tick test” provides remarkably accurate directional inferences.” ([Lee and Ready, 1991, p. 14](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=14&annotation=RM9ZI259))

“We also identify two potential problems with classifying trades as buys or sells using quoted spreads” ([Lee and Ready, 1991, p. 14](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=14&annotation=IWNRE3DU))

“n a sample of trades on the NYSE during 1988, more than half of the quote changes resulting from trades are recorded ahead of the trade” ([Lee and Ready, 1991, p. 14](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=14&annotation=W6A2E2NU))

“We show that the problem of quote identification can be mitigated by using a time-delayed quote which, in the case of 1988 data, is the quote in effect 5 seconds before the trade time stamp” ([Lee and Ready, 1991, p. 14](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=14&annotation=A733UX9Y))

“We present evidence that trading inside the spread is due largely to “standing orders” that cause the effective spread to be narrower than the quoted spread” ([Lee and Ready, 1991, p. 14](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=14&annotation=J2SNEGZM))

“For trades closer to the bid or ask we show that the tick test continues to perform well, although a simple assignment of trades as buys (sells),if they are closer to the bid (ask), will also perform well.” ([Lee and Ready, 1991, p. 14](zotero://select/library/items/FW283V5Z)) ([pdf](zotero://open-pdf/library/items/SVM9XEPW?page=14&annotation=PCR8DYSJ))