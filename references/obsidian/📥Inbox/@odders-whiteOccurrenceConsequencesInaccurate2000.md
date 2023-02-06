*title:* On the Occurrence and Consequences of Inaccurate Trade Classification
*authors:* Elizabeth R Odders-White
*year:* 1999
*tags:* #lr #result-evaluation #evaluation
*status:* #📦 
*related:*
- [[@leeInferringTradeDirection1991]] ()
- [[@leeInferringInvestorBehavior2000]] (this article is cited; performs similar study but with some flaws)
- [[@aitkenIntradayAnalysisProbability1995]] (this article is cited; similar study for the Australian Market)
- [[@ellisAccuracyTradeClassification2000]] (this article is cited; similar results)

## Notes 
- Her study evaluates the performance of the LR algorithm on the TORQ data set. Dataset contains 144 NYSE stocks from 1.11.1990 - 31.01.1991. 
- Dataset has been previously studied by [[@leeInferringInvestorBehavior2000]]. This study however filters out certain trades, where the LR algorithm performs esspecially poorly. Thus, the results are positively biased.
- She finds that the algorithm classifies 85 % of the transactions correctly, but systematically misclassifies transactions at the midpoint of the bid-ask spread, small transactions and transactions in large or frequently traded stocks.
- Quote rule misclassifies 9.1 % of the transactions and fails to classify 15.9 %. Tick method misclassifies 21.4 % and LR algorithm misclassifies 15.0 %. 
- Intrestingly she provides evidence on the biases in other academic work due to the flaws of the LR algorithm.
- Odders-White distinguishes between the two sources of missclassification, namely noise and bias. It's common in ML to decompose the error into these components (+ variance). If the probability of missclassification is the same for all types of trades, it's not problematic, as it will just add a random error to the data. If some trades are more likely to be misclassified than others, then misclassification will add a systematic error to the data and may ultimately bias the results.
- If the findings are consistent across partitions, researchers can be confident that the results are robust to misclassification bias. If results however change with the dimensions without any clear explanation in the domain studied, missclassification poses a problem. 
- Intrestingly she also recognizes that the data is not error-free. She writes "I am implicitly assuming that the data acurately represent the truth. While no data set is error free, the TORQ data are quite clean and I have not reason to suspect aht any non-random errors that could bias my results exist". -> No justification is given. What means quite?
- **Initiator:** 
	- There is no consistent definition of the iniator used in finance.
	- **Common view:** Initiator is a trader who demand immediate execution. Thus a trader who places a market order (or limit order at the opposite quote) are labeled as iniators. Traders placing limit orders are viewed as non-iniators. This definition was used in [[@leeInferringInvestorBehavior2000]] and is problematic with crossed market orders or when limit orders are matched with other limit orders or when market orders are stopped. See also discussion in [[@theissenTestAccuracyLee2000]].
	- **chronological view (hers):** "The iniator of the transaction is the investor (buyer or seller) who placed his or her order last, chronologically." As the it uses the notion of time. The *chronological* view can be applied when the immediacy defintion cannot. E. g. for limit orders the one who placed the limit order iniated the trade, which maintains consistency with market orders.
- **Results:**
	- [[@leeInferringInvestorBehavior2000]] achieved an accuracy rate of 93 %, but they eliminate a subsample where the accuracy dropped. Also the error is systematic.
	- She proposes to not just look at the accuracy as the single criteria for performance, but also investigate the performance in different groups e. g., if trade is inside the quote etc. 
	- She recognizes that blindly applying the LR algorithm can be wrong in certain situations and thus it's applicability has to be investiged for a specific application.
	- Zero-ticks are esspecially problematic, if the prior trade took place long ago.
- **Evaluation:**
	![[lr-odders-white.png]]
	(contains more tables that could be used in evaluation)
- **Conclusion:**
	- She recommends researchs to partition the transactions along the dimensions investigated in the paper and study the impact on the results. If the results can be maintained across partitions, then one can assume that results are robust to misclassification bias. Sudden changes indicate a problem with misclassification bias. If results vary across partitions results should be discussed along with the full sample results. How certain transactions are treated e. g., elimination as in [[@leeInferringInvestorBehavior2000]] differs from one study to another.
## Annotations

“This study uses the TORQ data to investigate the performance of the Lee and Ready (1991, Journal of Finance 46, 733}746.) trade classi"cation algorithm.” ([Odders-White, 2000, p. 259](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=1&annotation=N9FKIQ62))

“I "nd that the algorithm correctly classi"es 85% of the transactions in my sample, but systematically misclassi"es transactions at the midpoint of the bid}ask spread, small transactions, and transactions in large or frequently traded stocks.” ([Odders-White, 2000, p. 259](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=1&annotation=FHU2H7XD))

“The validity of many economic studies hinges on the ability to accurately classify trades as buyer or seller-initiated.” ([Odders-White, 2000, p. 259](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=1&annotation=HJD43PDE))

“Lee and Ready (1991) examined a pair of commonly used algorithms, namely the quote method and the tick method, which classify transactions based on execution prices and quotes. Lee and Ready then recommended that a combination of the two algorithms be used in practice (hereafter referred to as the Lee and Ready method).” ([Odders-White, 2000, p. 260](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=2&annotation=MBD6RIUY))

“If the probability of misclassi"cation is the same for all types of trades (e.g. large buys occurring the in the morning are as likely to be misclassi"ed as small sells occurring in the afternoon), then trade misclassi"cation will simply add random error to the data. If instead, particular types of transactions are more likely than others to be misclassi"ed, then trade misclassi"cation will add systematic error to the data and may ultimately bias the results.” ([Odders-White, 2000, p. 260](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=2&annotation=FDF6RIFZ))

“Using the TORQ (Trades, Orders, Reports, and Quotes) database from the NYSE, which makes the direct determination of the initiator of a transaction possible, I evaluate the overall performance of the Lee and Ready algorithms and examine the consequences of misclassi"cation.” ([Odders-White, 2000, p. 260](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=2&annotation=4LVXNLVZ))

“I "nd that the quote method misclassi"es 9.1% of the transactions in my sample and fails to classify 15.9% of the transactions. The tick method misclassi"es 21.4% of the transactions, and the combination recommended by Lee and Ready misclassi"es 15.0%. Moreover, transactions inside the bid}ask spread, small transactions, and transactions in large or frequently traded stocks are especially problematic” ([Odders-White, 2000, p. 260](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=2&annotation=S9LRNZUE))

“Second, although they also use the TORQ database, they focus on a smaller subset of the data” ([Odders-White, 2000, p. 261](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=3&annotation=IMJJ48D5))

“One way to describe initiators is as traders who demand immediate execution (hereafter, the immediacy de"nition). A natural consequence of this de"nition is that traders placing market orders (or limit orders at the opposite quote) are labeled the initiators, and traders placing limit orders are viewed as non-initiators or passive suppliers of liquidity” ([Odders-White, 2000, p. 261](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=3&annotation=HBVFMGBW))

“Problems with this de"nition arise, however, when market orders cross, when limit orders are matched with other limit orders, and when market orders are FINMAR=38=KGM=VVC=BG E.R. Odders-White / Journal of Financial Markets 3 (2000) 259}286 26” ([Odders-White, 2000, p. 261](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=3&annotation=GNP9AT73))

“Fig. 1. Sample transaction. stopped, all of which can occur frequently.” ([Odders-White, 2000, p. 262](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=4&annotation=DI3BBDUF))

“De5nition. The initiator of a transaction is the investor (buyer or seller) who placed his or her order last, chronologically.” ([Odders-White, 2000, p. 262](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=4&annotation=ATXL3H6L))

“The advantage of the chronological de"nition is that it can be applied when the immediacy de"nition cannot.” ([Odders-White, 2000, p. 262](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=4&annotation=47IDRFTE))

“First, they demonstrated that because updated quotes are often reported before the transactions that triggered them, a comparison of the execution price to the quotes in e!ect at the time of the transaction is inappropriate” ([Odders-White, 2000, p. 263](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=5&annotation=WGMCZZ5C))

“The disadvantage is that the tick method incorporates less information than the quote method since it does not use the posted quotes.” ([Odders-White, 2000, p. 264](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=6&annotation=9WE2MXZS))

“First, they noted that &the primary limitation of the tick test is its relative imprecision when compared to a quotebased approach'. This implies that the quote method should be employed whenever possible.” ([Odders-White, 2000, p. 264](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=6&annotation=YEJ36TA3))

“Lee and Ready recognized that these algorithms were imperfect, however, and emphasized the di$culty in truly evaluating their performance without data on the true trade classi"cation” ([Odders-White, 2000, p. 264](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=6&annotation=4HZKX5SV))

“he sample for this study comes from the TORQ database, which contains data on 144 NYSE stocks for the period from November 1, 1990 to January 31, FINMAR=38=KGM=VVC 264 E.R. Odders-White / Journal of Financial Markets 3 (2000) 259}28” ([Odders-White, 2000, p. 264](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=6&annotation=K5B5KQ2Q))

“2 For a description of the TORQ database, see Hasbrouck (1992). 1991. The TORQ data consist of transaction, quote, and order records for all orders placed through one of the automated routing systems, as well as audit trail data, providing information on the parties involved and other detailed information about the trades.” ([Odders-White, 2000, p. 265](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=7&annotation=HW22JAEI))

“Recall that Lee and Radhakrishna (1996) found a 93% accuracy rate for the Lee and Ready method. Their accuracy rate exceeds the 85% rate found here because the trades that they eliminate are more likely to be misclassi"ed by the algorithm.” ([Odders-White, 2000, p. 267](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=9&annotation=XDW4N7TA))

“For example, if the 50,000 transactions misclassi"ed by the Lee and Ready method constitute a representative cross-section of the entire sample, then the misclassi"cation will simply add noise to the data. In this case, the 85% accuracy rate is quite good.” ([Odders-White, 2000, p. 268](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=10&annotation=6GG2JDJU))

“If, on the other hand, the Lee and Ready method systematically misclassi"es certain types of transactions, a bias could result” ([Odders-White, 2000, p. 268](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=10&annotation=NJBY7LTC))

“I divide the sample into three groups: transactions that occurred at or outside the quotes, transactions that occurred at the spread midpoint, and transactions that occurred elsewhere inside the spread (not at the midpoint).” ([Odders-White, 2000, p. 268](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=10&annotation=FUZW9DMZ))

“The Chi-square statistic tests the hypothesis that the frequency of misclassi"cation is independent of price.” ([Odders-White, 2000, p. 269](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=11&annotation=2RUY3ZZZ))

“In fact, zero-tick trades are problematic in general because the prior trade is often an inappropriate benchmark. For example, if the prior trade took place long ago, it is &stale' and does not re#ect current market information.” ([Odders-White, 2000, p. 275](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=17&annotation=PFNHW7ML))

“If the "ndings are consistent across partitions, then researchers can be reasonably con"dent that their results are robust to misclassi"cation bias. On the other hand, if the results change along these dimensions without any clear explanation given the focus of the research, this suggests that misclassi"cation may be a problem. In this case, choices should clearly be guided by the goal of the study in question and the nature of the data.” ([Odders-White, 2000, p. 276](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=18&annotation=D3VLUTJ2))

“Because these data are used to determine the true initiator of each transaction, I am implicitly assuming that the data accurately represent the truth. While no data set is error free, the TORQ data are quite clean and I have no reason to suspect that any non-random errors that could bias my results exist.” ([Odders-White, 2000, p. 276](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=18&annotation=YKW8ZMSH))

“Evidence of the impact of inaccurate trade classi"cation on economic research is provided.” ([Odders-White, 2000, p. 280](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=22&annotation=I53QTFQJ))

“In light of this evidence, I recommend that researchers partition their transaction samples along the dimensions investigated in the paper and examine the impact on the results of their studies. If the "ndings are consistent across partitions, then researchers can be reasonably con"dent that their results are robust to misclassi"cation bias. On the other hand, if the results change along these dimensions without any clear explanation given the focus of the research, this suggests that misclassi"cation may be a problem. In this case, at a minimum, di!erences across partitions should be discussed along with the overall results.” ([Odders-White, 2000, p. 280](zotero://select/library/items/U8BCAAHY)) ([pdf](zotero://open-pdf/library/items/NXMYR8U5?page=22&annotation=6GHSPMUE))