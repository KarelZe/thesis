---
title: Hyperparameter Tuning with Python
authors: Louis Owen
year: 2021
---

“To sign option trades as buys and sells, researchers often employ stock trade classification rules including the quote, the Lee and Ready (1991), the Ellis, Michaely, and O’Hara (2000), and the tick methods” (Savickas and Wilson, 2003, p. 881)

“We find that the components of index option complex trades not executed on the Retail Automated Execution System are misclassified almost 50% of the time by any method. The elimination of these trades (15% of the sample) results in a success rate of over 87% for the quote rule.” (Savickas and Wilson, 2003, p. 881)

“The correct classification rate varies inversely with a rule’s reliance on the past transaction price: 59%, 78%, 80%, and 83% for the tick, EMO, LR, and quote rules, respectively” (Savickas and Wilson, 2003, p. 882)

“the four rules exhibit lower (up to 24 percentage points) classification precision with options data than with stock data” (Savickas and Wilson, 2003, p. 882)

“We use two datasets covering the period July 3, 1995December 31, 1995; both are provided by CBOE.” (Savickas and Wilson, 2003, p. 882)

“Each record in the matched trade dataset reports the following information: date and execution time of a trade, trade size, price, the option’s parameters (put/call, expiration month and year, and strike price), the underlying security,” (Savickas and Wilson, 2003, p. 883)

“the reporting trader’s type code, and a dummy variable that indicates whether the reporting trader bought or sold the option. Thus, there is one record for each trading party for all trades on CBOE.” (Savickas and Wilson, 2003, p. 884)

“We group all records in the matched trade dataset into pairs, one pair per trade. The two records in each pair have the same trade date, underlying security, option’s parameters, trade size, and trade price. The buy/sell indicators are different for the two records in each pair (a buy and a sell).” (Savickas and Wilson, 2003, p. 884)

“onsistent with previous literature, we assume that all MC and BC trades are initiated by the customer, all MB trades are initiated by the broker, all MF and BF trades are initiated by the firm, and all MN and BN trades are initiated by the non-member.4” (Savickas and Wilson, 2003, p. 885)

“All trades and quotes for which the underlying asset price is reported as zero and all cancelled trades are excluded from the sample.” (Savickas and Wilson, 2003, p. 885)

“According to the quote rule, a trade is buyer- (seller-) initiated if it occurs above (below) the midpoint of the bid-ask spread. Trades that occur exactly at the midspread cannot be classified. The tick rule, on the other hand, classifies any trade as a buy (sell) if its price is above (below) the most recent transaction price that is different from the price of the transaction being classified. The LR and EMO methods combine the quote and tick rules to various degrees. The LR approach uses the tick rule to classify only at-midspread trades and the quote rule for all the other trades. The EMO method uses the quote rule for at-quote trades and the tick rule for all the other trades.” (Savickas and Wilson, 2003, p. 885)

“Each of the four classification methods cannot classify some trades. The quote rule is unable to classify midspread trades. The tick rule cannot classify the” (Savickas and Wilson, 2003, p. 885)

“first trade of the day for a given security. If the second trade of the day takes place at the same price as the first one, both trades cannot be classified by the tick rule. The LR and EMO rules also lose some trades since they are combinations of the tick and the quote rules.” (Savickas and Wilson, 2003, p. 886)

“The degree of success varies inversely with a rule’s reliance on past transaction prices: 82.78% for the quote rule, 80.11% for LR, 76.54% for EMO, and 59.4% for the tick method.” (Savickas and Wilson, 2003, p. 886)

“To address this issue, we form a common sample in which every trade can be classified by each of the four rules.” (Savickas and Wilson, 2003, p. 887)

“For each of the four classification rules, we use eight variables that may potentially explain observed performance. These variables are: the trade size,6 time from the previous trade, time from the previous quote, the moneyness ratio, trading days to maturity, the absolute value of the relative change in the underlying price, the location of trade relative to the bid-ask quotes, and whether a trade is executed on RAES.” (Savickas and Wilson, 2003, p. 887)

“trades inside the quotes are more likely to be misclassified.” (Savickas and Wilson, 2003, p. 888)

“On the one hand, we would expect that the greater (smaller) the transaction price relative to the midspread, the more likely that the transaction is a buy (sell) and occurs on an uptick (a downtick), implying higher classification success for outside-quote trades, especially for large trades in which the trade initiator is willing to pay a premium for the execution of his large order.” (Savickas and Wilson, 2003, p. 888)

“Trade size affects classification precision in two related ways. First, as found by EMO (2000), small trades are more likely to be executed at the quotes, implying an inverse relation between classification precision and trade size.” (Savickas and Wilson, 2003, p. 888)

“The longer the time from the previous trade, the less relevant is the information in the previous trade price, adversely affecting tick-based classifications.” (Savickas and Wilson, 2003, p. 888)

“First, as in the case of the past price, the older the quotes, the less relevant the information they convey, making quote-based classification less precise. Second, the older the quote, the greater underlying asset price change since that quote, which can enhance quote-based classification as described in the next paragraph.” (Savickas and Wilson, 2003, p. 888)

“Option moneyness has two indirect effects on classification precision. First, deep in-the-money options have deltas close to one (in absolute value) and, therefore, are more sensitive to the underlying asset price changes. Second, there is a positive relation between option moneyness and trade size” (Savickas and Wilson, 2003, p. 889)

“Time to maturity also has an indirect effect on classification precision. Trades in options with longer maturity tend to be smaller, resulting in negative correlation between the effects of maturity and of trade size.” (Savickas and Wilson, 2003, p. 889)

“Specifically, one of the most noticeable regularities is that smaller trades are classified more precisely. This is because these trades are more likely to be executed at quotes and are less prone to reversed-quote trading (partially due to the fact that many small trades are executed on RAES)” (Savickas and Wilson, 2003, p. 889)

“Thus, there seems to be a direct relation between using the quote information and successful option trade classification. This relation implies that the information conveyed by the past transaction prices is less relevant than information contained in quotes.” (Savickas and Wilson, 2003, p. 891)

“To determine the marginal impacts of individual variables, we estimate the following logit model, ProbabilityYi 1 1 ProbabilityYi 1 + 11 k1 k Xk i + i i 1N (1) where N is the total number of trades in the sample; Yi1 if a trade is misclassified and Yi 0otherwise;X1 is the size of the trade in contracts; X2 1ifthetrade occurs at the bid-ask spread midpoint and X2 0otherwise;X3 1ifthetrade occurs at the quotes and X3 0otherwise;X4 1 if the trade occurs outside the quotes and X4 0otherwise;X5 is the time from the previous trade in seconds; X6 is the time from the previous quote in seconds; X7 is the moneyness ratio  X8 is the absolute value of the relative underlying change since the last” (Savickas and Wilson, 2003, p. 892)

“Savickas and Wilson 893 quotes; X9 is the time to expiration in trading days; X10 0 if it is a call option and X10 1 if it is a put; X11 1 if the trade is executed on RAES and X11 0 otherwise.” (Savickas and Wilson, 2003, p. 893)

“The most economically significant variables in all four regressions are the trade location relative to quotes (including the RAES dummy), the absolute value of the relative underlying price change, and the put/call dummy.” (Savickas and Wilson, 2003, p. 893)

“s Table 4 indicates, larger trades have a higher probability of being traded on reversed quotes, as do shorter-term options. When the underlying price change is controlled for, we observe a positive relation between option moneyness and the probability of an RQ trade.” (Savickas and Wilson, 2003, p. 895)

“hese trades are correctly classified by the quote rule and are misclassified by our “true” classification, which assumes that all MC trades are customer-initiated. This suggests that the overall misclassification rate for quotebased rules may in fact be substantially lower than that reported in Table 1. Another factor not included in the regressions above is related to corporate dividend capture.” (Savickas and Wilson, 2003, p. 896)

“The quote and the tick rules are the two extremes. The LR and EMO methods take their respective places on the continuum between the quote and the tick rules. The EMO approach uses the quote rule to a lesser extent than does the LR algorithm; therefore, the EMO method exhibits a lower degree of spread overestimation than does the LR method.” (Savickas and Wilson, 2003, p. 897)

“In other words, the lower limit on the classification rate by a variety of methods seems to be 50%, rather than 0%. Eliminating a small subset of trades that are classified correctly about half the time would produce improvements, an approach that we employ in Section VI.B.” (Savickas and Wilson, 2003, p. 898)

“Using the earliest available time stamp for each trade results in an increase in the quote rule’s classification precision by about 2.5 percentage points.” (Savickas and Wilson, 2003, p. 898)

“Experimentation with time adjustments of one second to 10 minutes results in the deterioration of the quote rule’s performance. Thus, although the reporting delays are also present in the options market, they are less uniform and are found to be uncorrelated to option or trade parameters.” (Savickas and Wilson, 2003, p. 898)

“Index option trading involves many complex trades because taking covered positions in index options is not as easy (or possible) as in equity options. Frequently, the only alternatives to naked positions in index options are complex options.” (Savickas and Wilson, 2003, p. 899)

“Therefore, one way to reduce the problem of complex trades is to exclude all index trades.” (Savickas and Wilson, 2003, p. 899)

“Eliminating all complex trades cuts the sample in half. To avoid this large sample reduction, we focus only on index complex trades, summarized in Table 6. To further preserve the sample, we do not eliminate index complex trades executed on RAES (the RAES dummy is available in BODB).” (Savickas and Wilson, 2003, p. 899)

“The main sources of misclassification by all rules are reversed-quote trades (buys at the bid and sells at the ask) and wrong-side trades (buys below the bid and sells above the ask), which occur more frequently in options than in stocks and are associated with large orders to trade deep in-the-money, near-maturity options.” (Savickas and Wilson, 2003, p. 901)

“e are able to isolate a category of trades that are highly misclassified (approximately 50%–60% of the time) by all methods.” (Savickas and Wilson, 2003, p. 901)