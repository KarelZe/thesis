
title: On Inferring the Direction of Option Trades
authors: Robert Savickas, Arthur J Wilson
year: 2003
tags : #trade-classification #lr #emo #quote-rule #tick-rule #application 
status : #📦 
related:
- [[@hasbrouckTradesQuotesInventories1988]]
- [[@holthausenEffectLargeBlock1987]]
- [[@ellisAccuracyTradeClassification2000]]
- [[@chakrabartyTradeClassificationAlgorithms2007]]
- [[@grauerOptionTradeClassification2022]]

## Key takeaways

- “To sign option trades as buys and sells, researchers often employ stock trade classification rules including the quote, the LR, the EMO, and the tick methods” (Savickas and Wilson, 2003, p. 881)
- Author correctly sign 83% (quote), 80% (LR), 77% (EMO), and 59% (tick method) of all classifiable trades, respectively. This is much higher than the later work of [[@grauerOptionTradeClassification2022]].
- Precision is up to **24 % lower** for **options** than for stock data. (Savickas and Wilson, 2003, p. 882)
- “We find that the components of index option complex trades not executed on the Retail Automated Execution System are misclassified almost 50% of the time by any method. The elimination of these trades (15% of the sample) results in a success rate of over 87% for the quote rule.” (Savickas and Wilson, 2003, p. 881)
- They apply trade classification to effective spread calculation. Could easily adapt this for my problem.

## Data Set
- “We use two datasets covering the period July 3, 1995 to December 31, 1995; both are provided by CBOE.”  (Savickas and Wilson, 2003, p. 882) This is very **tiny**. 
- “Each record in the matched trade dataset reports the following information: date and execution time of a trade, trade size, price, the option’s parameters (put/call, expiration month and year, and strike price), the underlying security, the reporting trader’s type code, and a dummy variable that indicates whether the reporting trader bought or sold the option. Thus, there is one record for each trading party for all trades on CBOE.” (Savickas and Wilson, 2003, p. 884)
- “We group all records in the matched trade dataset into **pairs**, one pair per trade. The two records in each pair have the same trade date, underlying security, option’s parameters, trade size, and trade price. The buy/sell indicators are different for the two records in each pair (a buy and a sell).” (Savickas and Wilson, 2003, p. 884)

- “Consistent with previous literature, we assume that all MC and BC trades are initiated by the customer, all MB trades are initiated by the broker, all MF and BF trades are initiated by the firm, and all MN and BN trades are initiated by the non-member.4” (Savickas and Wilson, 2003, p. 885) Isn't this a bit **hand-wavy**? ❌
- “All trades and quotes for which the underlying asset price is reported as zero and all cancelled trades are excluded from the sample.” (Savickas and Wilson, 2003, p. 885)

## Classical rules
- **Quote Rule:** “According to the quote rule, a trade is buyer- (seller-) initiated if it occurs above (below) the midpoint of the bid-ask spread. Trades that occur exactly at the midspread cannot be classified. 
- **The tick rule**, on the other hand, classifies any trade as a buy (sell) if its price is above (below) the most recent transaction price that is different from the price of the transaction being classified. 
- The **LR** and **EMO methods** combine the quote and tick rules to various degrees. The LR approach uses the tick rule to classify only at-midspread trades and the quote rule for all the other trades. The EMO method uses the quote rule for at-quote trades and the tick rule for all the other trades.” (Savickas and Wilson, 2003, p. 885)

- **Limitations of rules:** “Each of the four classification methods cannot classify some trades. The quote rule is unable to classify midspread trades. The tick rule cannot classify the first trade of the day for a given security. If the second trade of the day takes place at the same price as the first one, both trades cannot be classified by the tick rule. The LR and EMO rules also lose some trades since they are combinations of the tick and the quote rules.” (Savickas and Wilson, 2003, p. 886)

- “The degree of success varies inversely with a rule’s reliance on past transaction prices: 82.78% for the quote rule, 80.11% for LR, 76.54% for EMO, and 59.4% for the tick method.” (Savickas and Wilson, 2003, p. 886)

## Results of classical rules
### Univariate Setting
- They compare to individual rules on a sample-to-sample basis. Requires some preprocessing, as not all rules can be applied all the the time. “To address this issue, we form a common sample in which every trade can be classified by each of the four rules.” (Savickas and Wilson, 2003, p. 887)
- “For each of the four classification rules, we use eight variables that may potentially explain observed performance. These variables are: the trade size, time from the previous trade, time from the previous quote, the moneyness ratio, trading days to maturity, the absolute value of the relative change in the underlying price, the location of trade relative to the bid-ask quotes, and whether a trade is executed on RAES.” (Savickas and Wilson, 2003, p. 887)
- **Observation 1:** “trades inside the quotes are more likely to be misclassified.” (Savickas and Wilson, 2003, p. 888)
- **Observation 2:** “Trade size affects classification precision in two related ways. First, as found by EMO (2000), small trades are more likely to be executed at the quotes, implying an inverse relation between classification precision and trade size.” (Savickas and Wilson, 2003, p. 888)
- **Observation 3:** “The longer the time from the previous trade, the less relevant is the information in the previous trade price, adversely affecting tick-based classifications.” (Savickas and Wilson, 2003, p. 888)
- **Observation 4:** “First, as in the case of the past price, the older the quotes, the less relevant the information they convey, making quote-based classification less precise. Second, the older the quote, the greater underlying asset price change since that quote, which can enhance quote-based classification as described in the next paragraph.” (Savickas and Wilson, 2003, p. 888)
- **Observation 5:** “Option moneyness has two indirect effects on classification precision. First, deep in-the-money options have deltas close to one (in absolute value) and, therefore, are more sensitive to the underlying asset price changes. Second, there is a positive relation between option moneyness and trade size” (Savickas and Wilson, 2003, p. 889)
- **Observation 6:** “Time to maturity also has an indirect effect on classification precision. Trades in options with longer maturity tend to be smaller, resulting in negative correlation between the effects of maturity and of trade size.” (Savickas and Wilson, 2003, p. 889)
- **Observation 7:** “Specifically, one of the most noticeable regularities is that smaller trades are classified more precisely. This is because these trades are more likely to be executed at quotes and are less prone to reversed-quote trading (partially due to the fact that many small trades are executed on RAES)” (Savickas and Wilson, 2003, p. 889) **Similar:** [[@grauerOptionTradeClassification2022]]
- **Observation 8:** “There seems to be a direct relation between using the quote information and successful option trade classification. This relation implies that the information conveyed by the past transaction prices is less relevant than information contained in quotes.” (Savickas and Wilson, 2003, p. 891)
- “Specifically, the only difference between the tick and the EMO rules is that the latter uses the quote rule for at-the-quote trades. Consequently, the EMO method outperforms the tick rule only for those trades. Similarly, the LR and EMO methods treat at-the-quote and at-midspread trades exactly the same, but the LR approach applies the quote rule to all other trades.” (Savickas and Wilson, 2003, p. 891)
### Multivariate Analysis
- For multivariate analysis authors use **logistic regression** to study the most import of 8 covariats in a multivariate setting.
- “The most economically significant variables in all four regressions are the trade location relative to quotes (including the RAES dummy), the absolute value of the relative underlying price change, and the put/call dummy.” (Savickas and Wilson, 2003, p. 893)
- "Larger trades have a higher probability of being traded on reversed quotes, as do shorter-term options. When the underlying price change is controlled for, we observe a positive relation between option moneyness and the probability of an RQ trade.” (Savickas and Wilson, 2003, p. 895)
- **Observation: 1** “The quote and the tick rules are the two extremes. The LR and EMO methods take their respective places on the continuum between the quote and the tick rules. The EMO approach uses the quote rule to a lesser extent than does the LR algorithm; therefore, the EMO method exhibits a lower degree of spread overestimation than does the LR method.” (Savickas and Wilson, 2003, p. 897)
- **Observation 2:** “In other words, the lower limit on the classification rate by a variety of methods seems to be 50%, rather than 0%. Eliminating a small subset of trades that are classified correctly about half the time would produce improvements, an approach that we employ in Section VI.B.” (Savickas and Wilson, 2003, p. 898)
- **Observation 3:** “Using the earliest available time stamp for each trade results in an increase in the quote rule’s classification precision by about 2.5 percentage points.” (Savickas and Wilson, 2003, p. 898)
- **Observation 4:** “Index option trading involves many complex trades because taking covered positions in index options is not as easy (or possible) as in equity options. Frequently, the only alternatives to naked positions in index options are complex options. Therefore, one way to reduce the problem of complex trades is to exclude all index trades. Eliminating all complex trades cuts the sample in half.

## Conclusion 
- “The main sources of misclassification by all rules are reversed-quote trades (buys at the bid and sells at the ask) and wrong-side trades (buys below the bid and sells above the ask), which occur more frequently in options than in stocks and are associated with large orders to trade deep in-the-money, near-maturity options.” (Savickas and Wilson, 2003, p. 901)
- “Authors identify the components of index spreads and combinations not executed on RAES.” (Savickas and Wilson, 2003, p. 901)
