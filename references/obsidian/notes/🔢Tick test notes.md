Tags: #trade-classification  #tick-rule 

One of the first works who mention the tick test is [[@holthausenEffectLargeBlock1987]] (referred to as tick classification rule) or [[@hasbrouckTradesQuotesInventories1988]] (referred to as transaction rule)

**Algorithm:** Formal description in [[@olbrysEvaluatingTradeSide2018]] and [[@carrionTradeSigningFast2020]] (see below) and [[@jurkatisInferringTradeDirections2022]]:
Formally denoting the trade price of security $i$ at time $t$ as $P_{i, t}$ and $\Delta P_{i, t}$ as the price change between two successive trades and the assigned trade direction at time $t$ as Trade, we have:
If $\Delta P_{i, t}>0$, Trade $_{i, t}=$ Buy,
If $\Delta P_{i, t}<0$, Trade $_{i, t}=$ Sell,
If $\Delta P_{i, t}=0$, Trade $_{i, t}=$ Trade $_{i, t-1}$.

**Informal description:** Tick tests use changes in trade prices and look at previous trade prices to infer trade direction. If the trade occurs at a higher price, hence uptick, as the previous trade its classified as as buyer-initiated. If the trade occurs at a lower price its seller-initiated. If the price change is zero, the last price is taken, that is different from the current price. (see e. g., [[@grauerOptionTradeClassification2022]] or [[@finucaneDirectTestMethods2000]] or [[@leeInferringTradeDirection1991]] for similar framing)

**Variant:** 
- A variant of the tick test is the *reverse tick test* as popularized by [[@leeInferringTradeDirection1991]]. Instead of using the previous distinguishable trade price, the subsequent trade price, that is different from the current trade is used. 
- Instead of the previous trade, the reverse tick rule uses the subsequent trade price to classify the current trade. 
- If the next trade price that is different from the current price, is below the current price the trade (on a down tick or zero down tick) is classified as buyer-initiated. If the next distinguishable price is above the current price (up tick or zero up tick), the current price the trade is seller-initiated. (loosely adapted from [[@grauerOptionTradeClassification2022]]) (see also [[@leeInferringTradeDirection1991]])
**Lower bound:** [[@perlinPerformanceTickTest2014]] proved that the tick test performs better than random chance. 
**Data efficiency:** low data requirements, as only transaction data is needed. (see [[@theissenTestAccuracyLee2000]]). Could be good enough though. 
**Data efficiency:** “The advantages of the tick method are that it requires only transaction data (quotes are not necessary) and that no trades are left unclassified. The disadvantage is that the tick method incorporates less information than the quote method since it does not use the posted quotes.” ([[@odders-whiteOccurrenceConsequencesInaccurate2000]], 2000, p. 264)

**Limitations:** 👩‍🚒“First, options are much more illiquid than stocks with many series not recording a trade for days or weeks. For that reason, tick rules that depend on the information from preceding or succeeding trades might be problematic.” ([[@grauerOptionTradeClassification2022]], 2022, p. 1)
**Limitations:** 👩‍🚒Tick test can not handle if two trades do not involve market orders e. g. two limit orders. In such a case the tick rule could be applied, but there is ambiguous. (see [[@finucaneDirectTestMethods2000]])
**Limitations:** 👩‍🚒 “For example, the quote rule is unable to classify trades occurring at the bid-ask spread midpoint, but the tick rule cannot classify the opening trade of the day (and any subsequent trades at an identical price)” ([[@savickasInferringDirectionOption2003]], 2003, p. 882)
**Limitations:** 👩‍🚒“Each of the four classification methods cannot classify some trades. The quote rule is unable to classify midspread trades. The tick rule cannot classify the first trade of the day for a given security. If the second trade of the day takes place at the same price as the first one, both trades cannot be classified by the tick rule. The LR and EMO rules also lose some trades since they are combinations of the tick and the quote rules.” ([[@savickasInferringDirectionOption2003]], 2003, p. 886) 

**Bias in selection:** “As the probability of observing only buy trades or only sell trades decreases with an increasing number of trades, the number of trades per option day is lower and the time between two trades is higher in our matched samples compared to their full sample equivalents. Because tick tests depend on the information from preceding or succeeding trades as a precise signal for the fair option price, our results might therefore underestimate their performance.” ([[@grauerOptionTradeClassification2022]]., 2022, p. 9)

**Results:** 💸 “All four rules perform worst when applied to index options, and best with equity options. The poor performance of the tick rule is a consequence of the fact that only 59.7% (58.7%) of all option buys (sells) occur on an uptick (downtick).” (Savickas and Wilson, 2003, p. 886)
**Results:** 💸 “This implies that the sample used for assessing the performance of the tick rule will consist of a higher proportion of index option trades (because some equity options trade infrequently and cannot be classified by the tick rule) and will be significantly smaller. Therefore, the performance measure will be biased downward for the tick rule.” (Savickas and Wilson, 2003, p. 887)
**Results:** 💸 “When the tick rule is not “reset” at the beginning of each day, the number of trades classified by the rule increases (as expected), but the classification precision does not improve: only 55.73% of all 1,404,365 classifiable trades are labeled correctly by the tick rule.” (Savickas and Wilson, 2003, p. 886)
**Results:** 💸 “Specifically, the only difference between the tick and the EMO rules is that the latter uses the quote rule for at-the-quote trades. Consequently, the EMO method outperforms the tick rule only for those trades. Similarly, the LR and EMO methods treat at-the-quote and at-midspread trades exactly the same, but the LR approach applies the quote rule to all other trades.” (Savickas and Wilson, 2003, p. 891)
**Results: 💸 “Generally, quote rules outperform tick rules by far. For the tick rule, a higher success rate can be achieved using prices across all exchanges and information from subsequent trades” (Grauer et al., 2022, p. 3) -> lead them to propose the [[🔢Depth rule]].
**Results:** 💸 “For example, Easley, Lopez de Prado, and O’Hara (2012) show that the tick test works reasonably well for e-mini S&P 500 Futures, correctly classifying 86% of the trades, but it is less accurate in gold futures (79%) and in oil futures (67%).” (Grauer et al., 2022, p. 5)
**Effective spread:** “The tick rule severely underestimates effective spread. This is a consequence of the method’s classifying correctly just slightly more than half of all trades.” ([[@savickasInferringDirectionOption2003]], 2003, p. 896)

