- Tick tests use changes in trade prices and look at previous trade prices to infer trade direction. If the trade occurs at a higher price, hence uptick, as the previous trade its classified as as buyer-initiated. If the trade occurs at a lower price its seller-iniated. If the price change is zero, the last price is taken, that is different from the current price. (see e. g., [[@grauerOptionTradeClassification2022]] or [[@finucaneDirectTestMethods2000]] or [[@leeInferringTradeDirection1991]] for similar framing)

**Results:** “For example, Easley, Lopez de Prado, and O’Hara (2012) show that the tick test works reasonably well for e-mini S&P 500 Futures, correctly classifying 86% of the trades, but it is less accurate in gold futures (79%) and in oil futures (67%).” (Grauer et al., 2022, p. 5)

**Notation:**


- Consider  for citation [[@leeInferringTradeDirection1991]] .
- One of the first works who mention the tick test is [[@holthausenEffectLargeBlock1987]] (referred to as tick classification rule) or [[@hasbrouckTradesQuotesInventories1988]] (referred to as transaction rule)
- ![[formula-tick-rule.png]]
	Adapted from [[@olbrysEvaluatingTradeSide2018]]
	![[tick-rule-formulae-alternative.png]]
Copied from [[@carrionTradeSigningFast2020]]
- Sources of error in the tick test, when quotes change.
 [[@finucaneDirectTestMethods2000]]
- low data requirements, as only transaction data is needed. (see [[@theissenTestAccuracyLee2000]]) Could be good enough though.
- Tick test can not handle if two trades do not involve market orders e. g. two limit orders. In such a case the tick rule could be applied, but there is ambiguous. (see [[@finucaneDirectTestMethods2000]])
- [[@perlinPerformanceTickTest2014]] proved that the tick test performs better than random chance. 


The tick rule is very data efficient, as only transaction data ... However, in option markets

**Reverse Tick Test**
A variant of the tick test is the reverse tick test as popularized by [[@leeInferringTradeDirection1991]]. Instead of using the previous distinguishable trade price, the subsequent trade price, that is different from the current trade is used. 

- Instead of the previous trade, the reverse tick rule uses the subsequent trade price to classify the current trade. 
- If the next trade price that is differnet from the current price, is below the current price the trade (on a down tick or zero down tick) is classified as buyer-initiated. If the next distinguishable price is above the current price (up tick or zero up tick), the current price the trade is seller-initiated. (loosely adapted from [[@grauerOptionTradeClassification2022]]) (see also [[@leeInferringTradeDirection1991]])
