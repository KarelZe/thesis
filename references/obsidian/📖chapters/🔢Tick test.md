Based on the rationale that buys increase the trade price and sells lower them, the *tick test* classifies trades by the change in trade price ([[@easleyDiscerningInformationTrade2016]] 271). Its first use is documented in (cp. [[@holthausenEffectLargeBlock1987]] 244) and ([[@hasbrouckTradesQuotesInventories1988]] 240). 

We denote the trade price of the $i$-th security at time $t$ as $p_{i,t}$ and the price change between two successive trades. The tick test is formally defined as 

$$
  \begin{equation}

    \text{Trade}_{i,t}=

    \begin{cases}
      1, & \text{if}\ p_{i, t} > p_{i, t-1}\\
      0, & \text{if}\ p_{i, t} < p_{i, t-1}\\
	  p_{i,t-1} = p_{i,t-2},& \text{else}.
    \end{cases}
  \end{equation}
$$
If the trade price is higher than the previous price (uptick) the trade is classified as a buy. Reversely, if it is below the previous price (downtick), the trade is classified as a sell. If the price change is zero (zero tick), the signing uses the last price different from the current price. ([[@leeInferringTradeDirection1991]] 3)

<mark style="background: #BBFABBA6;">“The primary limitation of the tick test is its relative imprecision when compared to a quote-based approach, particularly if the prevailing quote has changed or it has been a long time since the last trade.” (Lee and Ready, 1991, p. 3)</mark> -> employs no constraints

By this means, the tick rule can sign all trades as long as there is a last distinguishable trade price. Being only dependent on transaction data makes the tick rule highly data efficient ([[@odders-whiteOccurrenceConsequencesInaccurate2000]] 264) or ([[@theissenTestAccuracyLee2000]] 1). Waiving any quote data for classification contributes to this efficiency, but also poses a major limitation with regard to trades at the bid or ask, as discussed in ([[@finucaneDirectTestMethods2000]] 557--558). For instance, if quotes rise between trades, then a sale at the bid on an uptick or zero uptick, is misclassified as buys by tick test due to the overall increased trade price. Similarly for falling quotes, buys at the ask on downticks or zero downticks will be erroneously classified as a sell. 

A variant of the tick test is the *reverse tick test* as applied in ([[@hasbrouckTradesQuotesInventories1988]] 241). It is similar to the tick rule but classifies based on the trade price of the next, distinguishable trade price. As donated in Equation ... the trade is classified as seller-initiated, if the next trade is on an uptick or a zero uptick, and classified as buyer-initiated for trades at a downtick or a zero downtick. ([[@leeInferringTradeDirection1991]] 735--636).

$$
  \begin{equation}
    \text{Trade}_{i,t}=
    \begin{cases}
      1, & \text{if}\ p_{i, t} > p_{i, t+1}\\
      0, & \text{if}\ p_{i, t} < p_{i, t+1}\\
	  P_{i,t+1} = P_{i,t+2},& \text{else}
    \end{cases}
  \end{equation}
$$
Both the tick test and reverse tick test result in the same classification, if the current trade is bracketed by a price reversal and the price change after the trade is opposite from the change before the trade, but differ for price continuations when price changes are in the same direction ([[@leeInferringTradeDirection1991]] 736). In practice, ([[@grauerOptionTradeClassification2022]] 29--32) observe higher accuracies for the reverse tick test on a sample of option trades recorded at the ISE and CBOE. This result contradicts results in the stock market ([[@leeInferringTradeDirection1991]] 737). 

**Notes:**
[[🔢Tick test notes]]