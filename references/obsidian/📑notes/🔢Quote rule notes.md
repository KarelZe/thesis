Tags: #trade-classification #quote-rule 


**Notation:** For notation see [[@carrionTradeSigningFast2020]] or  [[@jurkatisInferringTradeDirections2022]] or [[@olbrysEvaluatingTradeSide2018]]..  Denoting the midpoint of the quoted spread by $m_{i, t}$, the predicted trade direction as per the quote rule is as follows:
$$
\begin{aligned}
& \text { If } P_{i, t}>m_{i, t} \text { Trade }_{i, t}=\text { Buy, } \\
& \text { If } P_{i, t}<m_{i, t}, \text { Trade }_{i, t}=\text { Sell. } \\
&
\end{aligned}
$$

$$
\begin{array}{lll}
\hline \text { Rule } & \text { Conditions } & \\
\hline & P_t>P_t^{\text {mid }} & \text { Trade is classified as buyer-initiated } \\
\text { QR } & P_t<P_t^{\text {mid }} & \text { Trade is classified as seller-initiated } \\
& P_t=P_t^{\text {mid }} & \text { Trade is not classified }
\end{array}
$$

- The quote rule classifies a trade as buyer initiated if the trade price is above the midpoint of the buy and ask as buys and if it is below as seller-initiated. Can not classify at the midpoint of the quoted spread. (see e.g., [[@leeInferringTradeDirection1991]] or [[@finucaneDirectTestMethods2000]])
- **Attribution:** “The quote rule compares trade prices to the prevailing quote at the time of the trade. Trades at or above the ask are classified as buys; trades at or below the bid are classified as sells. Trades inside the spread are classified based on their proximity to the bid and ask ([[@harrisDayEndTransactionPrice1989]], 1989). Trades at the midpoint of the spread cannot be classified by using the quote rule.” ([Pöppe et al., 2016, p. 166](zotero://select/library/items/5A83SDDB)) ([pdf](zotero://open-pdf/library/items/4XIK47X6?page=2&annotation=U4U9PWDS))
- See [[@hasbrouckTradesQuotesInventories1988]]. Might be one of the first to mention the quote rule. It is however not very clearly defined. Paper also discusses some (hand-wavy) approaches to treat midpoint transactions.
- “In other words, given a midpoint trade on a down (up) tick the next price change would likely be an up (down) tick” (Lee and Ready, 1991, p. 9)
- **Results:** 💸“The degree of success varies inversely with a rule’s reliance on past transaction prices: 82.78% for the quote rule, 80.11% for LR, 76.54% for EMO, and 59.4% for the tick method.” ([[@savickasInferringDirectionOption2003]], 2003, p. 886)
- **Results:** 💸 “We find that the quote rule performs best when applied successively to National Best Bid and Offer (NBBO) and ISE quotes as proposed by Muravyev and Ni (2020). Generally, quote rules outperform tick rules by far.” ([[@grauerOptionTradeClassification2022]], p. 3)
- **Results:** 💸 “An additional finding from this data set is that for trades with price improvements occurring inside the NBBO spread, the quote rule works better when applied first to quotes at the trading venue and then to the NBBO.” ([[@grauerOptionTradeClassification2022]], p. 3)
- **Results:** 💸 “Similarly, Savickas and Wilson (2003) show that all common methods perform poorly at estimating effective spreads for options. While the quote rule overestimates the spread due to its inability to account for reversed-quote trades, the tick rule severely underestimates effective spreads due to its low classification success.” ([[@savickasInferringDirectionOption2003]] p. 6)
- by definition quote rule is unable to classify trades at the bids-ask spread midpoint.
