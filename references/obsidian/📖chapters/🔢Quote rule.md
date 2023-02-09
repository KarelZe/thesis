
<mark style="background: #ADCCFFA6;">“Methods of inferring trade direction can be classified as: tick tests, which use changes in trade prices; the quote method, which compares trade prices to quotes;” (Finucane, 2000, p. 557)</mark>

The quote rule compares the trade price against the corresponding quotes at the time of the trade. <mark style="background: #FFB86CA6;">(Intuition?)</mark> If the trade price $P_{i,t}$ is above the midpoint of the bid-ask spread, denoted by $m_{i,t}$, the trade is classified as a buy and if it is below the midpoint, as a sell ([[@harrisDayEndTransactionPrice1989]] p.41). Thus, the classification rule, is formally given by:
$$
%\tag{10}
  \begin{equation}
    \text{Trade}_{i,t}=
    \begin{cases}
      1, & \text{if}\ P_{i, t}>m_{i, t} \\
      0, & \text{if}\ P_{i, t}<m_{i, t}  \\
      %\\\texttt{[NAN]}, & \text{otherwise} %
    \end{cases}
  \end{equation}
$$
By definition, the quote rule cannot classify trades at the midpoint of the quoted spread. ([[@hasbrouckTradesQuotesInventories1988]] p.241) discusses multiple alternatives for signing mid-spread transactions based on the subsequent quotes, contemporaneous, or the subsequent transaction. However, the most common approach overcomes this limitation is, to combine the quote rule with other approaches, such as the tick rule, into an ensemble.

The quote rule requires to match *one* bid and ask quote with each trade based on a timestamp. Due to the finite resolution of the dataset's timestamps and active markets, multiple quote changes can co-occur at the time of trade, some of which, actually after the trade. As such, it remains unclear which quote to consider in trade classification, and a *quote timing technique* must be employed. Empirically,  ([[@holdenLiquidityMeasurementProblems2014]] p.1,765) observe, that the most common choice is to use the last quote by order from the time increment (e. g., the second) before the trade.

In contrast to the tick rule, the quote rule requires both the trade price and quote data, and is thus less data efficient. The reduced dependence on past transaction prices and the focus on quotes has nonetheless positively impacted classification accuracies in option markets, as the studies of ([[@savickasInferringDirectionOption2003]] p.886) and ([[@grauerOptionTradeClassification2022]] p.3) reveal. Especially, if trade classification is performed on the NBBO.

<mark style="background: #D2B3FFA6;">TODO: </mark>🧨 See [[@hasbrouckTradesQuotesInventories1988]] for analysis of the quote rule for limit orders, crossed orders, etc.

**Notes:**
[[🔢Quote rule notes]]