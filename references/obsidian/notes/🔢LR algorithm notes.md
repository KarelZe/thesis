![[lr-algo.png]]


Precise description from [[@carrionTradeSigningFast2020]]:


The tick rule (TICK) relies solely on trade prices for classifying trades and does not use any quote data. To classify trades, the tick rule compares the current trade price to the price of the preceding trade. A trade is classified as a buy if the trade price is higher than the preceding trade price (uptick). Likewise, a trade is classified as a sell if the trade price is lower than the preceding trade price (downtick). If the preceding trade price is the same, then the tick rule looks back to the last different price to classify the trade. Likewise, a trade is classified as a sell if it occurs on a zero-downtick. Formally denoting the trade price of security $i$ at time $t$ as $P_{i, t}$ and $\Delta P_{i, t}$ as the price change between two successive trades and the assigned trade direction at time $t$ as Trade, we have:
If $\Delta P_{i, t}>0$, Trade $_{i, t}=$ Buy,
If $\Delta P_{i, t}<0$, Trade ${i, t}=$ Sell,
If $\Delta P_{i, t}=0$, Trade $_{i, t}=$ Trade $_{i, t-1}$.
The LR algorithm is based on a combination of the tick rule and the quote rule. Using the quote rule, a trade is classified as a buy if the price is above the midpoint of the quoted bid and ask and as a sell if the price is below the midpoint. Denoting the midpoint of the quoted spread by $m_{i, t}$, the predicted trade direction as per the quote rule is as follows:
$$
\begin{aligned}
& \text { If } P_{i, t}>m_{i, t}, \text { Trade }_{i, t}=\text { Buy, } \\
& \text { If } P_{i, t}<m_{i, t} \text { Trade }_{i, t}=\text { Sell. }
\end{aligned}
$$



- According to [[@bessembinderIssuesAssessingTrade2003]] the most widley used algorithm to categorize trades as buyer or seller-initiated.
- Accuracy has been tested in [[@odders-whiteOccurrenceConsequencesInaccurate2000]], [[@finucaneDirectTestMethods2000]] and [[@leeInferringInvestorBehavior2000]] on TORQ data set which contains the true label. (see [[@bessembinderIssuesAssessingTrade2003]])
- combination of quote and tick rule. Use tick rule to classify trades at midpoint and use the quote rule else where
- <mark style="background: #FFB86CA6;">(for a short discussion see [[@carrionTradeSigningFast2020]])</mark>


- LR algorithm
![[lr-algorithm-formulae.png]]
- in the original paper the offset between transaction prices and quotes is set to 5 sec [[@leeInferringTradeDirection1991]]. Subsequent research like [[@bessembinderIssuesAssessingTrade2003]] drop the adjustment. Researchers like [[@carrionTradeSigningFast2020]] perform robustness checks with different, subsequent delays in the robustness checks.
- See [[@carrionTradeSigningFast2020]] for comparsions in the stock market at different frequencies. The higher the frequency, the better the performance of LR. Similar paper for stock market [[@easleyFlowToxicityLiquidity2012]]
- Also five second delay isn't universal and not even stated so in the paper. See the following comment from [[@rosenthalModelingTradeDirection2012]]
> Many studies note that trades are published with non-ignorable delays. Lee and Ready (1991) first suggested a five-second delay (now commonly used) for 1988 data, two seconds for 1987 data, and “a different delay . . . for other time periods”. Ellis et al. (2000) note (Section IV.C) that quotes are updated almost immediately while trades are published with delay2. Therefore, determining the quote prevailing at trade time requires finding quotes preceding the trade by some (unknown) delay. Important sources of this delay include time to notify traders of their executions, time to update quotes, and time to publish the executions. For example, an aggressive buy order may trade against sell orders and change the inventory (and quotes) available at one or more prices. Notice is then sent to the buyer and sellers; quotes are updated; and, the trade is made public. This final publishing timestamp is what researchers see in nonproprietary transaction databases. Erlang’s (1909) study of information delays forms the theory for modeling delays. Bessembinder (2003) and Vergote (2005) are probably the best prior studies on delays between trades and quotes.
- For short discussion timing offset in CBOE data see [[@easleyOptionVolumeStock1998]] . For reasoning behind offset (e. g., why it makes senses / is necessary) see [[@bessembinderIssuesAssessingTrade2003]], who study the offset for the NASDAQ. Their conclusion is, that there is no universal optimal offset.
- for LR on CBOE data set see [[@easleyOptionVolumeStock1998]]
- LR can not handle the simultanous arrival of market buy and sell orders. Thus, one side will always be wrongly classified. Equally, crossed limit orders are not handled correctly as both sides iniate a trade independent of each other (see [[@finucaneDirectTestMethods2000]]). 
- LR is also available in bulked 
**Reverse LR algorithm:**
- first introduced in [[@grauerOptionTradeClassification2022]] (p 12)
- combines the quote and reverse tick rule
- performs fairly well for options as shown in [[@grauerOptionTradeClassification2022]]
- Lee and Ready SAS implementatin https://github.com/jblocher/sas_util/blob/master/LR_Trade_ID.sas

- Poor performance on option data sets explain why

“The established methods, most notably the algorithms of Lee and Ready (1991) (LR), Ellis et al. (2000) (EMO), and Chakrabarty et al. (2007) (CLNV), classify trades based on the proximity of the transaction price to the quotes in effect at the time of the trade. This is problematic due to the increased frequency of order submission and cancellation. With several quote changes taking place at the time of the trade, it is not clear which quotes to select for the decision rule of the algorithm.” (Jurkatis, 2022, p. 6)

**Limitations:** “If the second trade of the day takes place at the same price as the first one, both trades cannot be classified by the tick rule. The LR and EMO rules also lose some trades since they are combinations of the tick and the quote rules.” (Savickas and Wilson, 2003, p. 886) -> Wouldn't this be filled by the prev. trade that is different, even if it is from long ago?


“The paper further shows that, while the Lee and Ready (1991) algorithm has been the default choice among the traditional trade classification algorithm—possibly partly due to being automatically supplied by data vendors, partly due to its simplicity—the similar simplistic algorithms of Chakrabarty et al. (2007) and Ellis et al. (2000) tend to perform better and may be preferred in certain applications.” (Jurkatis, 2022, p. 23)