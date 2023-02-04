![[lr-algo.png]]

- According to [[@bessembinderIssuesAssessingTrade2003]] the most widley used algorithm to categorize trades as buyer or seller-initiated.
- Accuracy has been tested in [[@odders-whiteOccurrenceConsequencesInaccurate2000]], [[@finucaneDirectTestMethods2000]] and [[@leeInferringInvestorBehavior2000]] on TORQ data set which contains the true label. (see [[@bessembinderIssuesAssessingTrade2003]])
- combination of quote and tick rule. Use tick rule to classify trades at midpoint and use the quote rule else where

- LR algorithm
![[lr-algorithm-formulae.png]]
- in the original paper the offset between transaction prices and quotes is set to 5 sec [[@leeInferringTradeDirection1991]]. Subsequent research like [[@bessembinderIssuesAssessingTrade2003]] drop the adjustment. Researchers like [[@carrionTradeSigningFast2020]] perform robustness checks with different, subsequent delays in the robustness checks.
- See [[@carrionTradeSigningFast2020]] for comparsions in the stock market at different frequencies. The higher the frequency, the better the performance of LR. Similar paper for stock market [[@easleyFlowToxicityLiquidity2012]]
- Also five second delay isn't universal and not even stated so in the paper. See the following comment from [[@rosenthalModelingTradeDirection2012]]
>Many studies note that trades are published with non-ignorable delays. Lee and Ready (1991) first suggested a five-second delay (now commonly used) for 1988 data, two seconds for 1987 data, and “a different delay . . . for other time periods”. Ellis et al. (2000) note (Section IV.C) that quotes are updated almost immediately while trades are published with delay2. Therefore, determining the quote prevailing at trade time requires finding quotes preceding the trade by some (unknown) delay. Important sources of this delay include time to notify traders of their executions, time to update quotes, and time to publish the executions. For example, an aggressive buy order may trade against sell orders and change the inventory (and quotes) available at one or more prices. Notice is then sent to the buyer and sellers; quotes are updated; and, the trade is made public. This final publishing timestamp is what researchers see in nonproprietary transaction databases. Erlang’s (1909) study of information delays forms the theory for modeling delays. Bessembinder (2003) and Vergote (2005) are probably the best prior studies on delays between trades and quotes.
- For short discussion timing offset in CBOE data see [[@easleyOptionVolumeStock1998]] . For reasoning behind offset (e. g., why it makes senses / is necessary) see [[@bessembinderIssuesAssessingTrade2003]], who study the offset for the NASDAQ. Their conclusion is, that there is no universal optimal offset.
- for LR on CBOE data set see [[@easleyOptionVolumeStock1998]]
- LR can not handle the simultanous arrival of market buy and sell orders. Thus, one side will always be wrongly classified. Equally, crossed limit orders are not handled correctly as both sides iniate a trade independent of each other (see [[@finucaneDirectTestMethods2000]]). 
- LR is also available in bulked 
**Reverse LR algorithm:**
- first introduced in [[@grauerOptionTradeClassification2022]] (p 12)
- combines the quote and reverse tick rule
- performs fairly well for options as shown in [[@grauerOptionTradeClassification2022]]
- Lee and Ready SAS implementatin https://github.com/jblocher/sas_util/blob/master/LR_Trade_ID.sas