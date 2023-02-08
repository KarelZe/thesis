“The trade indicator is a binary variable stating whether the buyer or seller of an asset has initiated the trade by submitting a market order or an immediately executed limit order.” (Frömmel et al., 2021, p. 4)

- “Although extant algorithms are adequate to the basic job of sorting trades, our work suggests that a refinement to the extant methods of classifying trades will do even better.” (Ellis et al., 2000, p. 539) -> nice word "extant methods"
- Limit to view, yet theoretically promising techniques as derived in [[#^d8f019]]
- "The validity of many economic studies hinges on the ability to acurately classify trades as buyer or seller-initiated." (found in [[@odders-whiteOccurrenceConsequencesInaccurate2000]])
- trade site classification matters for several reasons, market liquidity measures, short sells, study of bid-ask-spreads.
- Where is trade side classification applied? Why is it important? Do citation search.
- Repeat in short the motivation
- Outperformance in similar / other domains
- Obtain probabilities for further analysis

- Crisp sentence of what ML is and why it is promising here. 

- goal is to outperform existing classical approaches

- [[@rosenthalModelingTradeDirection2012]] lists fields where trade classification is used and what the impact of wrongly classified trades is.
- The extent to which inaccurate trade classification biases empirical research depends on whether misclassifications occur randomly or systematically [[@theissenTestAccuracyLee2000]].
- There is no common sense of who is the initiatiator of a trade. See discussion in [[@odders-whiteOccurrenceConsequencesInaccurate2000]]
- over time proposed methods applied more filters / got more sophisticated but didn't substantialy improve im some cases. See e. g., [[@finucaneDirectTestMethods2000]] Time to switch to another paradigm and let the data speak?
- Works that require trade side classification in option markets:
	- [[@muravyevOrderFlowExpected2016]]
	- [[@huDoesOptionTrading2014]]
- and its consequences are an important, but understudied, cause for concern.
- Commonly stock trade classification algorithms are used



- “Who is buying and who is selling are important elements in determining the information content of trades, the order imbalance and inventory accumulation of liquidity providers, the price impact of large trades, the effective spread, and many other related questions. Unfortunately, commonly available high frequency databases do not provide in? formation on trade direction. Consequently, empirical researchers have relied on trade direction algorithms to classify trades as buyer or seller motivated.” ([[@ellisAccuracyTradeClassification2000]], p. 529)
- “The Berkeley Options Data Base does not classify trades as buyer-initiated or seller-initiated. This classification must be done using quote and trade information.” ([Easley et al., 1998, p. 453](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=23&annotation=GXHQMKIW))

- “Such indication made directly from the data is nowadays in mostly cases inaccessible, since the majority of public databases including transaction data do not contain information of trade initiators and trade direction.” ([[@nowakAccuracyTradeClassification2020]], p. 65)

“The information which party to a trade is a trade initiator is indispensable to specify the trade indicator models used to investigate the intraday price formation (Glosten and Harris 1988; Huang and Stoll 1997; Madhavan 1992; McGroarty et al. 2007; Hagströmer et al. 2016). Moreover, the identification of party to a trade which is responsible for initiating a particular transaction is advantageous to clarify many important issues related to the market microstructure. First of all, it may be used to ascertain the information content of trades. Second, it can help to figure out the magnitude of the order imbalance as well as the proportion of the inventory accumulation made by the liquidity suppliers. Third, it helps to assess the price impact of large in volume transactions as well as the magnitude of effective spread (Ellis et al. 2000).” ([[@nowakAccuracyTradeClassification2020]], 2020, p. 66)

“Various papers illustrate the consequences of inaccurate trade classification in empirical finance. For example, Boehmer et al. (2007) show analytically and empirically that inaccurate classification of trades leads to downward-biased PIN (probability of informed trade) estimates and that the magnitude of the bias is related to a security’s trading intensity. Using two separate periods around the NYSE’s change to a tick size of $1/16 in June 1997, Peterson and Sirri (2003) report that actual execution costs are overstated by up to 17% using effective spread estimates that incorporate errors in trade direction and benchmark quote assignments, and that the highest biases occur for small trades and for trades of larger firms.” ([[@aktasTradeClassificationAccuracy2014]], 2014, p. 260)


“Much of market microstructure analysis is built on the concept that traders learn from market data. Some of this learning is prosaic, such as inferring buys and sells from trade execution. Other learning is more complex, such as inferring underlying new information from trade executions. In this paper, we investigate the general issue of how to discern underlying information from trading data. We examine the accuracy and efficacy of three methods for classifying trades: the tick rule, the aggregated tick rule, and the bulk volume classification methodology. Our results indicate that the tick rule is a reasonably good classifier of the aggressor side of trading, both for individual trades and in aggregate. Bulk volume is shown to also be reasonably accurate for classifying buy and sell trades, but, unlike the tick-based approaches, it can also provide insight into other proxies for underlying information.” ([Easley et al., 2016, p. 284](zotero://select/library/items/X6ZNZ556)) ([pdf](zotero://open-pdf/library/items/HPC6KBMF?page=16&annotation=VC98DC2N))