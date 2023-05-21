Tags: #trade-classification 

- “6 Level-1 algorithms use only trade price data; level-2 algorithms use both trade and quote data.” ([[@chakrabartyTradeClassificationAlgorithms2012]], p. 6)

- Differences in market data: https://www.thebalancemoney.com/level-i-or-level-ii-market-data-1031144


In absence of the . 

We start by the popular quote rule and tick test in Section ... and continue with two recent alternatives. 

Because every trade has both a buyer and a seller, it is necessary to classify the “active” side of each option transaction.

- requires separatiing buys from sells in the raw trading data

While the information about the initator of a trade is missing in public data sets ... we infer the 

**Possible openings:** 
 Publicly available data generally do not distinguish buys and sells, so an algorithm is necessary to infer buys and sells. The algorithm we use seems to work well, but the algorithm itself is independent of the VPIN metric. Any algorithm could be used to provide input to the estimation of VPIN. 146” ([[@easleyFlowToxicityLiquidity2012]], p. 1465)

“The improved ability to discern whether a trade was a buy order or a sell order is of particular importance” ([[@leeInferringTradeDirection1991]], p. 1)
“Therefore, trade classification rules (TCR) have been developed in order to classify trades as buyer- or seller-initiated, when the true originator is unknown” (Frömmel et al., 2021, p. 4)

“The trade indicator is a binary variable stating whether the buyer or seller of an asset has initiated the trade by submitting a market order or an immediately executed limit order.” ([[@frommelAccuracyTradeClassification2021]], p. 4)

“The goal of the trade side classification is to determine the initiator of the transaction and to classify trades as being either buyer or seller motivated. However, a formal definition of a trade initiator is rarely stated in the literature.” ([[@olbrysEvaluatingTradeSide2018]], p. 4)

“Trade classification rules (hereafter referred to as TCR) are intended to indicate the party to a trade who initiates a transaction. It may by either a buyer or a seller. Such indication made directly from the data is nowadays in mostly cases inaccessible, since the majority of public databases including transaction data do not contain information of trade initiators and trade direction.” ([[@nowakAccuracyTradeClassification2020]], p. 65)


“Methods of inferring trade direction can be classified as: tick tests, which use changes in trade prices; the quote method, which compares trade prices to quotes;” (Finucane, 2000, p. 557)



<mark style="background: #FFB86CA6;">“Methods of inferring trade direction can be classified as: tick tests, which use changes in trade prices; the quote method, which compares trade prices to quotes;” (Finucane, 2000, p. 557)
</mark>


**Synonyms**
1. Broader term is **trade site classification** = assign the side to a to a transaction and differentiate between buyer- and seller-initiated transactions
2. It's also sometimes called trade sign classification

**Dominating approaches:**
- There is no universally superior rule. “As ([[@easleyDiscerningInformationTrade2016]]) note, each trade classification rule may demonstrate both strengths and weakness, depending on the underlying market characteristics.” (Bowe et al., 2018, p. 30)
- Make a table with recent studies?

**Bulked classification:**
“Similarly to the tick rule, bulk classification only requires tick data to infer trade direction. Instead of classifying on a trade-bytrade basis, however, bulk classification determines the share of buys and sells of a chunk of aggregated trading volume” ([Pöppe et al., 2016, p. 167](zotero://select/library/items/5A83SDDB)) ([pdf](zotero://open-pdf/library/items/4XIK47X6?page=3&annotation=X7FLEJPM))

- “They ([[@easleyDiscerningInformationTrade2016]]) vable trading data. Ideally, we would like to specify the data generating processes for both the underlying unobservable variables and subsequently for the observed data, conditional on the realisations of the underlying unobservable data.” ([[@boweNewClassicalBayesian]], 2018, p. 14)
- “They claim that every trade classification algorithm can be regarded as an approximation to this Bayesian approach, and that their bulk volume classification (BVC) methodology is conceptually closer to this ideal than traditional approaches such as the Tick rule, since BVC assigns a probability to a given trade being either a buy or sell.” ([[@boweNewClassicalBayesian]], p. 14)

- Submitters of market orders are called liquidity demanders, while submitters of limit orders stored stored in the book are liquidity providers.
- The BVC paper ([[@easleyDiscerningInformationTrade2016]]) treats trade classification as a probabilistic trade classificatio nproblem. Incorporate this idea into the classical rules section.
- BVC is illsuited for my task, as we require to sign each trade? (see [[@panayidesComparingTradeFlow2014]])
- Algorithms like LR, tick rule etc. are also available in bulked versions. See e. g., [[@chakrabartyEvaluatingTradeClassification2015]] for a comparsion in the stock market. 
- “the bulk volume classification (BVC) methodology and tick rules. Tick rule approaches use simple movements in trade prices (upticks or downticks) to classify a trade as either a buy or a sell. The 2 This problem is also particularly acute in the new swap trading markets. The Dodd-Frank Wall Street Reform and Consumer Protection Act currently requires reporting of nonblock trades to the Swap Data Repository, but current reporting rules allow a 30-minute delay. So there is no way to determine the correct order of trades. See also Ding, Hanna, and Hendershot (2014) for evidence on how speed differences between proprietary feeds and the consolidated tape complicate knowing current quotes. 3 See Hasbrouck (2013) for an excellent analysis of quote volatility and its implications. 4 These changes also mean that trade information will not be linked with other variables of interest such as trader identity. For example, Lee and Radhakrishnan (2000) and Campbell, Ramdorai, and Schwartz (2008) propose size cutoff rules on trades that they argue identify institutional trading. Even using data from the year 2000, Campbell, Ramdorai, and Schwartz note problems in identification arising from what they suspect was algorithmic trading. With trade sizes now all collapsing to minimum levels, and institutions trading dynamically with limit orders, inferring trader identity from trade size is a daunting task. bulk volume technique, which was first applied in Easley, Lopez de Prado, and O’Hara (2011), aggregates trades over short time or volume intervals and then uses a standardised price change between the beginning and end of the interval to approximate the percentage of buy and sell order flow. Each of these techniques maps observable data into proxies for trading intentions, but how well any of these approaches works in the new high frequency world is unclear.” ([[@easleyDiscerningInformationTrade2016]],, p. 270)

“The BVC approach relies on order flows, not individual orders, and is agnostic about what the underlying information has to be. Its statistical basis is more forgiving with respect to the data difficulties (i.e., time stamp issues, orders out of sequence, massive data bases) characteristic of modern markets. As such, BVC can be a useful addition to the microstructure tool kit.” ([[@easleyDiscerningInformationTrade2016]], p. 271)