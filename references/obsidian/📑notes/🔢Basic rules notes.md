Tags: #trade-classification 



Denote sells with $0$ and buys with $1$.

“The trade indicator is a binary variable stating whether the buyer or seller of an asset has initiated the trade by submitting a market order or an immediately executed limit order.” (Frömmel et al., 2021, p. 4)


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

- “They ([[@easleyDiscerningInformationTrade2016]]) vable trading data. Ideally, we would like to specify the data generating processes for both the underlying unobservable variables and subsequently for the observed data, conditional on the realizations of the underlying unobservable data.” ([[@boweNewClassicalBayesian]], 2018, p. 14)
- “They claim that every trade classification algorithm can be regarded as an approximation to this Bayesian approach, and that their bulk volume classification (BVC) methodology is conceptually closer to this ideal than traditional approaches such as the Tick rule, since BVC assigns a probability to a given trade being either a buy or sell.” ([[@boweNewClassicalBayesian]], p. 14)

- Submitters of market orders are called liquidity demanders, while submitters of limit orders stored stored in the book are liquidity providers.
- The BVC paper ([[@easleyDiscerningInformationTrade2016]]) treats trade classification as a probabilistic trade classificatio nproblem. Incorporate this idea into the classical rules section.
- BVC is illsuited for my task, as we require to sign each trade? (see [[@panayidesComparingTradeFlow2014]])
- Algorithms like LR, tick rule etc. are also available in bulked versions. See e. g., [[@chakrabartyEvaluatingTradeClassification2015]] for a comparsion in the stock market. 

**Views on the trade initator:**
- There is no single definition / understanding for the one who initiates trades. [[@olbrysEvaluatingTradeSide2018]] distinguish / discuss immediacy and initiator.
“We will consequently follow an approach employed in the paper of Miłob ̨ edzki and Nowak (2018) and classify a particular trade as initiated by a buyer (seller) if its resulting price is equal to the best ask (bid) or higher (lower) than that. It is pertinent to note that another attitudes are to be found in the literature: i.e., an investor is assessed to be a trade initiator if he (she) places either a market order or places his (her) order chronologically last. A trade initiator as also understood as the last party to agree to the trade or the party whose decision causes the trade to occur (Lee and Radhakrishna 2000; Odders-White 2000; Chakrabarty et al. 2012).” ([[@nowakAccuracyTradeClassification2020]], p. 66)
“Since the times and IDs of each order and trade are available for our propriety dataset from the BIST, we use the more accurate chronological approach as the benchmark for the true initiator of a trade.3 “3 When trader IDs are not available (Lee and Ready, 1991), the immediacy approach defines the trade initiator as the trader who demands immediate execution (i.e., places a market order) and the non-initiator as the trader who is a liquidity provider but does not require immediate execution (i.e., places a limit order).” ([[@aktasTradeClassificationAccuracy2014]], 2014, p. 261)
“When trade IDs are not available, a trade is assumed to be initiated by the trader whose market order has been executed against a standing limit order. The advantage of this immediacy approach is that it considers both dimensions of liquidity for trades that match a market with a limit order that are on opposite sides of the market. However, as noted by Odders-White (2000), this approach cannot identify the actual trade initiator for crossed market orders, limit–limit order matches and stopped market orders.” (Aktas and Kryzanowski, 2014, p. 267)
“When trade IDs are available, the chronological approach is used where the trade initiator is identified as being the trader who places an order last chronologically. The two-part rationale behind this approach is that: (i) the first-in party to the trade acts as the liquidity provider at its chosen price; and (ii) the last-in party pays the “immediacy premium” for the rapid execution of the trade. The advantage of this approach is that it considers both dimensions of liquidity for a wider set of order type pairings.” (Aktas and Kryzanowski, 2014, p. 267)
“Given different conceptual benchmarks, it is not surprising that they arrive at different conclusions about the reliability of using the LR algorithm for classifying trades when the LR algorithm classifies most (majority of) trades involving short sales as being buyer-initiated for stocks (not) subject to either the uptick or inside bid rule depending upon the trading venue examined.” (Aktas and Kryzanowski, 2014, p. 267)
- There are different views of what is considered as buyer / seller iniated i. e. [[@odders-whiteOccurrenceConsequencesInaccurate2000]] vs. [[@ellisAccuracyTradeClassification2000]]
(see [[@theissenTestAccuracyLee2000]] for more details)
- Different views on iniatiting party (i. e. [[@odders-whiteOccurrenceConsequencesInaccurate2000]] vs. [[@chakrabartyTradeClassificationAlgorithms2012]]) (see [[@aktasTradeClassificationAccuracy2014]] for more details)
- Submitters of market orders are called liquidity demanders, while submitters of limit orders stored stored in the book are liquidity providers.

“the bulk volume classification (BVC) methodology and tick rules. Tick rule approaches use simple movements in trade prices (upticks or downticks) to classify a trade as either a buy or a sell. The 2 This problem is also particularly acute in the new swap trading markets. The Dodd-Frank Wall Street Reform and Consumer Protection Act currently requires reporting of nonblock trades to the Swap Data Repository, but current reporting rules allow a 30-minute delay. So there is no way to determine the correct order of trades. See also Ding, Hanna, and Hendershot (2014) for evidence on how speed differences between proprietary feeds and the consolidated tape complicate knowing current quotes. 3 See Hasbrouck (2013) for an excellent analysis of quote volatility and its implications. 4 These changes also mean that trade information will not be linked with other variables of interest such as trader identity. For example, Lee and Radhakrishnan (2000) and Campbell, Ramdorai, and Schwartz (2008) propose size cutoff rules on trades that they argue identify institutional trading. Even using data from the year 2000, Campbell, Ramdorai, and Schwartz note problems in identification arising from what they suspect was algorithmic trading. With trade sizes now all collapsing to minimum levels, and institutions trading dynamically with limit orders, inferring trader identity from trade size is a daunting task. bulk volume technique, which was first applied in Easley, Lopez de Prado, and O’Hara (2011), aggregates trades over short time or volume intervals and then uses a standardized price change between the beginning and end of the interval to approximate the percentage of buy and sell order flow. Each of these techniques maps observable data into proxies for trading intentions, but how well any of these approaches works in the new high frequency world is unclear.” ([[@easleyDiscerningInformationTrade2016]],, p. 270)

“The BVC approach relies on order flows, not individual orders, and is agnostic about what the underlying information has to be. Its statistical basis is more forgiving with respect to the data difficulties (i.e., time stamp issues, orders out of sequence, massive data bases) characteristic of modern markets. As such, BVC can be a useful addition to the microstructure tool kit.” ([[@easleyDiscerningInformationTrade2016]], p. 271)