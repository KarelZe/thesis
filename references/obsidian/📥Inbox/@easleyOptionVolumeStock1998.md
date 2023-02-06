*title:* Option Volume and Stock Prices: Evidence on Where Informed Traders Trade
*authors:* David Easley, Maureen O'Hara, P.S. Srinivas
*year:* 1997
*tags:* #cboe #trade-classification #lr 
*status:* #📦 
*related:*
- [[@grauerOptionTradeClassification2022]]
- [[@leeInferringTradeDirection1991]]

## Note

- Authors apply the LR algorithm ([[@leeInferringTradeDirection1991]]) to CBOE data.

- They argue most data bases e. g. Berkley Options Data Base / data sets are missing the indicator whether a trade is buyer-initiated or seller-iniated. So researchers have to rely on quote or trade information.

- They make observations about the lag in LR algorithm for CBOE data.

- They observe the percentage of trades at the spread midpoint shows a strong downward trend (some time in the '80s). Also, the number the number of buys and sells isn't equal. This indicates that options are more actively bought rather than sold. ➡️ Test if this is also true for my sample. Might require up-/ or down sampling. Ground truth is **unknown**!.

## Annotations

“The Berkeley Options Data Base does not classify trades as buyer-initiated or seller-initiated. This classification must be done using quote and trade information.” ([Easley et al., 1998, p. 453](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=23&annotation=GXHQMKIW))

“For researchers using transactions data, this classification problem is ubiquitous, and a cursory review of empirical papers using equity transaction data reveals many trade classification techniques. Lee and Ready (1991) present an excellent survey of techniques currently in use and evaluate their efficiency using NYSE transactions data.” ([Easley et al., 1998, p. 453](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=23&annotation=Q59K9DID))

“The approach we pursue is as follows. For each trade, the active quote is identified. Then, we identify the trade as a buy or a sell by the following algorithm: 1. Trades occurring in the lower half of the spread, at the bid or below, are classified as sells. A similar scheme is used for trades in the upper half of the spread and these are classified as buys. Trades occurring below the bid or above the ask are classified similarly. 2. Trades occurring at the midpoint of the spread are first classified using the “tick test” applied to the previous trade. If the current trade price occurs at a price higher than the previous one, it is classified as a buy (trade on an uptick). A trade on a downtick is classified as a sell. Trades unclassifiable using the previous trade are classified using the “zerouptick” or the “zero-downtick” test, which identifies the last price change and then uses the tick test strategy.” ([Easley et al., 1998, p. 453](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=23&annotation=J8S9E565))

“On the CBOE, however, the frequency of quote revisions is less than five seconds, and so using the most recent quote does not cause a bias in classification. The classification scheme we use captures the natural concept that buys tend to go at higher prices, and sales at lower prices. We caution, however, that trading and reporting protocols on the CBOE may introduce timing difficulties in the data, and to the extent that these are large, our classification scheme will be affected.” ([Easley et al., 1998, p. 453](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=23&annotation=N4CCK5WQ))

“First, the percentage of trades going off at the midpoint of the spread is far lower in CBOE trades than in NYSE trades. Vijh offers the explanation that the market design of the CBOE—a competitive dealer system—might be the cause of this phenomenon as marketmakers offer their lowest quotes, and hence are not willing to bargain on transactions prices. An alternative explanation is that if informed trading occurs on the CBOE, and, if it is harder to detect given the multiplicity of dealers, then marketmakers protect themselves by trading at quoted prices more often” ([Easley et al., 1998, p. 454](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=24&annotation=APMBYNEV))

“A second observation from Table II is that, over time, the percentage of trades executed at the spread midpoint shows a strong downward trend. This should make trade data more easily classifiable using quote data alone. Also, although studies of NYSE transactions report a roughly even split between buys and sells, it is clear that trades on the CBOE are increasingly buys. Hence, options are actively bought, rather than sold. This strengthens the argument against using transactions prices in studies of option marketstock market interactions, as these prices are more likely to be at the ask than at the bid and, hence, would bias upward the implied stock price.” ([Easley et al., 1998, p. 454](zotero://select/library/items/593W67XA)) ([pdf](zotero://open-pdf/library/items/ZBEQIUNK?page=24&annotation=2LW9T8MQ))