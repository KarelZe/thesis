*title:* Evaluation of the biases in execution cost estimation using trade and quote data
*authors:* Mark Peterson, Erik Sirri
*year:* 2003
*tags:* #effective-spread #trade-classification 
*status:* #📦 
*related:*
*code:*
*review:*

## Notes 📍

## Annotations 📖
Note: 

“. Examples of errors in trading cost estimates The trading cost measures considered here are the effective spread and the relative effective spread. The effective spread, which represents the round trip execution costs, less commissions, is calculated as: Effective spread ¼ 2 D ðPrice midpointÞ where D is the trade direction, þ1 for a buy, and 1 for a sell. Using only TAQ data, one must infer D: The midpoint must also be estimated because the TAQ data report the trade time, not the order submission time. As noted in Bacidore et al. (1999), execution quality is most appropriately measured setting the benchmark quote to that prevailing at order submission time. The relative effective spread is calculated as Relative effective spread ¼ Effective spread=price: Next, we demonstrate the possible limitations of trade and quote data in estimating trading costs by providing two specific examples—price improvement in minimum variation markets and quote changes prior to trade execution.” ([Peterson and Sirri, 2003, p. 261](zotero://select/library/items/8H44XMRH)) ([pdf](zotero://open-pdf/library/items/N5WH3RYR?page=3&annotation=9YNKPFT3))

“In the literature, researchers use different definitions of trade initiators based presumably on data availability. Odders-White (2000) considers the last arriving order to be the trade initiator. She can make this determination because the TORQ database includes the NYSE audit file, which contains order-entry time for both sides of the trade. Papers such as Lee (1992) and Petersen and Fialkowski (1994) consider the active side to be market orders. Kraus and Stoll (1972) consider the active side to be the side with fewer parties. Finucane (2000) and Lee and Radhakrishna (2000) note many orders cannot be unambiguously defined as buyeror seller-initiated. Finucane (2000) finds that nearly one-quarter of all trades do not occur as the result of the arrival of a market order. In his final analysis, Finucane (2000) examines trades with at least one standard non-tick sensitive buy or sell market order in the trade. Ellis et al. (2000) and Theissen (2000) take the approach of inferring trade direction fromthe side contra to the dealer.” ([Peterson and Sirri, 2003, p. 263](zotero://select/library/items/8H44XMRH)) ([pdf](zotero://open-pdf/library/items/N5WH3RYR?page=5&annotation=ULLIJ7VP))

“Because we do not have access to the NYSE audit file, we cannot define a trade initiator in the same way as those who have used TORQ data. Therefore, our approach will be to begin with all regular-way orders and exclude orders that are most likely not initiators. The following orders are excluded: (a.) limit orders that are not ‘marketable’, that is buy orders with limit price less than the ask or sell orders with limit price greater than the bid, (b.) tick sensitive orders because they usually do not initiate trades, (c.) stopped,3 or guaranteed orders, because these orders tend to be more like limit orders, and (d.) partial executions of marketable limit orders for more shares than are at the best quote and execute in multiple parts.” ([Peterson and Sirri, 2003, p. 264](zotero://select/library/items/8H44XMRH)) ([pdf](zotero://open-pdf/library/items/N5WH3RYR?page=6&annotation=5IBM4V9P))