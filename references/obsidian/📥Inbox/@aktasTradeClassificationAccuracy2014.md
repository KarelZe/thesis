*title:* Trade Classification Accuracy for the Bist
*authors:* Osman Ulas Aktas, Lawrence Kryzanowski
*year:* 2014
*tags:* #trade-classification #lr
*status:* #📦 
*related:*
- [[@odders-whiteOccurrenceConsequencesInaccurate2000]]
- [[@chakrabartyTradeClassificationAlgorithms2012]]
- [[@leeInferringTradeDirection1991]]

## Notes
- The paper investigates the results of different trade classification algorithms and not really interesting.
- However, it contains an interesting discussion / comparsion of the different views of the trader who iniated a trade (i. e. [[@odders-whiteOccurrenceConsequencesInaccurate2000]] vs. [[@chakrabartyTradeClassificationAlgorithms2012]]).

## Annotations
“When trade IDs are not available, a trade is assumed to be initiated by the trader whose market order has been executed against a standing limit order. The advantage of this immediacy approach is that it considers both dimensions of liquidity for trades that match a market with a limit order that are on opposite sides of the market. However, as noted by Odders-White (2000), this approach cannot identify the actual trade initiator for crossed market orders, limit–limit order matches and stopped market orders.” ([Aktas and Kryzanowski, 2014, p. 267](zotero://select/library/items/P92ZHJPU)) ([pdf](zotero://open-pdf/library/items/EGAFIN4U?page=9&annotation=64DI9JFR))

“When trade IDs are available, the chronological approach is used where the trade initiator is identified as being the trader who places an order last chronologically. The two-part rationale behind this approach is that: (i) the first-in party to the trade acts as the liquidity provider at its chosen price; and (ii) the last-in party pays the “immediacy premium” for the rapid execution of the trade. The advantage of this approach is that it considers both dimensions of liquidity for a wider set of order type pairings.” ([Aktas and Kryzanowski, 2014, p. 267](zotero://select/library/items/P92ZHJPU)) ([pdf](zotero://open-pdf/library/items/EGAFIN4U?page=9&annotation=6LC949RW))

“Given different conceptual benchmarks, it is not surprising that they arrive at different conclusions about the reliability of using the LR algorithm for classifying trades when the LR algorithm classifies most (majority of) trades involving short sales as being buyer-initiated for stocks (not) subject to either the uptick or inside bid rule depending upon the trading venue examined.” ([Aktas and Kryzanowski, 2014, p. 267](zotero://select/library/items/P92ZHJPU)) ([pdf](zotero://open-pdf/library/items/EGAFIN4U?page=9&annotation=HT3MUSAS))