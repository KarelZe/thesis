title: Trade classification algorithms for electronic communications network trades
authors: Bidisha Chakrabarty, Bingguang Li, Vanthuan Nguyen, Robert A. Van Ness
year: 2007
tags : #trade-classification #lr #tick-rule #quote-rule #CLVN
status : #📦 
related:
- [[@leeInferringTradeDirection1991]]
- [[@ellisAccuracyTradeClassification2000]]

# Annotations

“From the quote records, we construct the time-stamped National Best Bid and Offer (NBBO) series for each of the sample stocks. Using only the TAQ trades and the NBBO quotes, we apply the LR, EMO and tick rules to identify trade direction. This is done using the data for April 2005 only.” ([Chakrabarty et al., 2007, p. 3809](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=4&annotation=EDLDSKAM))

“We build our intuition regarding the performance of the various classification algorithms from the April sample, and then test them, and our proposed alternative algorithm, on the May–June sample. This is done to ensure out-of-sample validity of our results.6” ([Chakrabarty et al., 2007, p. 3809](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=4&annotation=363NFXKV))

“We compare the accuracy rates of various algorithms in classifying ECN trades. We divide trades according to their price distribution relative to quotes. We expect the classification rules to perform better when trades occur at the ask or bid. When trades receive price improvement, buys and sells will execute at prices inside the quotes. In these instances, not only does the quote rule meet with some challenge, the tick rule will also be more difficult as buys (sells) will more likely occur on downticks (upticks).” ([Chakrabarty et al., 2007, p. 3811](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=6&annotation=6NIJNJ58))

“When comparing the performance of the tick and quote rules for each decile, a clear pattern arises showing which rule performs better in each decile. Uniformly, the quote rule is superior to the tick rule when transaction prices are closer to the quotes (in deciles A3–A5 and B3–B5). However, when transaction prices are closer to the midpoints (in deciles A1, A2, B1, and B2), the performance of the tick rule is better than that of the quote rule.” ([Chakrabarty et al., 2007, p. 3811](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=6&annotation=NEYHHSVW))

“When trades receive price improvement, classifications are problematic as buys (sells) execute at prices away from the ask (bid). Dividing inside trades into deciles shows that the quote rule is better for trades closer to the ask and the bid and the tick rule does better when transaction prices are closer to the midpoint.” ([Chakrabarty et al., 2007, p. 3812](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=7&annotation=ASB83EBG))

“Our algorithm is a hybrid of the tick and quote rules; it uses the quote rule when transaction prices are closer to the ask and bid and uses the tick rule when transaction prices are closer to the midpoint. Specifically, we divide the spread into deciles (10% increments). We use the quote rule if transaction prices are in the top (A5, A4 and A3) and bottom (B5, B4 and B3) three deciles. If transaction prices are in the two deciles above the midpoint or two deciles below the midpoint, we use the tick rule. For trades at the quotes, we use the quote rule, since the results from Table 1 Panel A show that the quote rule is better at these points. Fig. 1 illustrates this alternative algorithm.” ([Chakrabarty et al., 2007, p. 3812](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=7&annotation=4QD7Q4NX))

“Finucane (2000) finds that a large proportion of incorrectly classified trades are trades with zeroticks.” ([Chakrabarty et al., 2007, p. 3814](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=9&annotation=6YW8JBQ6))

“Trade size may also affect the accuracy of trade classification rules. Odders-White (2000) finds that the success rate is higher for large trades than for small trades while Ellis et al. (2000) find that large trades are more frequently misclassified than small trades” ([Chakrabarty et al., 2007, p. 3814](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=9&annotation=RNDU5P5Z))

“The evidence for the success of our algorithm is based on INET data. While INET handles the largest share of NASDAQ stocks, we also verify the success rate of our algorithm using data from the ArcaEx, the second largest venue for NASDAQ stocks. We obtain one week’s complete order book and trade data (September 12–16, 2005) for a sample of the top 100 (by trading volume) NASDAQ stocks traded on the ArcaEx, and compare the accuracy rates of the various trade classification algorithms with the actual numbers of buy/sell-initiated trades.” ([Chakrabarty et al., 2007, p. 3815](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=10&annotation=XIFPZQET))

“Studies show that apart from the relative distance of the trade price to the quotes, there are other factors that affect classification accuracy. We employ a logistic regression to assess the marginal effects of each factor on the probability of correct classification.” ([Chakrabarty et al., 2007, p. 3817](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=12&annotation=KNPQG69B))

“sing INET data, we assess the accuracy of the tick, LR, and EMO rules in identifying trade direction for NASDAQ stocks traded on an ECN. We show the limited success of these algorithms in classifying trade direction, especially for trades that occur inside the quotes. We propose an alternative algorithm that performs better than the current rules for classifying ECN trades, especially for trades inside the quotes.” ([Chakrabarty et al., 2007, p. 3820](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=15&annotation=88LWZ5QI))

“The overall success rates of the LR, EMO, and tick rules are 74.42%, 75.80%, and 75.40%, respectively.  ([Chakrabarty et al., 2007, p. 3821](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=16&annotation=I4A9CCUN))
