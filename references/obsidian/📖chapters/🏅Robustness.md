- “In light of this evidence, I recommend that researchers partition their transaction samples along the dimensions investigated in the paper and examine the impact on the results of their studies. If the "ndings are consistent across partitions, then researchers can be reasonably con"dent that their results are robust to misclassi"cation bias. On the other hand, if the results change along these dimensions without any clear explanation given the focus of the research, this suggests that misclassi"cation may be a problem. In this case, at a minimum, di!erences across partitions should be discussed along with the overall results.” ([[@odders-whiteOccurrenceConsequencesInaccurate2000]], 2000, p. 280)
- “Finally we examine the accuracy of the rules conditional on the characteristics of the trades, i.e. separately for the buy and sell side ([[@aitkenIntradayAnalysisProbability1995]], Omrane and Welsh 2016), for trades inside the quote (Ellis et al. 2000)3, and for zero versus non-zero ticks ([[@aitkenIntradayAnalysisProbability1995]], 1996; Theissen, 2001; Omrane and Welch, 2016)” (Frömmel et al., 2021, p. 7)
- “Furthermore, the most important biases encountered in the literature have been confirmed in this study: Seller-initiated trades perform remarkably better than buyer-initiated trades. The EMO rule, and especially the MEMO rule, offer substantial improvements over LR as they have far more power for classifying trades that occurred inside the quotes. The biggest disadvantage of the TR is its poor performance for zero ticks.” ([[@frommelAccuracyTradeClassification2021]], p. 9)

- analyse the classifier performance based on the $|\cdot|$ proximity of the quotes?

- LR-algorithm (see [[#^370c50]]) require an offset between the trade and quote. How does the offset affect the results? Do I even have the metric at different offsets?
- Perform binning like in [[@grauerOptionTradeClassification2022]]
- Study results over time like in [[@olbrysEvaluatingTradeSide2018]]
- Are probabilities a good indicator reliability e. g., do high probablities lead to high accuracy.
- Are there certain types of options that perform esspecially poor?
- Confusion matrix
- create kde plots to investigate misclassified samples further
- ![[kde-plot-results.png]]
- What is called robustnesss cheques is also refered as **slice-based evaluation**. The data is separated into subsets and your model's performance on each subset is evaluated. A reason why slice-based evaluation is crucial is Simpson's paradox. A trend can exist in several subgroups, but disappear or reverse when the groups are combined. Slicing could happen based on heuristics, errors or a slice finder (See [[@huyenDesigningMachineLearning]])
![[rankwise-correlations.png]]
(found in `@hansenApplicationsMachineLearning`, but also other papers)

“Finucane (2000) finds that a large proportion of incorrectly classified trades are trades with zeroticks.” ([Chakrabarty et al., 2007, p. 3814](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=9&annotation=6YW8JBQ6))

“Trade size may also affect the accuracy of trade classification rules. Odders-White (2000) finds that the success rate is higher for large trades than for small trades while Ellis et al. (2000) find that large trades are more frequently misclassified than small trades” ([Chakrabarty et al., 2007, p. 3814](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=9&annotation=RNDU5P5Z))

- “Furthermore, the most important biases encountered in the literature have been confirmed in this study: Seller-initiated trades perform remarkably better than buyer-initiated trades. The EMO rule, and especially the MEMO rule, offer substantial improvements over LR as they have far more power for classifying trades that occurred inside the quotes. The biggest disadvantage of the TR is its poor performance for zero ticks.” (Frömmel et al., 2021, p. 9) -> How are things in [[@savickasInferringDirectionOption2003]] and [[@grauerOptionTradeClassification2022]]


- “Overall, there is a monotonic relationship: better classification for smaller trades. For example, trades of less than 200 shares are correctly classified 81.73% of the time compared with 77.85% for trades over 10,000 shares. This is in contrast to Odders-White's (2000) finding that accuracy is lower for smaller NYSE trades.” ([Ellis et al., 2000, p. 536](zotero://select/library/items/54BPHWMV)) ([pdf](zotero://open-pdf/library/items/TTB4YUW6?page=9&annotation=SDMJDLDI))
- “Conditioning on the location of the trade, we find trade classification accuracy increases for larger trades, but overall accuracy is lower.” ([Ellis et al., 2000, p. 536](zotero://select/library/items/54BPHWMV)) ([pdf](zotero://open-pdf/library/items/TTB4YUW6?page=9&annotation=E56HF6MG))
- “The regression shows trade size, firm size, trading speed, and quoting speed are each less significant in determining the probability of correct classification than is proximity to the quotes. The probability of correct classification increases with trade size, decreases with firm size, increases with the time between trades (less rapid trading), and increases with the time between a quote update and a trade.” ([Ellis et al., 2000, p. 539](zotero://select/library/items/54BPHWMV)) ([pdf](zotero://open-pdf/library/items/TTB4YUW6?page=12&annotation=96FSFA7I))


![[viz-robustness.png]]
(from [[@carionEndtoEndObjectDetection2020]]; 12)

![[rules-across-underlyings.png]]


“Trade size may also affect the accuracy of trade classification rules. Odders-White (2000) finds that the success rate is higher for large trades than for small trades while Ellis et al. (2000) find that large trades are more frequently misclassified than small trades” (Chakrabarty et al., 2007, p. 3814)