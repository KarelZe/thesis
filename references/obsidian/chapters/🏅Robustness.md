- LR-algorithm (see [[#^370c50]]) require an offset between the trade and quote. How does the offset affect the results? Do I even have the metric at different offsets?
- Perform binning like in [[@grauerOptionTradeClassification2022]]
- Study results over time like in [[@olbrysEvaluatingTradeSide2018]]
- Are probabilities a good indicator reliability e. g., do high probablities lead to high accuracy.
- Are there certain types of options that perform esspecially poor?
- Confusion matrix
- create kde plots to investigate misclassified samples further
- ![[kde-plot-results.png]]
- What is called robustnesss checks is also refered as **slice-based evaluation**. The data is separated into subsets and your model's performance on each subset is evaluated. A reason why slice-based evaluation is crucial is Simpson's paradox. A trend can exist in several subgroups, but disappear or reverse when the groups are combined. Slicing could happen based on heuristics, errors or a slice finder (See [[@huyenDesigningMachineLearning]])
![[rankwise-correlations.png]]
(found in [[@hansenApplicationsMachineLearning]], but also other papers)

“Finucane (2000) finds that a large proportion of incorrectly classified trades are trades with zeroticks.” ([Chakrabarty et al., 2007, p. 3814](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=9&annotation=6YW8JBQ6))

“Trade size may also affect the accuracy of trade classification rules. Odders-White (2000) finds that the success rate is higher for large trades than for small trades while Ellis et al. (2000) find that large trades are more frequently misclassified than small trades” ([Chakrabarty et al., 2007, p. 3814](zotero://select/library/items/XSSKWNCJ)) ([pdf](zotero://open-pdf/library/items/VQAL9PWT?page=9&annotation=RNDU5P5Z))

- “Furthermore, the most important biases encountered in the literature have been confirmed in this study: Seller-initiated trades perform remarkably better than buyer-initiated trades. The EMO rule, and especially the MEMO rule, offer substantial improvements over LR as they have far more power for classifying trades that occurred inside the quotes. The biggest disadvantage of the TR is its poor performance for zero ticks.” (Frömmel et al., 2021, p. 9) -> How are things in [[@savickasInferringDirectionOption2003]] and [[@grauerOptionTradeClassification2022]]