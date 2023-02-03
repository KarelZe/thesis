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