Tags: #trade-classification 


- Why is there a need for hybrid classification rules? What are the problems of the tick and quote rule?
- The previous trade classification rules are applicable to certain trades or come with their own drawbacks e. g., they perform poorly for a greater distance to quotes. To mitigate the disadvantages, rules are combined to hybrids through ensembeling. In certain cases through stacking.
- “Sophisticated algorithms combine quote and tick rule: trades at the midpoint are always classified by the tick rule and trades at best bid or best ask by the quote rule. The three most common algorithms differ in the choice of how to split up the remaining trades between quote and tick rule, as illustrated in [[#^3d69f3]]” ([[@poppeSensitivityVPINChoice2016]]; p. 9)
- “The paper further shows that, while the Lee and Ready (1991) algorithm has been the default choice among the traditional trade classification algorithm—possibly partly due to being automatically supplied by data vendors, partly due to its simplicity—the similar simplistic algorithms of Chakrabarty et al. (2007) and Ellis et al. (2000) tend to perform better and may be preferred in certain applications.” (Jurkatis, 2022, p. 23)
- use the problems of the single tick test to motivate extended rules like EMO / LR?
- that lead to a fine-grained  fragmentation?

![[viz-rules.png]]
(similar to [[@poppeSensitivityVPINChoice2016]])  ^3d69f3
“Fig. 1. Classification algorithms. This chart illustrates the functioning of three different trade-by-trade classification algorithms: LR by Lee and Ready (1991), EMO by Ellis et al. (2000) and CLNV by Chakrabarty et al. (2007).” ([Pöppe et al., 2016, p. 167](zotero://select/library/items/5A83SDDB)) ([pdf](zotero://open-pdf/library/items/4XIK47X6?page=3&annotation=8XUJ32R2))

“Sophisticated algorithms combine the quote and tick rule: Thus, trades at the midpoint are always classified by the tick rule, and trades at the best bid or ask are classified by the quote rule. The three most common algorithms differ in how they divide the remaining trades between quote and tick rule, as illustrated in Fig. 1.” ([Pöppe et al., 2016, p. 166](zotero://select/library/items/5A83SDDB)) ([pdf](zotero://open-pdf/library/items/4XIK47X6?page=2&annotation=4A3YAHN2))


- **Bridge to ML:** 🌉 Interestingly, researchers gradually segment the decision surface starting with quote and tick rule, continuing with LR, EMO and CLNV. This is very similar to what is done in a decision tree. Could be used to motivate decision trees. All the hybrid methods could be considered as an ensemble with some sophisticated weighting scheme (look up the correct term) -> In recommender the hybrid recommender is called switching.
- Current hybrid approaches use stacking ([[@grauerOptionTradeClassification2022]] p. 11). Also, due to technical limitations. Why not try out the majority vote/voting classifier with a final estimator? Show how this relates to ML.
- In stock markets applying those philtres i. e. going from tick and quote rule did not always improve classification accuracies. The work of [[@finucaneDirectTestMethods2000]] raises critique about it in the stock market.
