- Why is there a need for hybrid classification rules? What are the problems of the tick and quote rule?
- The previous trade classification rules are applicable to certain trades or come with their own drawbacks e. g., they perform poorly for a greater distance to quotes. To mitigate the disadvantages, rules are combined to hybrids through ensembeling. In certain cases through stacking.
- “Sophisticated algorithms combine quote and tick rule: trades at the midpoint are always classified by the tick rule and trades at best bid or best ask by the quote rule. The three most common algorithms differ in the choice of how to split up the remaining trades between quote and tick rule, as illustrated in [[#^3d69f3]]” ([[@poppeSensitivityVPINChoice2016]]; p. 9)
- use the problems of the single tick test to motivate extended rules like EMO / LR?
- that lead to a fine-grained  fragmentation?
![[visualization-of-quote-and-tick.png]]
(image copied from [[@poppeSensitivityVPINChoice2016]])  ^3d69f3

- **Bridge to ML:** 🌉 Interestingly, researchers gradually segment the decision surface starting with quote and tick rule, continuing with LR, EMO and CLNV. This is very similar to what is done in a decision tree. Could be used to motivate decision trees. All the hybrid methods could be considered as an ensemble with some sophisticated weighting scheme (look up the correct term) -> In recommender the hybrid recommender is called switching.
- Current hybrid approaches use stacking ([[@grauerOptionTradeClassification2022]] p. 11). Also, due to technical limitations. Why not try out the majority vote/voting classifier with a final estimator? Show how this relates to ML.
- In stock markets applying those filters i. e. going from tick and quote rule did not always improve classification accuracies. The work of [[@finucaneDirectTestMethods2000]] raises critique about it in the stock market.

**Algorithms:**
![[pseudocode-of-algorithms.png]]
(found in [[👨‍👩‍👧‍👦Related Works/@jurkatisInferringTradeDirections2022]]). Overly complex description but helpful for implementation?