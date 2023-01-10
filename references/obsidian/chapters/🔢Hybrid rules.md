The previous trade classification rules are applicable to certain trades or come with their own drawbacks. To mitigate 
these,... through ensembling

Naturally, 

- use the problems of the single tick test to motivate extended rules like EMO.
- that lead to a fine-grained  fragmentation 
- What are common extensions? How do new algorithms extend the classical ones? What is the intuition? How do they perform? How do the extensions relate? Why do they fail? In which cases do they fail?
- [[@savickasInferringDirectionOption2003]]
- [[@grauerOptionTradeClassification2022]]
- Which do I want to cover? What are their theoretical properties?
- What are common observations or reasons why authors suggested extensions? How do they integrate to the previous approaches? Could this be visualised for a streamlined overview / discussion. 
- What do we find, if we compare the rules 

**Interesting observations:**
![[visualization-of-quote-and-tick 1.png]]
(image copied from [[@poppeSensitivityVPINChoice2016]]) 
- Interestingly, researchers gradually segment the decision surface starting with quote and tick rule, continuing with LR, EMO and CLNV. This is very similar to what is done in a decision tree. Could be used to motivate decision trees.
- All the hybrid methods could be considered as an ensemble with some sophisticated weighting scheme (look up the correct term) -> In recommender the hybrid recommender is called switching.
- Current hybrid approaches use stacking ([[@grauerOptionTradeClassification2022]] p. 11). Also, due to technical limitations. Why not try out the majority vote/voting classifier with a final estimator? Show how this relates to ML.
- In stock markets applying those filters i. e. going from tick and quote rule did not always improve classification accuracies. The work of [[@finucaneDirectTestMethods2000]] raises critique about it in the stock market.
![[pseudocode-of-algorithms 1.png]]
(found in [[👨‍👩‍👧‍👦Related Works/@jurkatisInferringTradeDirections2022]]). Overly complex description but helpful for implementation?