([[@ellisAccuracyTradeClassification2000]] 536) examine the performance of the previous algorithms for stocks traded at NASDAQ. By analysing miss-classified trades with regard to the proximity of the trade to the quotes, they observe, that the [[🔢Quote rule]] and by extension of the [[🔢LR algorithm]] performs particularly well at classifying trades executed at the bid and ask price but trail the performance of the tick rule for trades inside or outside the spread ([[@ellisAccuracyTradeClassification2000]] 535-536). The authors combine these observations into a single rule, known as the EMO algorithm.

As such, the EMO algorithm ([[@ellisAccuracyTradeClassification2000]] 540) extends the tick rule by classifying trades at the quotes using the quote rule, and all other trades with the tick test. Formally, the classification rule is given by:
$$
  \begin{equation}

    \text{Trade}_{i,t}=

    \begin{cases}
      1, & \text{if}\ P_{i, t} = A_{i, t} \\
      0, & \text{if}\ P_{i, t} = B_{i, t}  \\
	  \operatorname{tick}(), & \text{else}.
    \end{cases}
  \end{equation}
$$
Equation (...) embeds both the quote and tick rule. As trades off the quotes are classified by the tick rule, the algorithm's overall success rate is dominated by the tick test. For option markets this dependence caused the performance to lag behind quote-based approaches ([[@savickasInferringDirectionOption2003]] 891), ([[@grauerOptionTradeClassification2022]] 21), contrary to the successful adaption in the stock market ([[@ellisAccuracyTradeClassification2000]] 541)([[@chakrabartyTradeClassificationAlgorithms2007]] 3818). In ([[@grauerOptionTradeClassification2022]] 31-35) the authors achieve minor improvements in classification accuracy on option exchange data by applying the reverse tick test (see chapter [[🔢Tick test]]) as a proxy for the tick test.

**Notes:**
[[🔢EMO rule notes]]
