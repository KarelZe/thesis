- combination of quote rule and tick rule. Use tick rule to classify all trades except trades at hte ask and bid at which points the quote rule is applied. A trade is classified as  abuy (sell) if it is executed at the ask (bid).
- turns the principle of LR up-side-down: apply the tick rule to all trades except those at the best bid and ask.
- EMO Rule
![[emo-rule-formulae 1.png]]
- classify trades by the quote rule first and then tick rule
- Based on the observation that trades inside the quotes are poorly classified. Proposed algorithm can improve
- They perform logistic regression to determin that e. g. , trade size, firm size etc. determines the proablity of correct classification most
- cite from [[@ellisAccuracyTradeClassification2000]]

The tick rule can be exchanged for the reverse tick rule, as previously studied in [[@grauerOptionTradeClassification2022]].
