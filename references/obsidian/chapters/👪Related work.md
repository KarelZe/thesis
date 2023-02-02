
[[👪Related works notes]]

**Trade classification on option data sets** 💸

While classical trade classification algorithms are extensively tested in the stock markets (e. g., Chakrabarty et al., 2012; Odders-White, 2000), few works have examined trade classification in option markets (Grauer et al., 2022; Savickas and Wilson, 2003). For option markets, the sole focus is on classical classification rules. Even in stock markets, machine learning has hardly been applied to trade classification. An early work of rosenthalModelingTradeDirection2012 incorporates standard trade classification rules into a logistic regression model and achieves outperformance in the stock market. 

**Trade classification using machine learning** 📊
Similarly, 

The closest work to ours, is the work of [[@ronenMachineLearningTrade2022]] and [[@fedeniaMachineLearningCorporate2021]]

[[@fedeniaMachineLearningCorporate2021]] and [[@ronenMachineLearningTrade2022]] improve upon classical rules with a random forest, a tree-based ensemble. Albeit their work considers a broad range of approaches, the selection leaves the latest advancements in artificial neural networks and ensemble learning aside. Even if the focus is on standard techniques, the unclear research agenda with regards to model selection, tuning, and testing hampers the transferability of their results to the yet unstudied option market [^1].

Their model selection remains vague and is mainly guided by computational constraints. 

Due to computational constraints, they finalize on random forests, feed-forward networks, and logistic regression.

Our work improves on their work with respect to two aspects. More specifically, 

The transferability of their results is limited.

The chosen train-test split . More over, the omitted data pre-processing, favours models methods that are not reliant on gradient descent. 

---

[[@rosenthalModelingTradeDirection2012]] (p. 5) bridges the gap between classical trade classification and machine learning by estimating a logistic regression model on lagged and unlagged features inherently used in the tick rule, quote rule, and EMO algorithm. Instead of using the rule's outcome in their discretized form, [[@rosenthalModelingTradeDirection2012]] introduces an information strength criterion, to model the proximity 






The improvement in accuracy is only minor with 2 % for Nasdaq stocks and 1.1 % for the NYSE. [[@rosenthalModelingTradeDirection2012]] (p. 15). 

Our work tries to widden this gap through the use of 



The work of [[@blazejewskiLocalNonparametricModel2005]] (p. 481 f.) compares a $k$-nearest neighbour classifier against logistic regression, as well as simple heuristics like the majority vote over past trades for signing trades at the Australian stock exchange. Their results indicate that the parametric $k$-nearest neighbour classifier improves upon a linear logistic regression in terms of classification accuracy, even when trained on fewer features. The work is unique from the aforementioned works with regard to feature set definition. 

[[@blazejewskiLocalNonparametricModel2005]] (p. 3) use no quote or trade prices, but rather the order book volumes, trade sizes, and past trade signs for classification. However, no accuracies for classical trade signing rules are reported, which impedes a comparison across different works. 

Inline with their results, we focus on non-linear models in the form gradient boosting and transformers. Additionally, our paper addresses the mentioned shortcomings by benchmarking against state-of-the-art trade classification rules. We share the idea of using the trade size, and bid and ask sizes for classification, for some of our feature sets, but greedily predict using non-historic features.

Closest to our work are two recent publications of [[@ronenMachineLearningTrade2022]] and [[@fedeniaMachineLearningCorporate2021]], as they facilitate a comparison between 

[^1:] We have contacted the authors about these concerns.

