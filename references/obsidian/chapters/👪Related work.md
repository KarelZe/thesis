
[[👪Related works notes]]

**Trade classification on option data sets** 💸

While classical trade classification algorithms are extensively tested in the stock markets (e. g., [[@chakrabartyTradeClassificationAlgorithms2012]]; [[@odders-whiteOccurrenceConsequencesInaccurate2000]], few works have examined trade classification in option markets. 

The most comprehensive study is the one of [[@grauerOptionTradeClassification2022]]

The work of , as it does not perform now rules but

The work of grauer performs a comprehensive study . As the datasets are identical to our work, the 
The reported results r, thus, they serve as a benchmark in this work.

[[@savickasInferringDirectionOption2003]] do not propose new classification rules, but compare the tick rule, quote rule, the Lee and Ready algorithm, and the EMO rule on aof options traded at the CBOE. The data set is significantly smaller, than the one studied by [[@grauerOptionTradeClassification2022]], and spans a period from July 3, 1995 - December 31, 1995. The authors report the highest accuracies for the quote rule and find that all rules perform worst when applied to index options.


**Trade classification using machine learning** 📊
[[@rosenthalModelingTradeDirection2012]] (p. 5) bridges the gap between classical trade classification and machine learning by fitting a logistic regression model on lagged and unlagged predictors inherent to the tick rule, quote rule, and EMO algorithm, as well as a sector-specific and a time-specific term. Instead of using the rule's discretized outcome as a feature, [[@rosenthalModelingTradeDirection2012]] models the rules through so-called information strength functions. The proximity to the quotes, central to the EMO algorithm, is thus modelled by a proximity function. Likewise, the information strength of the quote and tick rule is estimated as the log return between the trade price and the midpoint or the previous trade price. However, it only improves the accuracy of the EMO algorithm by a marginal $2~\%$ for Nasdaq stocks and $1.1~\%$ for NYSE stocks [[@rosenthalModelingTradeDirection2012]] (p. 15). Our work aims to improve the model by exploring non-linear estimators and minimizing data modelling assumptions.

The work of [[@blazejewskiLocalNonparametricModel2005]] (p. 481 f.) compares a $k$-nearest neighbour classifier against logistic regression, as well as simple heuristics like the majority vote over past trades for signing trades at the Australian stock exchange. Their results indicate that the parametric $k$-nearest neighbour classifier improves upon a linear logistic regression in terms of classification accuracy, even when trained on fewer features. The work is unique from the remaining works about feature set definition. Notably, [[@blazejewskiLocalNonparametricModel2005]] (p. 3) use no quote or trade prices, but rather the order book volumes, trade sizes, and past trade signs for classification. No accuracies for classical trade signing rules are reported, which impedes a comparison across different works. In line with their results, we focus on non-linear models in the form of gradient boosting and transformers. Additionally, our paper addresses the mentioned shortcomings by benchmarking against state-of-the-art trade classification rules. We share the idea of using the trade size, as well as the bid and ask sizes for classification for some of our feature sets, but greedily predict using non-historic features.

Closest to our work is a publication of [[@ronenMachineLearningTrade2022]] (1. f). Therein, the authors compare a selection of machine learning algorithms against classical trade signing rules in the bond and stock market. Their comparison is the first to consider both logistic regression, a random forest, as well as feed-forward networks. Over a wide range of feature sets the tree-based ensemble consistently outperforms by out-of-sample accuracy the tick rule and Lee and Ready algorithm, as well as all remaining machine learning models. For the TRACE and ITCH data set, their best variant of the random forest outperforms the tick rule by $8.3~\%$ and $3.3~\%$, respectively [[@ronenMachineLearningTrade2022]] (p. 57). Whilst the superiority of random forests is consistent for the bond and equity market, fitted classifiers do not transfer across markets, as diminishing accuracies for the transfer learning by model transfer indicate.

The results convincingly demonstrate the potential of machine learning, i. e., of tree-based ensembles, for trade classification. Yet, the comparability of the results is limited by the classifier's reliance on additional features beyond quote and price data. Albeit, [[@ronenMachineLearningTrade2022]] (p. 4) consider a wide range of approaches, their selection leaves the latest advancements in artificial neural networks and ensemble learning aside and is mainly guided by computational constraints. Even if the focus is on standard techniques, the unclear research agenda concerning model selection, tuning, and testing hampers the transferability of their results to the yet unstudied option market [^1]. 

In summary, machine learning has been applied successfully in the context of trade classification. To the best of our knowledge, no previous work perform machine learning-based classification in the options markets.

[^1:] Major concerns are data pre-processing favours approaches non-reliant on gradient descent, an unclear model specification, and optimization. We have contacted the authors about these concerns.

