
[[👪Related works notes]]

**Trade classification on option data sets** 💸

While classical trade classification algorithms are extensively tested in the stock markets (e. g., Chakrabarty et al., 2012; Odders-White, 2000), few works have examined trade classification in option markets (Grauer et al., 2022; Savickas and Wilson, 2003). For option markets, the sole focus is on classical classification rules. Even in stock markets, machine learning has hardly been applied to trade classification. An early work of rosenthalModelingTradeDirection2012 incorporates standard trade classification rules into a logistic regression model and achieves outperformance in the stock market. 

The work of , as it does not perform now rules but

The work of grauer performs a comprehensive study . As the datasets are identical to our work, the 
The reported results r, thus, they serve as a benchmark in this work.


**Trade classification using machine learning** 📊
Similarly, 


---
[[@rosenthalModelingTradeDirection2012]] (p. 5) bridges the gap between classical trade classification and machine learning by fitting a logistic regression model on lagged and unlagged predictors inherent to the tick rule, quote rule, and EMO algorithm, as well as a sector-specific and a time-specific term. Instead of using the rule's discretized outcome as a feature, [[@rosenthalModelingTradeDirection2012]] models the rules through so-called information strength functions. The proximity to the quotes, central to the EMO algorithm, is thus modelled by a proximity function. Likewise, the information strength of the quote and tick rule is estimated as the log return between the trade price and the midpoint or the previous trade price. In extensive comparisons, Rosenthal's method can only marginally improve the EMO algorithm by 2 % out-of-sample accuracy for stocks traded at the Nasdaq and by 1.1 % for stocks at the NYSE [[@rosenthalModelingTradeDirection2012]] (p. 15). We hypothesize, that the observed performance is bound by the linearity assumption in the logistic regression and that the extensive modelling assumptions contribute to the model's bias. As an alternative, our work explores estimators, capable of capturing non-linearities in the data. Also, we keep data modelling to a minimum to mitigate any bias.

The work of [[@blazejewskiLocalNonparametricModel2005]] (p. 481 f.) compares a $k$-nearest neighbour classifier against logistic regression, as well as simple heuristics like the majority vote over past trades for signing trades at the Australian stock exchange. Their results indicate that the parametric $k$-nearest neighbour classifier improves upon a linear logistic regression in terms of classification accuracy, even when trained on fewer features. The work is unique from the remaining works about feature set definition. Notably, [[@blazejewskiLocalNonparametricModel2005]] (p. 3) use no quote or trade prices, but rather the order book volumes, trade sizes, and past trade signs for classification. No accuracies for classical trade signing rules are reported, which impedes a comparison across different works. In line with their results, we focus on non-linear models in the form of gradient boosting and transformers. Additionally, our paper addresses the mentioned shortcomings by benchmarking against state-of-the-art trade classification rules. We share the idea of using the trade size, as well as the bid and ask sizes for classification for some of our feature sets, but greedily predict using non-historic features.

Closest to our work is a publication of [[@ronenMachineLearningTrade2022]] (1. f). Therein, the authors compare a selection of machine learning-based predictors against classical trade signing rules in the bond and stock market. Their comparison is the first to consider both logistic regression, random forest, as well as feed-forward networks. Over a wide range of feature sets the tree-based ensemble consistently outperforms the tick rule and Lee and Ready algorithm, as well all remaining machine learning models by out-of-sample accuracy. For the TRACE and ITCH data set, their best variant of the random forest outperforms the tick rule by $8.3~\%$ and $3.3~\%$, respectively [[@ronenMachineLearningTrade2022]] (p. 57). Whilst the superiority of random forests is consistent for the bond and equity market, fitted classifiers do not transfer across markets, as diminishing accuracies for the transfer learning by model transfer indicate. 

The results convincingly demonstrate the potential of machine learning, i. e., of tree-based ensembles, for trade classification. Yet, their comparison against classical rules remains biased, as all classifiers are trained on substantially more features, than just the quote and price data used in the tick rule or the Lee and Ready algorithm.

Albeit, [[@ronenMachineLearningTrade2022]] (p. 4) consider a wide range of approaches, their selection leaves the latest advancements in artificial neural networks and ensemble learning aside and is mainly guided by computational constraints. Even if the focus is on standard techniques, the unclear research agenda concerning model selection, tuning, and testing hampers the transferability of their results to the yet unstudied option market [^1]. 

In summary, machine learning has been applied successfully in the context of trade classification. To the best of our knowledge, no previous work perform machine learning-based classification in the options markets.

[^1:] Major concerns are data pre-processing favours approaches non-reliant on gradient descent, an unclear model specification, and optimization. We have contacted the authors about these concerns.

