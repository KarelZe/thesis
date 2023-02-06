**Why:**
The goal of my feature set definition is: 
1. Have a minimal feature set required to calculate LR, EMO or CLNV. →  Makes it easier to transfer our method to other markets.
2. Extend the minimal feature set for date-time features. →  Makes it easy to transfer our method to other markets, but also takes into account temporal info.
3. Have an extended feature set to calculate the SOTA of Grauer et. al. → Makes it easy to compare with the previous baseline.
4. Add time and option features. → Makes sense, as we look at an option data set. Temporal features are easy to derive. 
5. Scaling is typically not required for gradient boosting, but is useful for neural networks. But both $z$-scaling and min-max-scaling don't change the distribution of data. (see [here.](https://stats.stackexchange.com/a/562204/351242)).
6. Do not discretize features just because it's done in rules due to hard hard borders. (see [[@kuhnFeatureEngineeringSelection2020]])(p. 130) We use ratios or differences instead. This is needed as some models can not synthesize ratios or differences (see [[@heatonEmpiricalAnalysisFeature2016]])

**Other research:**
- Add frequency of trade and order imbalances as features. Came up with this idea when reading [[@aitkenIntradayAnalysisProbability1995]]
- Some feature ideas like order imbalance could be adapted from [[@aitkenIntradayAnalysisProbability1995]].
- [[@blazejewskiLocalNonparametricModel2005]] use more previous trades to a $k$-nn search. Not feasible, as my data set only contains the previous trade and it misses rather often already.
- [[@ronenMachineLearningTrade2022]] suggest to use models that can handle time series components. This would limit our choices. Thus we use feature engineering to induce a notion of time into our models.

<mark style="background: #ABF7F7A6;">TODO: Add to each feature, where it has been used.
TODO: Point out some interesting features here in the text.
TODO: We are less concerned about providing redundant data to the model, as trees do not base their splitting process on correlation. Also, with neural nets being a universal function approximator.
TODO: explain why we don't discretize / binarize features, other than classical models. See [[@kuhnFeatureEngineeringSelection2020]]
TODO: Explain why we don't include the results of the classical rules themselves, due to redundant low-precision encoding. May give an example explaining for decision trees.
TODO: Adress the usefullness of the engineered features. Sketch the results of the *adversarial validation*. Which features are very different in the training set and the validation set? Which features are most important in adversarial validation?
TODO: Plot distributions of features from training and validation set. Could also test using https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test test if samples are drawn from the same distribution.
TODO: Economic intuition:  [[@zhuClusteringStructureMicrostructure2021 1]]
TODO: For different types of spreads see [[@zhuClusteringStructureMicrostructure2021 1]]
TODO: Introduce the word "feature crosses" and motivation in deep learning https://stats.stackexchange.com/questions/349155/why-do-neural-networks-need-feature-selection-engineering/349202#349202
https://financetrain.com/why-lognormal-distribution-is-used-to-describe-stock-prices
</mark>

| Feature               | Feature Category            | Why? | FS 1 (Classical) | FS 2 (F1 + Grauer) | FS 3 (F2 + temp) | FS 4 (F3 + Others) | Transform   |
| --------------------- | ----------------------------|----- | ---------------- | ------------------ | ---------------- | ------------------ | ----------- |
| TRADE_PRICE           | tick rule                   | See [[@leeInferringTradeDirection1991]]     | ✅               | ✅                 | ✅               | ✅                 | log         |
| price_ex_lag          | tick rule                   | See above.     | ✅               | ✅                 | ✅               | ✅                 | log         |
| price_all_lag         | tick rule                   | See above.    | ✅               | ✅                 | ✅               |✅                   | log         |
| chg_ex_lag            | tick rule                   | See above.     | ✅               | ✅                 | ✅               | ✅                 | standardize |
| chg_all_lag           | tick rule                   | See above.     | ✅               | ✅                  | ✅               |✅                   | standardize |
| price_ex_lead         | reverse tick rule           | See above.     | ✅               | ✅                 | ✅               | ✅                 | log         |
| price_all_lead        | reverse tick rule           | See above.     | ✅               | ✅                  | ✅               |✅                    | log         |
| chg_ex_lead           | reverse tick rule           | See above.     | ✅               | ✅                 | ✅               | ✅                 | standardize |
| chg_all_lead          | reverse tick rule           | See above.     | ✅               | ✅                  | ✅               |✅                    | standardize |
| BEST_BID              | quote rule                  | See above.   | ✅               | ✅                  | ✅               | ✅                   | log         |
| bid_ex                | quote rule                  | See above.    | ✅               | ✅                 | ✅               | ✅                 | log         |
| BEST_ASK              | quote rule                  | See above.    | ✅               | ✅                  | ✅               | ✅                    | log         |
| mid_ex                | mid quote 🆕                | See above.     |                  |                     |                  |                    | log         |
| mid_best              | mid quote 🆕                | See above.     |                  |                     |                  |                    | log         |
| ask_ex                | quote rule                   | See [[@leeInferringTradeDirection1991]]     | ✅               | ✅                 | ✅               | ✅                 | log         |
| bid_ask_ratio_ex      | Ratio of ask and bid 🆕      | ?     |                  | ✅                 | ✅               | ✅                 | standardize |
| spread_ex             | Absolute spread 🆕           | ?     |                  |                     |                  |                    | standardize |
| spread_best           | Absolute spread 🆕           | ?     |                  |                     |                   |                   | standardize |
| price_rel_nbb         | Tradeprice rel to nbb 🆕     | Relates trade exchange with nation-wide best.     |                  | ✅                 | ✅               | ✅                 | standardize |
| price_rel_nbo         | Tradeprice rel to nbo 🆕     | See above.     |                  | ✅                 | ✅               | ✅                 | standardize |
| prox_ex               | EMO / CLNV                   | Most important predictor in [[@ellisAccuracyTradeClassification2000]] and [[@chakrabartyTradeClassificationAlgorithms2012]]    | ✅               | ✅                 | ✅                | ✅                | standardize|
| prox_best             | EMO / CLNV                   | See above.     | ✅                | ✅                 | ✅                |✅                 | standardize |
| bid_ask_size_ratio_ex | Depth rule                   | See [[@grauerOptionTradeClassification2022]]      |                  | ✅                 | ✅               | ✅                 | standardize |
| bid_size_ex           | Depth rule / Trade size rule | See above.    |                  | ✅                 | ✅               | ✅                 | standardize |
| ask_size_ex           | Depth rule / Trade size rule | See above.     |                  | ✅                 | ✅               | ✅                 | standardize |
| rel_bid_size_ex       | Trade size rule              | See above.     |                  | ✅                 | ✅               | ✅                 | standardize |
| rel_ask_size_ex       | Trade size rule              | See above.     |                  | ✅                 | ✅               | ✅                 | standardize |
| TRADE_SIZE            | Trade size rule              | See above.     |                  | ✅                 | ✅               | ✅                 | standardize |
| STR_PRC               | option                       | ?     |                  |                    |                  | ✅                 | log         |
| day_vol               | option                       | ?     |                  |                    |                  | ✅                 | log         |
| bin_root              | option 🦺(many `UNKWN`)        | ?     |                  |                    |                  | ✅                 | binarize    |
| time_to_maturity      | option                       | ?     |                  |                    |                  | ✅                 | standardize |
| moneyness             | option                       | ?     |                  |                    |                  | ✅                 | standardize |
| bin_option_type       | option                       | ?     |                  |                    |                  | ✅                 | binarize    |
| bin_issue_type        | option                       | See [[@ronenMachineLearningTrade2022]]. Learn temporal patterns. Data is ordered by time.      |                  |                    |                  | ✅                 | binarize    |
| date_month_sin        | date                         | See above.     |                  |                    | ✅               | ✅                 | pos enc     |
| date_month_cos        | date                         | See above.     |                  |                    | ✅               | ✅                 | pos enc     |
| date_day_sin          | date                         | See above.     |                  |                    | ✅               | ✅                 | pos enc     |
| date_day_cos          | date                         | See above.     |                  |                    | ✅               | ✅                 | pos enc     |
| date_weekday_sin      | date                         | See above.     |                  |                    | ✅               | ✅                 | pos enc     |
| date_weekday_cos      | date                         | See above.     |                  |                    | ✅               | ✅                 | pos enc     |
| date_time_sin         | date                         | See above.     |                  |                    | ✅               | ✅                 | pos enc     |
| date_time_cos         | date                         | See above.     |                  |                    | ✅               | ✅                 | pos enc     |
| date_year             | date 🦺(uniformative)        | See above.     |                  |                    |                  |                    | None        |

