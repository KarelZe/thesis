The goal of my feature set definition is: Have a minimal feature set required to calculate LR, EMO or CLNV. →  Makes it easier to transfer our method to other markets.
2. Extend the minimal feature set for date-time features. →  Makes it easy to transfer our method to other markets, but also takes into account temporal info.
3. Have an extended feature set to calculate the SOTA of Grauer et. al. → Makes it easy to compare with the previous baseline.
4. Add time and option features. → Makes sense, as we look at an option data set. Temporal features are easy to derive. 
5. Scaling is typically not required for gradient boosting, but is useful for neural networks. But both $z$-scaling and min-max-scaling don't change the distribution of data. (see [here.](https://stats.stackexchange.com/a/562204/351242)).

- Think about using a frequency of trade feature or so. Also use order imbalances as features. Came up with this idea when reading [[@aitkenIntradayAnalysisProbability1995]]
- Some feature ideas like order imbalance could be adapted from [[@aitkenIntradayAnalysisProbability1995]].
- [[@ronenMachineLearningTrade2022]] suggest to use models that can handle time series components. This would limit our choices. Thus we use feature engineering to induce a notion of time into our models.

```python
features_date = [

    "date_month_sin",

    "date_month_cos",

    "date_time_sin",

    "date_time_cos",

    "date_weekday_sin",

    "date_weekday_cos",

    "date_day_sin",

    "date_day_cos",

]

  

features_option = [

    "STRK_PRC",

    "ttm",

    "bin_option_type",

    "bin_issue_type",

    "bin_root",

    "myn",

    "day_vol",

]

  

# https://github.com/KarelZe/thesis/blob/main/notebooks/

# 3.0a-mb-explanatory_data_analysis.ipynb

features_categorical: List[Tuple[str, int]] = [

    ("bin_root", 8667),

    ("bin_option_type", 2),

    ("bin_issue_type", 6),

]

  

features_classical = [

    "TRADE_PRICE",

    "bid_ex",

    "ask_ex",

    "BEST_ASK",

    "BEST_BID",

    "price_ex_lag",

    "price_ex_lead",

    "price_all_lag",

    "price_all_lead",

    "chg_ex_lead",

    "chg_ex_lag",

    "chg_all_lead",

    "chg_all_lag",

    "prox_ex",

    "prox_best",

]

  

features_size = [

    "bid_ask_size_ratio_ex",

    "rel_bid_size_ex",

    "rel_ask_size_ex",

    "TRADE_SIZE",

    "bid_size_ex",

    "ask_size_ex",

    "depth_ex",

]

  

features_classical_size = [

    *features_classical,

    *features_size,

]

  

features_ml = [*features_classical_size, *features_date, *features_option]

  

features_unused = [

    "price_rel_nbb",

    "price_rel_nbo",

    "date_year",

    "mid_ex",

    "mid_best",

    "spread_ex",

    "spread_best",

]
```

| Feature               | Feature Category             | FS 1 (Classical) | FS 2 (F1 + Grauer) | FS 3 (F2 + temp) | FS 4 (F3 + Others) | Transform   |
| --------------------- | ---------------------------- | ---------------- | ------------------ | ---------------- | ------------------ | ----------- |
| TRADE_PRICE           | tick rule                    | ✅               | ✅                 | ✅               | ✅                 | log         |
| price_ex_lag          | tick rule                    | ✅               | ✅                 | ✅               | ✅                 | log         |
| price_all_lag         | tick rule                    |                  | ✅                 | ❓               |                    | log         |
| chg_ex_lag            | tick rule                    | ✅               | ✅                 | ✅               | ✅                 | standardize |
| chg_all_lag           | tick rule                    |                  | ✅                  | ❓               |                    | standardize |
| price_ex_lead         | reverse tick rule            | ✅               | ✅                 | ✅               | ✅                 | log         |
| price_all_lead        | reverse tick rule            |                  | ✅                  | ❓               |                    | log         |
| chg_ex_lead           | reverse tick rule            | ✅               | ✅                 | ✅               | ✅                 | standardize |
| chg_all_lead          | reverse tick rule            |                  | ✅                  | ❓               |                    | standardize |
| BEST_BID              | quote rule                   |                  | ✅                  | ❓               |                    | log         |
| bid_ex                | quote rule                   | ✅               | ✅                 | ✅               | ✅                 | log         |
| BEST_ASK              | quote rule                   |                  | ✅                  | ❓               |                    | log         |
| ask_ex                | quote rule                   | ✅               | ✅                 | ✅               | ✅                 | log         |
| bid_ask_ratio_ex      | Ratio of ask and bid 🆕      |                  | ✅                 | ✅               | ✅                 | standardize |
| spread_ex             | Absolute spread 🆕           |                  | ✅                 | ✅               | ✅                 | standardize |
| spread_best           | Absolute spread 🆕           |                  | ✅                 | ✅               | ✅                 | standardize |
| price_rel_nbb         | Tradeprice rel to nbb 🆕     |                  | ✅                 | ✅               | ✅                 | standardize |
| price_rel_nbo         | Tradeprice rel to nbo 🆕     |                  | ✅                 | ✅               | ✅                 | standardize |
| prox_ex               | EMO / CLNV                   | ✅               | ✅                | ✅                | ✅                 | standardize|
| prox_best             | EMO / CLNV                   |                  | ✅                 | ✅                |✅                 | standardize |
| bid_ask_size_ratio_ex | Depth rule                   |                  | ✅                 | ✅               | ✅                 | standardize |
| bid_size_ex           | Depth rule / Trade size rule |                  | ✅                 | ✅               | ✅                 | standardize |
| ask_size_ex           | Depth rule / Trade size rule |                  | ✅                 | ✅               | ✅                 | standardize |
| rel_bid_size_ex       | Trade size rule              |                  | ✅                 | ✅               | ✅                 | standardize |
| rel_ask_size_ex       | Trade size rule              |                  | ✅                 | ✅               | ✅                 | standardize |
| TRADE_SIZE            | Trade size rule              |                  | ✅                 | ✅               | ✅                 | standardize |
| STR_PRC               | option                       |                  |                    |                  | ✅                 | log         |
| day_vol               | option                       |                  |                    |                  | ✅                 | standardize |
| ROOT                  | option                       |                  |                    |                  | ✅                 | binarize    |
| time_to_maturity      | option                       |                  |                    |                  | ✅                 | standardize |
| moneyness             | option                       |                  |                    |                  | ✅                 | standardize |
| option-type           | option                       |                  |                    |                  | ✅                 | binarize    |
| issue-type            | option                       |                  |                    |                  | ✅                 | binarize    |
| date_month_sin        | date                         |                  |                    | ✅               | ✅                 | pos enc     |
| date_month_cos        | date                         |                  |                    | ✅               | ✅                 | pos enc     |
| date_time_sin         | date                         |                  |                    | ✅               | ✅                 | pos enc     |
| date_time_cos         | date                         |                  |                    | ✅               | ✅                 | pos enc     |
| date_year             | date                         |                  |                    | ✅               | ✅                 | pos enc     |

