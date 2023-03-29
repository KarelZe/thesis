"""
Defines feature sets.

See notebook/3.0b-feature-engineering.ipynb for details.
"""

from typing import List, Tuple

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
    ("bin_option_type", 2),
    ("bin_issue_type", 6),
    ("bin_root", 9107),  # + 1 for UNK which may be in val and test set
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

features_exchange = [
    *[f"ASK_{i}" for i in range(1, 17)],
    *[f"BID_{i}" for i in range(1, 17)],
]

features_classical_size_exchanges = [
    *features_classical,
    *features_size,
    *features_exchange,
]

features_ml = [*features_classical_size, *features_option]


features_unused = [
    "price_rel_nbb",
    "price_rel_nbo",
    "date_year",
    "mid_ex",
    "mid_best",
    "spread_ex",
    "spread_best",
    *features_date,
]
