"""Defines feature sets.

See notebook/3.0b-feature-engineering.ipynb for details.
"""

from typing import List, Tuple

features_option = [
    "STRK_PRC",
    "ttm",
    "option_type",
    "issue_type",
    "root",
    "myn",
    "day_vol",
]

# https://github.com/KarelZe/thesis/blob/main/notebooks/
# 3.0a-mb-explanatory_data_analysis.ipynb
features_categorical: List[Tuple[str, int]] = [
    ("option_type", 2),
    ("issue_type", 6),
    ("root", 9107),  # + 1 for UNK which may be in val and test set
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


features_ml = [*features_classical_size, *features_option]
