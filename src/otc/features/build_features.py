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
    "date_year",
]

features_option = [
    "STRK_PRC",
    "ROOT",
    "time_to_maturity",
    "OPTION_TYPE",
]

features_trade = [
    "TRADE_SIZE",
    "TRADE_PRICE",
    "BEST_ASK",
    "BEST_BID",
    "price_ex_lag",
    "price_ex_lead",
    "price_all_lag",
    "price_all_lead",
    "bid_ex",
    "ask_ex",
    "bid_size_ex",
    "ask_size_ex",
    "midpoint_ex",
    "dis_mid_ex",
    "rel_bid_size_ex",
    "rel_ask_size_ex",
    "diff_ask_bid_size_ex",
]

# https://github.com/KarelZe/thesis/blob/main/notebooks/
# 3.0a-mb-explanatory_data_analysis.ipynb
features_categorical: List[Tuple[str, int]] = [
    ("ROOT", 8667),
    ("OPTION_TYPE", 2),
    ("issue_type", 6),
]

features_ml = [*features_trade, *features_date, *features_option]

features_classical = [
    "TRADE_SIZE",
    "TRADE_PRICE",
    "BEST_ASK",
    "BEST_BID",
    "price_ex_lag",
    "price_ex_lead",
    "price_all_lag",
    "price_all_lead",
    "bid_ex",
    "ask_ex",
    "bid_size_ex",
    "ask_size_ex",
]


features_classical_size = [
    "TRADE_PRICE",
    "bid_ask_size_ratio_ex",
    "rel_bid_size_ex",
    "rel_ask_size_ex",
    "depth_ex",
    "prox_ex",
    "prox_best",
    "spread_ex",
    "spread_best",
    "bid_ask_ratio_ex",
    "price_rel_nbb",
    "price_rel_nbo",
    "chg_ex_lead",
    "chg_ex_lag",
    "chg_all_lead",
    "chg_all_lag",
    "ask_ex",
    "bid_ex",
    "BEST_ASK",
    "BEST_BID",
    "price_all_lag",
    "price_all_lead",
    "price_ex_lag",
    "price_ex_lead",
    "TRADE_SIZE",
    "bid_size_ex",
    "ask_size_ex",
]
