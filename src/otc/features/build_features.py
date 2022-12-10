"""
Defines feature sets.

See notebook/3.0b-feature-engineering.ipynb for details.
"""

from typing import List

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

features_categorical: List[str] = ["ROOT", "OPTION_TYPE"]

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
    "TRADE_SIZE",
    "bid_size_ex",
    "ask_size_ex",
    "rel_ask_ex",
    "rel_bid_ex",
    "BEST_rel_bid",
    "BEST_rel_ask",
    "bid_ask_ratio_ex",
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
]
