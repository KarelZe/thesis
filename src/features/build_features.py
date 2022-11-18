"""
Defines feature sets.

See notebook/3.0b-feature-engineering.ipynb for details.
"""

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
]

features_categorical = ["ROOT", "OPTION_TYPE"]

features_ml = [*features_trade, *features_date, *features_option]
features_classical = [*features_trade]
