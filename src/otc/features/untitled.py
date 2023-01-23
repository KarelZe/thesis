import os
import gc

import gcsfs

import numpy as np
import numpy.typing as npt
import pandas as pd

import wandb

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score


# connect to weights and biases
run = wandb.init(project="thesis", job_type="dataset-creation", entity="fbv")


def sin_encode(x: pd.Series, period: int) -> npt.NDArray:
    """
    Encode a series with a sin function.

    Args:
        x (pd.Series): input series
        period (int): frequency

    Returns:
        npt.NDArray: encoded values
    """
    return np.sin(x * 2 * np.pi / period)


def cos_encode(x: pd.Series, period: int) -> npt.NDArray:
    """
    Encode a series with a sin function.

    Args:
        x (pd.Series): input series
        period (int): frequency

    Returns:
        npt.NDArray: encoded values
    """
    return np.cos(x * 2 * np.pi / period)


scaler = StandardScaler()
oe_option_type = OrdinalEncoder(
    unknown_value=-1, dtype=int, handle_unknown="use_encoded_value"
)
oe_root = OrdinalEncoder(
    unknown_value=-1, dtype=int, handle_unknown="use_encoded_value"
)
oe_issue_type = OrdinalEncoder(
    unknown_value=-1, dtype=int, handle_unknown="use_encoded_value"
)


def transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features, impute, and scale.

    Args:
        data (pd.DataFrame): input data frame.
    Returns:
        pd.DataFrame: updated data frame.
    """

    # set up df, overwrite later
    x = pd.DataFrame(data={"TRADE_PRICE": data["TRADE_PRICE"]}, index=data.index)

    # size features
    x["bid_ask_size_ratio_ex"] = data["bid_size_ex"] / data["ask_size_ex"]
    x["rel_bid_size_ex"] = data["TRADE_SIZE"] / data["bid_size_ex"]
    x["rel_ask_size_ex"] = data["TRADE_SIZE"] / data["ask_size_ex"]
    x["depth_ex"] = data["bid_size_ex"] - data["ask_size_ex"]

    # classical
    mid_ex = 0.5 * (data["ask_ex"] + data["bid_ex"])
    mid_best = 0.5 * (data["BEST_ASK"] + data["BEST_BID"])

    spread_ex = data["ask_ex"] - data["bid_ex"]
    spread_best = data["BEST_ASK"] - data["BEST_BID"]

    x["prox_ex"] = (data["TRADE_PRICE"] - mid_ex) / (0.5 * spread_ex)
    x["prox_best"] = (data["TRADE_PRICE"] - mid_best) / (0.5 * spread_best)

    # custom features
    x["spread_ex"] = spread_ex
    x["spread_best"] = spread_best
    x["bid_ask_ratio_ex"] = data["bid_ex"] / data["ask_ex"]
    x["price_rel_nbo"] = (data["TRADE_PRICE"] - data["BEST_ASK"]) / (
        data["BEST_ASK"] - mid_best
    )
    x["price_rel_nbb"] = (data["TRADE_PRICE"] - data["BEST_BID"]) / (
        mid_best - data["BEST_BID"]
    )

    # calculate change
    x["chg_ex_lead"] = data["TRADE_PRICE"] - data["price_ex_lead"]
    x["chg_ex_lag"] = data["TRADE_PRICE"] - data["price_ex_lag"]
    x["chg_all_lead"] = data["TRADE_PRICE"] - data["price_all_lead"]
    x["chg_all_lag"] = data["TRADE_PRICE"] - data["price_all_lag"]

    asks = [f"ASK_{i}" for i in range(1, 17)]
    bids = [f"BID_{i}" for i in range(1, 17)]
    
    # log transformed features
    x[
        [
            "ask_ex",
            "bid_ex",
            "BEST_ASK",
            "BEST_BID",
            "TRADE_PRICE",
            "price_all_lag",
            "price_all_lead",
            "price_ex_lag",
            "price_ex_lead",
            "TRADE_SIZE",
            "bid_size_ex",
            "ask_size_ex",
            "day_vol",
            "myn",
            "STRK_PRC",
            *asks,
            *bids
        ]
    ] = np.log1p(
        data[
            [
                "ask_ex",
                "bid_ex",
                "BEST_ASK",
                "BEST_BID",
                "TRADE_PRICE",
                "price_all_lag",
                "price_all_lead",
                "price_ex_lag",
                "price_ex_lead",
                "TRADE_SIZE",
                "bid_size_ex",
                "ask_size_ex",
                "day_vol",
                "myn",
                "STRK_PRC",
                *asks,
                *bids
            ]
        ]
    )
    x["mid_ex"] = np.log1p(mid_ex)
    x["mid_best"] = np.log1p(mid_best)

    x["ttm"] = (
        data["EXPIRATION"].dt.to_period("M") - data["QUOTE_DATETIME"].dt.to_period("M")
    ).apply(lambda x: x.n)

    # save num columns for scaler
    num_cols = x.columns.tolist()

    # date features
    x["date_year"] = data["QUOTE_DATETIME"].dt.year

    months_in_year = 12
    x["date_month_sin"] = sin_encode(data["QUOTE_DATETIME"].dt.month, months_in_year)
    x["date_month_cos"] = cos_encode(data["QUOTE_DATETIME"].dt.month, months_in_year)

    days_in_month = 31  # at max :-)
    x["date_day_sin"] = sin_encode(data["QUOTE_DATETIME"].dt.day, days_in_month)
    x["date_day_cos"] = cos_encode(data["QUOTE_DATETIME"].dt.day, days_in_month)

    days_in_week = 7
    x["date_weekday_sin"] = sin_encode(
        data["QUOTE_DATETIME"].dt.dayofweek, days_in_week
    )
    x["date_weekday_cos"] = cos_encode(
        data["QUOTE_DATETIME"].dt.dayofweek, days_in_week
    )

    seconds_in_day = 24 * 60 * 60
    seconds = (
        data["QUOTE_DATETIME"] - data["QUOTE_DATETIME"].dt.normalize()
    ).dt.total_seconds()

    x["date_time_sin"] = sin_encode(seconds, seconds_in_day)
    x["date_time_cos"] = cos_encode(seconds, seconds_in_day)

    # impute with zeros
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.fillna(0, inplace=True)

    # standardize continous columns (w/o date features)
    # bin encode categorical features
    try:
        x[num_cols] = scaler.transform(x[num_cols])
        x["bin_option_type"] = oe_option_type.transform(
            data["OPTION_TYPE"].astype(str).values.reshape(-1, 1)
        )
        x["bin_issue_type"] = oe_issue_type.transform(
            data["issue_type"].astype(str).values.reshape(-1, 1)
        )
        x["bin_root"] = oe_root.transform(
            data["ROOT"].astype(str).values.reshape(-1, 1)
        )
        print("transform (val + test)")
    except NotFittedError as e:
        x[num_cols] = scaler.fit_transform(x[num_cols])
        x["bin_option_type"] = oe_option_type.fit_transform(
            data["OPTION_TYPE"].astype(str).values.reshape(-1, 1)
        )
        x["bin_issue_type"] = oe_issue_type.fit_transform(
            data["issue_type"].astype(str).values.reshape(-1, 1)
        )
        x["bin_root"] = oe_root.fit_transform(
            data["ROOT"].astype(str).values.reshape(-1, 1)
        )
        print("fit_transform (train)")

    x["buy_sell"] = data["buy_sell"]
    return x

os.environ["GCLOUD_PROJECT"] = "flowing-mantis-239216"

name = "ise_log_standardized"

train = pd.read_parquet(
    f"gs://thesis-bucket-option-trade-classification/data/preprocessed/train_set_ultra_60.parquet",
    engine="fastparquet",
)

output_path = (
    f"gs://thesis-bucket-option-trade-classification/data/{name}/train_set_60.parquet"
)
train = transform(train)
train.to_parquet(output_path)
del train
gc.collect()

val = pd.read_parquet(
    f"gs://thesis-bucket-option-trade-classification/data/preprocessed/val_set_ultra_20.parquet",
    engine="fastparquet",
)

output_path = (
    f"gs://thesis-bucket-option-trade-classification/data/{name}/val_set_20.parquet"
)
val = transform(val)
val.to_parquet(output_path)
del val
gc.collect()

test = pd.read_parquet(
    f"gs://thesis-bucket-option-trade-classification/data/preprocessed/test_set_ultra_20.parquet",
    engine="fastparquet",
)

output_path = (
    f"gs://thesis-bucket-option-trade-classification/data/{name}/test_set_20.parquet"
)
test = transform(test)
test.to_parquet(output_path)

name = "ise_log_standardized"
dataset = wandb.Artifact(name=name, type="preprocessed_data")
dataset.add_reference(
    "gs://thesis-bucket-option-trade-classification/data/ise_log_standardized/train_set_60.parquet"
)
dataset.add_reference(
    "gs://thesis-bucket-option-trade-classification/data/ise_log_standardized/val_set_20.parquet"
)
dataset.add_reference(
    "gs://thesis-bucket-option-trade-classification/data/ise_log_standardized/test_set_20.parquet"
)
run.log_artifact(dataset)

run.finish()