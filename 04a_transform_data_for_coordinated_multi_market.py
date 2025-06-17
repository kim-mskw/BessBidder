import os.path
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.shared.config import START, END, DATA_PATH

TRAIN_START = START
TRAIN_END = END

ZEITUMSTELLUNGEN = [
    "2019-03-31T00:00:00+01:00",
    "2019-10-27T00:00:00+02:00",
    "2020-03-29T00:00:00+01:00",
    "2020-10-25T00:00:00+02:00",
    "2021-03-28T00:00:00+01:00",
    "2021-10-31T00:00:00+02:00",
    "2022-03-27T00:00:00+01:00",
    "2022-10-30T00:00:00+02:00",
    "2023-03-26T00:00:00+01:00",
    "2023-10-29T00:00:00+02:00",
]


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date_month"] = df.index.month
    # df["date_month_sin"] = np.sin(2 * np.pi * df["date_month"] / 12)
    # df["date_month_cos"] = np.cos(2 * np.pi * df["date_month"] / 12)

    ger_holidays = holidays.country_holidays("DE", years=[2019, 2020, 2021, 2022, 2023])
    ger_holidays = list(ger_holidays.keys())

    holiday_dates = np.isin(df.index.date, ger_holidays)
    df["day_of_week"] = df.index.day_of_week
    df.loc[holiday_dates, "day_of_week"] = 6

    # df.drop(columns=["date_month"], inplace=True)
    return df


def derive_daily_wind_forecast_stats(df: pd.DataFrame) -> pd.DataFrame:
    wind_forecast_sum = df[
        [
            "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
            "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
        ]
    ].sum(axis=1)

    daily_stats = wind_forecast_sum.resample("D").agg(["mean", "std"])

    df["wind_forecast_daily_mean"] = daily_stats["mean"].reindex(
        df.index, method="ffill"
    )
    df["wind_forecast_daily_std"] = daily_stats["std"].reindex(df.index, method="ffill")

    return df


def calculate_id_da_spread_and_stats(df: pd.DataFrame) -> pd.DataFrame:
    spread_id_full_da_h = df["id_full_h"] - df["epex_spot_60min_de_lu_eur_per_mwh"]
    daily_stats_h = (
        spread_id_full_da_h.resample("D").agg(["mean", "std", "min", "max"]).shift(1)
    )
    spread_id_full_da_qh = df["id_full_qh"] - df["epex_spot_60min_de_lu_eur_per_mwh"]
    daily_stats_qh = (
        spread_id_full_da_qh.resample("D").agg(["mean", "std", "min", "max"]).shift(1)
    )
    df["spread_id_full_da_h_mean"] = daily_stats_h["mean"].reindex(
        df.index, method="ffill"
    )
    df["spread_id_full_da_h_std"] = daily_stats_h["std"].reindex(
        df.index, method="ffill"
    )
    df["spread_id_full_da_h_min"] = daily_stats_h["min"].reindex(
        df.index, method="ffill"
    )
    df["spread_id_full_da_h_max"] = daily_stats_h["max"].reindex(
        df.index, method="ffill"
    )
    df["spread_id_full_da_qh_mean"] = daily_stats_qh["mean"].reindex(
        df.index, method="ffill"
    )
    df["spread_id_full_da_qh_std"] = daily_stats_qh["std"].reindex(
        df.index, method="ffill"
    )
    df["spread_id_full_da_qh_min"] = daily_stats_qh["min"].reindex(
        df.index, method="ffill"
    )
    df["spread_id_full_da_qh_max"] = daily_stats_qh["max"].reindex(
        df.index, method="ffill"
    )
    return df


def create_dataframes():
    df = pd.read_csv(
        Path("data", "data_2019-01-01_2024-01-01_hourly.csv"),
        index_col=0,
        parse_dates=True,
    ).ffill()
    df.index = df.index.tz_convert("Europe/Berlin")

    missing_days = np.unique([x.date() for x in df[df.isna().any(axis=1)].index])

    df = df[~df.index.to_series().apply(lambda x: x.date()).isin(missing_days)]

    df = df.loc["2019":"2023"].copy()

    df = make_time_features(df)
    df = derive_daily_wind_forecast_stats(df)
    df = calculate_id_da_spread_and_stats(df)

    df_spot_train = df.loc[TRAIN_START:TRAIN_END][
        [
            "epex_spot_60min_de_lu_eur_per_mwh",
            "exaa_spot_15min_de_lu_eur_per_mwh",
            "load_forecast_d_minus_1_1000_total_de_lu_mw",
            "pv_forecast_d_minus_1_1000_de_lu_mw",
            "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
            "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
            "date_month",
            "day_of_week",
            "wind_forecast_daily_mean",
            "wind_forecast_daily_std",
            "spread_id_full_da_h_mean",
            "spread_id_full_da_h_std",
            "spread_id_full_da_h_min",
            "spread_id_full_da_h_max",
            "spread_id_full_da_qh_mean",
            "spread_id_full_da_qh_std",
            "spread_id_full_da_qh_min",
            "spread_id_full_da_qh_max",
        ]
    ]

    df_spot_train.index = df_spot_train.index.tz_convert("utc")

    output_path = DATA_PATH
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_start = (
        df_spot_train.index.min().tz_convert("Europe/Berlin").date().isoformat()
    )
    train_end = df_spot_train.index.max().tz_convert("Europe/Berlin").date().isoformat()

    df_spot_train.to_csv(
        Path(
            output_path,
            f"df_spot_train_{train_start}_{train_end}_with_features_utc.csv",
        )
    )


if __name__ == "__main__":
    create_dataframes()
