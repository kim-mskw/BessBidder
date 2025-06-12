import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

from src.shared.config import (
    START,
    END,
    DATA_PATH,
)


def load_input_data(write_test=False) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_start = START.tz_convert("Europe/Berlin").date().isoformat()
    train_end = END.tz_convert("Europe/Berlin").date().isoformat()

    # path=f"df_spot_train_{train_start}_{train_end}_with_features_utc.csv"
    # TODOD: USE dynamic as soon as epex 15 problem is solved
    path = "df_spot_train_2019-01-01_2021-12-31_with_features_utc.csv"

    df_spot_train = pd.read_csv(
        os.path.join(
            DATA_PATH,
            path,
        ),
        index_col=0,
        parse_dates=True,
    )

    TRAIN_START = START
    TRAIN_END = END

    df_spot_train = df_spot_train[
        (df_spot_train.index > TRAIN_START) & (df_spot_train.index < TRAIN_END)
    ]

    # Step 1: Identify unique days (year-week pairs)
    df_spot_train["day"] = df_spot_train.index.to_period("d")  # Extract day information

    # Step 2: Randomly select a few days for the test set
    rng = np.random.default_rng(seed=42)  # Create a random generator with a seed
    test_days = rng.choice(df_spot_train["day"].unique(), size=5, replace=False)

    # Step 3: Split into train and test sets
    df_spot_test = df_spot_train[df_spot_train["day"].isin(test_days)]
    df_spot_train = df_spot_train[~df_spot_train["day"].isin(test_days)]

    # Drop the 'week' column (optional)
    df_spot_train = df_spot_train.drop(columns="day")
    df_spot_test = df_spot_test.drop(columns="day")

    if write_test == True:
        # store new df_spot_test
        df_spot_test.to_csv(
            os.path.join(
                "data",
                "simplified_data_jan_with_exaa_and_id_full",
                "df_spot_test_2023-01-01_2023-12-31_with_features_utc.csv",
            ),
            date_format="%Y-%m-%dT%H:%M:%S%z",
        )

    # df_spot_train.index = df_spot_train.index.tz_convert("Europe/Berlin")
    # df_spot_test.index = df_spot_test.index.tz_convert("Europe/Berlin")

    return df_spot_train, df_spot_test


def prepare_input_data(
    df: pd.DataFrame, versioned_scaler_path: str
) -> dict[str, dict[str, np.array]]:
    scalable_features = df[
        [
            "load_forecast_d_minus_1_1000_total_de_lu_mw",
            "pv_forecast_d_minus_1_1000_de_lu_mw",
            "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
            "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
            "date_month",
            "day_of_week",
            "wind_forecast_daily_mean",
            "wind_forecast_daily_std",
            "spread_id_full_da_qh_mean",
            "spread_id_full_da_qh_std",
            "spread_id_full_da_qh_min",
            "spread_id_full_da_qh_max",
        ]
    ].copy()
    scaler = MinMaxScaler()
    scaler.fit(scalable_features)
    features_scaled = scaler.transform(scalable_features)
    df_scaled = pd.DataFrame(
        features_scaled, columns=scalable_features.columns, index=df.index
    )
    joblib.dump(scaler, os.path.join(versioned_scaler_path, "scaler.pkl"))
    df = pd.concat(
        [
            df_scaled,
            df[
                [
                    "epex_spot_60min_de_lu_eur_per_mwh",
                    "epex_spot_15min_de_lu_eur_per_mwh",
                ]
            ],
        ],
        axis=1,
    )

    input_dict = {}
    days = np.unique(df.index.date)
    for day in days:
        if df.loc[day.isoformat()][
            [
                "epex_spot_60min_de_lu_eur_per_mwh",
                "epex_spot_15min_de_lu_eur_per_mwh",
                "load_forecast_d_minus_1_1000_total_de_lu_mw",
                "pv_forecast_d_minus_1_1000_de_lu_mw",
                "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
                "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
                "date_month",
                "day_of_week",
                "wind_forecast_daily_mean",
                "wind_forecast_daily_std",
                "spread_id_full_da_qh_mean",
                "spread_id_full_da_qh_std",
                "spread_id_full_da_qh_min",
                "spread_id_full_da_qh_max",
            ]
        ].isna().any().any() or df.loc[day.isoformat()][
            [
                "epex_spot_60min_de_lu_eur_per_mwh",
                "epex_spot_15min_de_lu_eur_per_mwh",
                "load_forecast_d_minus_1_1000_total_de_lu_mw",
                "pv_forecast_d_minus_1_1000_de_lu_mw",
                "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
                "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
                "date_month",
                "day_of_week",
                "wind_forecast_daily_mean",
                "wind_forecast_daily_std",
                "spread_id_full_da_qh_mean",
                "spread_id_full_da_qh_std",
                "spread_id_full_da_qh_min",
                "spread_id_full_da_qh_max",
            ]
        ].shape != (
            24,
            14,
        ):
            continue

        input_dict.update(
            {
                day.isoformat(): {
                    "price_forecast": np.array(
                        df.loc[day.isoformat()]["epex_spot_15min_de_lu_eur_per_mwh"]
                        .astype(np.float32)
                        .values
                    ),
                    "price_realized": np.array(
                        df.loc[day.isoformat()]["epex_spot_60min_de_lu_eur_per_mwh"]
                        .astype(np.float32)
                        .values
                    ),
                    "pv_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        df.loc[day.isoformat()]["pv_forecast_d_minus_1_1000_de_lu_mw"]
                        .astype(np.float32)
                        .values
                    ),
                    "wind_onshore_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        df.loc[day.isoformat()][
                            "wind_onshore_forecast_d_minus_1_1000_de_lu_mw"
                        ]
                        .astype(np.float32)
                        .values
                    ),
                    "wind_offshore_forecast_d_minus_1_1000_de_lu_mw": np.array(
                        df.loc[day.isoformat()][
                            "wind_offshore_forecast_d_minus_1_1000_de_lu_mw"
                        ]
                        .astype(np.float32)
                        .values
                    ),
                    "load_forecast_d_minus_1_1000_total_de_lu_mw": np.array(
                        df.loc[day.isoformat()][
                            "load_forecast_d_minus_1_1000_total_de_lu_mw"
                        ]
                        .astype(np.float32)
                        .values
                    ),
                    "date_month": np.array(
                        df.loc[day.isoformat()]["date_month"].astype(np.float32).values
                    ),
                    "day_of_week": np.array(
                        df.loc[day.isoformat()]["day_of_week"].astype(np.float32).values
                    ),
                    "wind_forecast_daily_mean": np.array(
                        df.loc[day.isoformat()]["wind_forecast_daily_mean"]
                        .astype(np.float32)
                        .values
                    ),
                    "wind_forecast_daily_std": np.array(
                        df.loc[day.isoformat()]["wind_forecast_daily_std"]
                        .astype(np.float32)
                        .values
                    ),
                    "spread_id_full_da_mean": np.array(
                        df.loc[day.isoformat()]["spread_id_full_da_qh_mean"]
                        .astype(np.float32)
                        .values
                    ),
                    "spread_id_full_da_std": np.array(
                        df.loc[day.isoformat()]["spread_id_full_da_qh_std"]
                        .astype(np.float32)
                        .values
                    ),
                    "spread_id_full_da_min": np.array(
                        df.loc[day.isoformat()]["spread_id_full_da_qh_min"]
                        .astype(np.float32)
                        .values
                    ),
                    "spread_id_full_da_max": np.array(
                        df.loc[day.isoformat()]["spread_id_full_da_qh_max"]
                        .astype(np.float32)
                        .values
                    ),
                    "timestamps": np.array(df.loc[day.isoformat()].index.values),
                }
            }
        )
    return input_dict


# Define a linear learning rate schedule
def linear_schedule(initial_value):
    """
    Returns a function that computes the learning rate linearly decaying
    from `initial_value` to 0 based on progress remaining.
    """

    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


def orthogonal_weight_init(module):
    """
    Custom weight initialization using orthogonal initialization.
    Applies orthogonal initialization to linear layers and zeros to biases.
    """
    if isinstance(module, nn.Linear):  # Apply only to Linear layers
        nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(module.bias)  # Initialize biases to 0
