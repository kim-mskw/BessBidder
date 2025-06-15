import os
import pandas as pd
from pathlib import Path

# GENERAL TECHNICAL CONFIGURATION

# Config file for parameters of the case study
C_RATE = 1
RTE = 0.86
MAX_CYCLES_PER_YEAR = 365
MAX_CYCLES_PER_DAY = 1


# Rolling Intrinsic specific
BUCKET_SIZE = 15
MIN_TRADES = 10

# Define model horizon
START = pd.Timestamp(year=2019, month=1, day=1, tz="Europe/Berlin")
END = pd.Timestamp(year=2024, month=1, day=1, tz="Europe/Berlin")

# ----------------------------------------------
# MODELLING CONFIGURATIONS

# 02 SINGLE MARKET CONFIGURATION

OUTPUT_DIR_DA = os.path.join("output", "myopic_multi_market", "day_ahead_milp")
FILENAME_DA = "11-12.2020_ACM.csv"
DATA_PATH_DA = Path("data", "data_2019-01-01_2024-01-01_hourly.csv")


# 03 MYOPIC MULTI-MARKET CONFIGURATION

# Should naturally be the same as OUTPUT_DIR_DA, but is separated in case different results should be used
INPUT_DIR_DA = os.path.join(OUTPUT_DIR_DA, FILENAME_DA)
LOGGING_PATH_MYOPIC = Path("output/myopic_multi_market/")

# 04 COORDINATED MULTI-MARKET CONFIGURATION
SEED = 42
TRAINING_STEPS_INTELLIGENT = 300000
TRAINING_STEPS_BASIC = 300000

DATA_PATH = Path("data", "simplified_data_jan_with_exaa_and_id_full")

COORDINATED_MODEL_NAME_QH = "model_intelligent_quarterhourly_products"
TRAIN_CSV_NAME = "basic_battery_dam_train_log_v3.csv"
TEST_CSV_NAME = "basic_battery_dam_test_log_v3.csv"

COORDINATED_STACKED_RI_QH_TRAINING_OUTPUT_CSV = (
    "output_ri_qh_intelligent_stacking_training.csv"
)
COORDINATED_STACKED_RI_H_TRAINING_OUTPUT_CSV = (
    "output_ri_h_intelligent_stacking_training.csv"
)


LOGGING_PATH_COORDINATED = os.path.join("output", "coordinated_multi_market", "logging")
TENSORBOARD_PATH_INTELLIGENT = os.path.join(
    "output", "coordinated_multi_market", "tensorboard"
)
MODEL_OUTPUT_PATH_COORDINATED = os.path.join(
    "output", "coordinated_multi_market", "models"
)
SCALER_OUTPUT_PATH_COORDINATED = os.path.join(
    "output", "coordinated_multi_market", "scalers"
)
