import os
from pathlib import Path
import pandas as pd
import torch
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv
from src.coordinated_multi_market.learning_utils import load_input_data

from src.coordinated_multi_market.basic_battery_dam_env import BasicBatteryDAM
from src.coordinated_multi_market.custom_ppo import CustomPPO

from src.archive.new_testing_rolling_intrinsic_qh_intelligent_stacking import (
    simulate_days_stacked_quarterhourly_products,
)
from src.shared.config import (
    BUCKET_SIZE,
    C_RATE,
    LOGGING_PATH_COORDINATED,
    MAX_CYCLES_PER_YEAR,
    MIN_TRADES,
    MODEL_OUTPUT_PATH_COORDINATED,
    RTE,
    SCALER_OUTPUT_PATH_COORDINATED,
    TEST_CSV_NAME,
)
from src.shared.evaluate_rl_model import evaluate_rl_model

from src.coordinated_multi_market.learning_utils import (
    prepare_input_data,
)


if __name__ == "__main__":

    # Specify to be analysed model
    model_number = "0"
    model_checkpoint = "ppo_stacked_checkpoint_250000_steps"

    versioned_log_path = os.path.join(LOGGING_PATH_COORDINATED, model_number)
    versioned_model_path = os.path.join(MODEL_OUTPUT_PATH_COORDINATED, model_number)
    versioned_scaler_path = os.path.join(SCALER_OUTPUT_PATH_COORDINATED, model_number)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s" % device)
    df_spot_train, df_spot_val, df_spot_test = load_input_data(write_test=True)
    
    input_data_test = prepare_input_data(df_spot_test, versioned_scaler_path, fit_scaler=False )

    model = CustomPPO.load(
        path=os.path.join(versioned_model_path, model_checkpoint + ".zip")
    )

    # ---------- TEST-SET: RL-Läufe ----------
    for key, value in input_data_test.items():
        env = BasicBatteryDAM(
            modus="test",
            logging_path=versioned_log_path,
            input_data={key: value},
        )
        env = DummyVecEnv([lambda: env])
        obs = env.reset()

        for _ in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                break
        env.close()

    logger.info(
        "Finished testing for period: df_spot_test (length %s)" % (len(df_spot_test))
    )

    # evaluate_rl_model(os.path.join(versioned_log_path, TEST_CSV_NAME))
    logger.info("Finished creating report of model behaviour")

    # ---------- TEST-SET: Rolling Intrinsic ----------
    ri_qh_output_path_test = os.path.join(   # <<< NEW: separat merken
        versioned_log_path,
        "rolling_intrinsic_intelligently_stacked_on_day_ahead_qh",
        "bs"
        + str(BUCKET_SIZE)
        + "cr"
        + str(C_RATE)
        + "rto"
        + str(RTE)
        + "mc"
        + str(MAX_CYCLES_PER_YEAR)
        + "mt"
        + str(MIN_TRADES),
    )

    # NEUER Aufruf: start_day / end_day anstatt list_dates  # <<< CHANGED
    start_day_test = (
        df_spot_test.index.min().tz_convert("Europe/Berlin").normalize()
    )
    end_day_test = (
        df_spot_test.index.max().tz_convert("Europe/Berlin").normalize()
        + pd.Timedelta(days=1)
    )

    simulate_days_stacked_quarterhourly_products(
        da_bids_path=os.path.join(versioned_log_path, TEST_CSV_NAME),
        output_path=ri_qh_output_path_test,
        start_day=start_day_test,
        end_day=end_day_test,
        discount_rate=0,
        bucket_size=BUCKET_SIZE,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=MAX_CYCLES_PER_YEAR,
        min_trades=MIN_TRADES,
    )

    logger.info(
        "Finished calculating intelligently stacked rolling intrinsic revenues with quarterhourly products (TEST)."
    )

    # profit.csv direkt aus dem oben definierten RI-Ordner lesen  # <<< CHANGED
    test_advanced_drl_ri_qh = pd.read_csv(
        os.path.join(ri_qh_output_path_test, "profit.csv")
    )

    mean_test_profit = (
        test_advanced_drl_ri_qh.sort_values(by="day").reset_index()["profit"].mean()
    )

    print("Test: ", mean_test_profit)

    # ---------- TRAIN-SET: RL-Läufe ----------
    input_data_train = prepare_input_data(df_spot_train, versioned_scaler_path)

    versioned_log_path_train = os.path.join(versioned_log_path, "train")  # <<< CHANGED
    Path(versioned_log_path_train).mkdir(parents=True, exist_ok=True)

    for key, value in input_data_train.items():
        env = BasicBatteryDAM(
            modus="test",
            logging_path=versioned_log_path_train,
            input_data={key: value},
        )
        env = DummyVecEnv([lambda: env])
        obs = env.reset()

        for _ in range(24):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                break
        env.close()

    logger.info(
        "Finished testing for period: df_spot_train (length %s)"  # <<< CHANGED Text
        % (len(df_spot_train))
    )

    # evaluate_rl_model(os.path.join(versioned_log_path_train, TEST_CSV_NAME))
    logger.info("Finished creating report of model behaviour (TRAIN)")

    # ---------- TRAIN-SET: Rolling Intrinsic ----------
    ri_qh_output_path_train = os.path.join(   # <<< NEW: eigener Pfad
        versioned_log_path_train,
        "rolling_intrinsic_intelligently_stacked_on_day_ahead_qh",
        "bs"
        + str(BUCKET_SIZE)
        + "cr"
        + str(C_RATE)
        + "rto"
        + str(RTE)
        + "mc"
        + str(MAX_CYCLES_PER_YEAR)
        + "mt"
        + str(MIN_TRADES),
    )

    start_day_train = (
        df_spot_train.index.min().tz_convert("Europe/Berlin").normalize()
    )
    end_day_train = (
        df_spot_train.index.max().tz_convert("Europe/Berlin").normalize()
        + pd.Timedelta(days=1)
    )

    simulate_days_stacked_quarterhourly_products(
        da_bids_path=os.path.join(versioned_log_path_train, TEST_CSV_NAME),
        output_path=ri_qh_output_path_train,
        start_day=start_day_train,
        end_day=end_day_train,
        discount_rate=0,
        bucket_size=BUCKET_SIZE,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=MAX_CYCLES_PER_YEAR,
        min_trades=MIN_TRADES,
    )

    logger.info(
        "Finished calculating intelligently stacked rolling intrinsic revenues with quarterhourly products (TRAIN)."
    )

    train_advanced_drl_ri_qh = pd.read_csv(
        os.path.join(ri_qh_output_path_train, "profit.csv")  # <<< CHANGED
    )

    mean_train_profit = (
        train_advanced_drl_ri_qh.sort_values(by="day").reset_index()["profit"].mean()
    )

    print("Average Profit of Test or Train set")
    print("Test: ", mean_test_profit)
    print("Train: ", mean_train_profit)

    # wirte error if the train profit is more than 10% higher than the test profit
    if mean_train_profit > 1.1 * mean_test_profit:
        raise ValueError(
            f"Train profit {mean_train_profit} is more than 10% higher than test profit {mean_test_profit}, this points towards overfitting - Check!"
        )
