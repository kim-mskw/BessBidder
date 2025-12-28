import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv

from src.coordinated_multi_market.learning_utils import load_input_data, prepare_input_data
from src.coordinated_multi_market.basic_battery_dam_env import BasicBatteryDAM
from src.coordinated_multi_market.custom_ppo import CustomPPO

# from src.coordinated_multi_market.rolling_intrinsic.new_testing_rolling_intrinsic_qh_intelligent_stacking import (
#     simulate_days_stacked_quarterhourly_products,)

from src.coordinated_multi_market.rolling_intrinsic.testing_rolling_intrinsic_qh_intelligent_stacking import (
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
    VAL_START,
    VAL_END,
)


def _compute_series_stats(series: pd.Series):
    """
    Hilfsfunktion: berechne total, mean, median, std, q95.
    (Sharpe lassen wir hier weg, wird nicht mehr benötigt.)
    """
    series = series.dropna()
    if len(series) == 0:
        return (None, None, None, None, None)

    total = series.sum()
    mean = series.mean()
    median = series.median()
    std = series.std(ddof=0)
    q95 = series.quantile(0.95) if len(series) > 1 else series.iloc[0]

    return total, mean, median, std, q95


if __name__ == "__main__":

    # -------- Parameter für die Validierung --------
    model_number = "1"

    # Start-Checkpoint, ab dem validiert werden soll
    # -> muss dem Namensschema aus dem Training entsprechen
    start_checkpoint = "ppo_stacked_checkpoint_180000_steps"

    # Schrittweite der Checkpoints in "Steps"
    STEP_INCREMENT = 10000

    # ------------------------------------------------

    versioned_log_base_path = os.path.join(LOGGING_PATH_COORDINATED, model_number)
    versioned_model_path = os.path.join(MODEL_OUTPUT_PATH_COORDINATED, model_number)
    versioned_scaler_path = os.path.join(SCALER_OUTPUT_PATH_COORDINATED, model_number)

    # Validation-Logs kommen in einen eigenen Unterordner
    validation_root_path = os.path.join(versioned_log_base_path, "validation")
    Path(validation_root_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s" % device)

    # ---- Daten laden & Validierungs-Set extrahieren ----
    df_spot_train, df_spot_val, df_spot_test = load_input_data(write_test=False)

    logger.info(
        "Validation period: %s -> %s (len=%s)"
        % (df_spot_val.index.min(), df_spot_val.index.max(), len(df_spot_val))
    )

    # ---- Checkpoint-Namensschema zerlegen ----
    parts = start_checkpoint.split("_")
    try:
        start_steps = int(parts[-2])
    except (ValueError, IndexError):
        raise ValueError(
            f"Kann Steps nicht aus Checkpoint-Namen '{start_checkpoint}' parsen. "
            "Erwartetes Schema: <prefix>_<steps>_steps"
        )
    checkpoint_prefix = "_".join(parts[:-2])  # z.B. "ppo_stacked_checkpoint"

    logger.info(
        f"Starte Validierung bei Checkpoint {start_checkpoint} "
        f"(prefix='{checkpoint_prefix}', start_steps={start_steps}, increment={STEP_INCREMENT})"
    )

    # Summary-Datei für alle Validierungs-Ergebnisse
    summary_path = os.path.join(validation_root_path, "validation_summary.csv")

    if os.path.exists(summary_path):
        logger.info(
            f"Existierende validation_summary.csv wird mit neuem Schema überschrieben: {summary_path}"
        )

    validation_summary = pd.DataFrame(
        columns=[
            "checkpoint_name",
            "steps",
            # RL metrics
            "mean_rl_reward",
            "total_rl_reward",
            # RI total metrics (daily)
            "num_days_ri",
            "total_profit",
            "mean_profit",
            "median_profit",
            "std_profit",
            "q95_profit",
            "idc_dam_profit_ratio",
            # DA metrics
            "total_dam_profit",
            "mean_dam_profit",
            # IDC metrics
            "total_idc_profit",
            "mean_idc_profit",
        ]
    )

    # ---- Über Checkpoints iterieren ----
    current_steps = start_steps

    while True:
        checkpoint_name = f"{checkpoint_prefix}_{current_steps}_steps"
        checkpoint_file = os.path.join(versioned_model_path, checkpoint_name + ".zip")

        if not os.path.exists(checkpoint_file):
            logger.info(f"Checkpoint {checkpoint_file} existiert nicht. Breche Schleife ab.")
            break

        logger.info(f"Validiere Checkpoint: {checkpoint_name}")

        # Pfad für diesen Checkpoint-Lauf
        ckpt_log_path = os.path.join(validation_root_path, checkpoint_name)
        Path(ckpt_log_path).mkdir(parents=True, exist_ok=True)

        # ---- RL-Teil auf Validierungsdaten ----
        model = CustomPPO.load(path=checkpoint_file)

        input_data_val = prepare_input_data(
            df_spot_val, versioned_scaler_path, fit_scaler=False
        )

        for key, value in input_data_val.items():
            env = BasicBatteryDAM(
                modus="test",
                logging_path=ckpt_log_path,
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
            "Finished RL evaluation on validation set for checkpoint %s (len(df_spot_val)=%s)"
            % (checkpoint_name, len(df_spot_val))
        )

        # ---- RL-Metriken aus behaviour_log.csv auslesen (falls vorhanden) ----
        behaviour_path = os.path.join(ckpt_log_path, TEST_CSV_NAME)
        mean_rl_reward = None
        total_rl_reward = None

        if os.path.exists(behaviour_path):
            df_behaviour = pd.read_csv(behaviour_path)
            if "reward" in df_behaviour.columns:
                total_rl_reward = df_behaviour["reward"].sum()
                mean_rl_reward = df_behaviour["reward"].mean()
            else:
                logger.warning(
                    f"In {behaviour_path} keine 'reward'-Spalte gefunden. RL-Kennzahlen werden als NaN gesetzt."
                )

        # ---- RI-Teil auf Validierungsdaten ----
        ri_qh_output_path_val = os.path.join(
            ckpt_log_path,
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

        # Output-Verzeichnis für RI leeren, um Resume-Effekte zu vermeiden
        if os.path.exists(ri_qh_output_path_val):
            shutil.rmtree(ri_qh_output_path_val)

        simulate_days_stacked_quarterhourly_products(
            da_bids_path=behaviour_path,
            output_path=ri_qh_output_path_val,
            start_day=VAL_START,
            end_day=VAL_END,
            discount_rate=0,
            bucket_size=BUCKET_SIZE,
            c_rate=C_RATE,
            roundtrip_eff=RTE,
            max_cycles=MAX_CYCLES_PER_YEAR,
            min_trades=MIN_TRADES,
        )

        logger.info(
            "Finished RI calculation on validation set for checkpoint %s."
            % checkpoint_name
        )

        # ---- RI-Profite und DA/IDC-Split auslesen ----
        profit_csv_path = os.path.join(ri_qh_output_path_val, "profit.csv")
        trades_path = os.path.join(ri_qh_output_path_val, "trades")

        # Defaults
        num_days_ri = 0
        total_profit = mean_profit = median_profit = std_profit = q95_profit = None
        total_dam_profit = mean_dam_profit = None
        total_idc_profit = mean_idc_profit = None
        idc_dam_profit_ratio = None

        if os.path.exists(profit_csv_path):
            df_profit = pd.read_csv(profit_csv_path)

            df_profit["day"] = pd.to_datetime(df_profit["day"], errors="coerce", utc=True,)
            df_profit["day"] = df_profit["day"].dt.tz_convert("Europe/Berlin")

            df_profit = df_profit.sort_values("day").reset_index(drop=True)

            df_profit["delivery_day"] = df_profit["day"].dt.date
            daily_total = df_profit.set_index("delivery_day")["profit"]

            (
                total_profit,
                mean_profit,
                median_profit,
                std_profit,
                q95_profit,
            ) = _compute_series_stats(daily_total)

            # Alle Trades einlesen für DA/IDC-Split
            if os.path.exists(trades_path):
                trade_files = [
                    f
                    for f in os.listdir(trades_path)
                    if f.startswith("trades_") and f.endswith(".csv")
                ]
                all_trades = []
                for f in trade_files:
                    fp = os.path.join(trades_path, f)
                    df_t = pd.read_csv(fp, parse_dates=["execution_time"])
                    all_trades.append(df_t)

                if len(all_trades) > 0:
                    trades_df = pd.concat(all_trades, ignore_index=True)
                    """
                    # Zeitzone für execution_time setzen, falls nötig
                    if trades_df["execution_time"].dt.tz is None:
                        trades_df["execution_time"] = trades_df["execution_time"].dt.tz_localize(
                            "Europe/Berlin"
                        )
                    else:
                        trades_df["execution_time"] = trades_df["execution_time"].dt.tz_convert(
                            "Europe/Berlin"
                        )
                    """
                    trades_df["execution_time"] = pd.to_datetime(trades_df["execution_time"], errors="coerce", utc=True)
                    trades_df["execution_time"] = trades_df["execution_time"].dt.tz_convert("Europe/Berlin")
                    

                    # Lieferzeitpunkt (product) in Datetime → Liefertag
                    trades_df["delivery_dt"] = pd.to_datetime(
                        trades_df["product"], errors="coerce", utc=True)
                    trades_df["delivery_dt"] = trades_df["delivery_dt"].dt.tz_convert("Europe/Berlin")
                
                    trades_df["delivery_day"] = trades_df["delivery_dt"].dt.date

                    # DA = Trades, die um 13:00 ausgeführt werden (execution_time)
                    is_dam = trades_df["execution_time"].dt.hour == 13

                    # DA-Profit pro Liefertag
                    daily_dam = (
                        trades_df[is_dam]
                        .groupby("delivery_day")["profit"]
                        .sum()
                    )

                    # DA-Serie auf alle Liefertage aus profit.csv ausrichten
                    daily_dam_aligned = daily_dam.reindex(
                        daily_total.index, fill_value=0.0
                    )

                    # IDC-Profit = Total - DA (pro Liefertag)
                    daily_idc = daily_total - daily_dam_aligned

                    # DA-Stats (wir nutzen nur total & mean)
                    (
                        total_dam_profit,
                        mean_dam_profit,
                        _,
                        _,
                        _,
                    ) = _compute_series_stats(daily_dam_aligned)

                    # IDC-Stats
                    (
                        total_idc_profit,
                        mean_idc_profit,
                        _,
                        _,
                        _,
                    ) = _compute_series_stats(daily_idc)

                    # Verhältnis DA / IDC
                    if total_idc_profit not in (None, 0):
                        idc_dam_profit_ratio = total_dam_profit / total_idc_profit
                    else:
                        idc_dam_profit_ratio = None

                else:
                    logger.warning(f"Keine Trade-Dateien im Pfad {trades_path} gefunden.")
            else:
                logger.warning(f"Trades-Pfad {trades_path} existiert nicht.")
        else:
            logger.warning(
                f"Keine profit.csv im RI-Output-Pfad {ri_qh_output_path_val} gefunden."
            )

        # ---- Ergebnisse in Summary anhängen ----
        validation_summary = pd.concat(
            [
                validation_summary,
                pd.DataFrame(
                    [
                        [
                            checkpoint_name,
                            current_steps,
                            # RL
                            mean_rl_reward,
                            total_rl_reward,
                            # RI total
                            num_days_ri,
                            total_profit,
                            mean_profit,
                            median_profit,
                            std_profit,
                            q95_profit,
                            idc_dam_profit_ratio,
                            # DA
                            total_dam_profit,
                            mean_dam_profit,
                            # IDC
                            total_idc_profit,
                            mean_idc_profit,
                        ]
                    ],
                    columns=validation_summary.columns,
                ),
            ],
            ignore_index=True,
        )

        # Summary speichern nach jedem Checkpoint
        validation_summary.to_csv(summary_path, index=False)
        logger.info(
            f"Validation metrics for {checkpoint_name} gespeichert in {summary_path}"
        )

        # Nächster Checkpoint
        current_steps += STEP_INCREMENT

    logger.info("Validation über alle Checkpoints abgeschlossen.")
    print(validation_summary)
