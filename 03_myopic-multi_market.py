"""
Myopic Multi-Market Simulation Script

This script:
- Loads results from the day-ahead MILP optimisation (single market).
- Simulates myopic rolling intrinsic bidding on the intraday market.
- Uses intelligent stacking of quarter-hourly products.
- Writes simulation logs and outputs to a versioned logging path.

Requires:
- Day-ahead results from MILP (CSV)
- Configuration from `src/shared/config.py`
- Simulation functions for quarter-hourly stacking
"""

import os
import pandas as pd
from loguru import logger

from src.coordinated_multi_market.rolling_intrinsic.testing_rolling_intrinsic_qh_intelligent_stacking import (
    simulate_days_stacked_quarterhourly_products,
)


from src.shared.config import (
    BUCKET_SIZE,
    C_RATE,
    INPUT_DIR_DA,
    LOGGING_PATH_MYOPIC,
    MAX_CYCLES_PER_YEAR,
    MIN_TRADES,
    RTE, START, END
)

if __name__ == "__main__":
    # Path where output logs and results will be stored
    versioned_log_path = LOGGING_PATH_MYOPIC

    # Load day-ahead optimisation results
    df_milp = pd.read_csv(INPUT_DIR_DA)
    df_milp["time"] = pd.to_datetime(df_milp["time"]).dt.tz_convert("Europe/Berlin")

    # Construct RI output path dynamically based on configuration
    ri_qh_output_path = os.path.join(
        versioned_log_path,
        "rolling_intrinsic_stacked_on_day_ahead_qh",
        f"bs{BUCKET_SIZE}cr{C_RATE}rto{RTE}mc{MAX_CYCLES_PER_YEAR}mt{MIN_TRADES}",
    )

    # Run the rolling intrinsic simulation with intelligent stacking (QH products)
    simulate_days_stacked_quarterhourly_products(
        da_bids_path=INPUT_DIR_DA,
        output_path=ri_qh_output_path,
        start_day=START,
        end_day=END,
        discount_rate=0,
        bucket_size=BUCKET_SIZE,
        c_rate=C_RATE,
        roundtrip_eff=RTE,
        max_cycles=MAX_CYCLES_PER_YEAR,
        min_trades=MIN_TRADES,
    )

    logger.info(
        "Finished calculating intelligently stacked rolling intrinsic revenues with quarterhourly products."
    )
