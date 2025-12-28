# precompute_vwaps.py

import os
import warnings
from typing import Set, Union
from pathlib import Path

import pandas as pd
import psycopg2
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)



# --- Database configuration -------------------------------------------------

PASSWORD = os.getenv("SQL_PASSWORD")
if PASSWORD:
    password_for_url = f":{PASSWORD}"
else:
    password_for_url = ""

THESIS_DB_NAME = os.getenv("POSTGRES_DB_NAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")

if not all([THESIS_DB_NAME, POSTGRES_USER, POSTGRES_DB_HOST]):
    raise RuntimeError(
        "Missing required database environment variables "
        "(POSTGRES_DB_NAME, POSTGRES_USER, POSTGRES_DB_HOST)."
    )

CONNECTION = (
    f"postgres://{POSTGRES_USER}{password_for_url}@{POSTGRES_DB_HOST}/{THESIS_DB_NAME}"
)


PathLike = Union[str, Path]


def precompute_vwaps_for_day(
    current_day: pd.Timestamp,
    bucket_size: int,
    min_trades: int,
    output_path: PathLike,
) -> None:
    """
    Precompute intraday VWAPs for a single delivery day and store them as a
    bucket_end x deliverystart matrix in a Parquet file.

    Steps:
    1. Fetch all relevant trades from the database.
    2. Assign each trade to an execution-time bucket (bucket_end).
    3. Compute VWAP for each (bucket_end, deliverystart) pair.
    4. Write a dense matrix (bucket_end rows, quarter-hour products columns) to Parquet.

    Parameters
    ----------
    current_day : pd.Timestamp
        Delivery day in Europe/Berlin timezone.
    bucket_size : int
        Bucket size in minutes (e.g. 15).
    min_trades : int
        Minimum number of trades per (bucket_end, deliverystart) to compute a VWAP.
    output_path : str | Path
        Directory where Parquet files (vwaps_YYYY-MM-DD.parquet) will be stored.
    """

    current_day = (
        pd.Timestamp(current_day)
        .tz_convert("Europe/Berlin")
        .normalize()
    )
    output_path = Path(output_path)

    trading_start = current_day - pd.Timedelta(hours=8)
    trading_end = current_day + pd.Timedelta(days=1)

    logger.info(f"Precomputing VWAPs for {current_day.date()}")

    # 1. Fetch trades from DB
    # -------------------------------------------------------------------------
    query = f"""
        SELECT 
            executiontime,
            deliverystart,
            weighted_avg_price,
            volume,
            trade_count
        FROM transactions_intraday_de
        WHERE executiontime >= '{trading_start:%Y-%m-%d %H:%M:%S}'
          AND executiontime <  '{trading_end:%Y-%m-%d %H:%M:%S}'
          AND side = 'BUY';
    """
    # Note: for production use, prefer parameterized queries instead of f-strings.

    with psycopg2.connect(CONNECTION) as conn, conn.cursor() as cursor:
        cursor.execute(query)
        raw = cursor.fetchall()

    columns = ["executiontime", "deliverystart", "price", "volume", "trade_count"]
    df = pd.DataFrame(raw, columns=columns)

    if df.empty:
        logger.warning(" → No trades found for this day, skipping.")
        return

    
    # Normalize timezones to Europe/Berlin
    df["executiontime"] = pd.to_datetime(df["executiontime"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )
    df["deliverystart"] = pd.to_datetime(df["deliverystart"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )


    # 2. Determine bucket index (execution_time_end)
    # -------------------------------------------------------------------------
    # Use UTC for flooring, then convert back to Europe/Berlin to be DST-safe
    
    
    et_utc = df["executiontime"].dt.tz_convert("UTC")
    df["bucket_end"] = (et_utc.dt.floor(f"{bucket_size}min") + pd.Timedelta(minutes=bucket_size)).dt.tz_convert("Europe/Berlin")
    
    df["bucket_start"] = (
        df["executiontime"].dt.tz_convert("UTC")
        .dt.floor(f"{bucket_size}min")
        .dt.tz_convert("Europe/Berlin")
        )

    bucket_index = pd.date_range(
        start=trading_start,
        end=trading_end - pd.Timedelta(minutes=bucket_size),
        freq=f"{bucket_size}min"
        )



    # All quarter-hourly products for this delivery day
    """
    start_of_day = (trading_end - pd.Timedelta(hours=2)).replace(hour=0, minute=0)
    end_of_day = start_of_day.replace(hour=23, minute=45)
    product_index = pd.date_range(start_of_day, end_of_day, freq="15min")
    """

    #new
    day_start = current_day.normalize()  # 00:00 Europe/Berlin
    product_index = pd.date_range(
        start=day_start,
        end=day_start + pd.Timedelta(days=1),
        freq="15min",
        inclusive="left",
    )


    # 3. Aggregate VWAPs
    # -------------------------------------------------------------------------
    # Precompute trade value (price * volume) for clarity
    df["trade_value"] = df["price"] * df["volume"]

    grouped = df.groupby(["bucket_end", "deliverystart"]).agg(
        total_value=("trade_value", "sum"),
        total_volume=("volume", "sum"),
        total_trades=("trade_count", "sum"),
    )

    # Filter out buckets with too few trades
    grouped = grouped[grouped["total_trades"] >= min_trades]

    grouped["vwap"] = grouped["total_value"] / grouped["total_volume"]

    # 4. Build bucket_end x deliverystart matrix
    # -------------------------------------------------------------------------
    matrix = grouped["vwap"].unstack(level="deliverystart")

    # Ensure full index/columns coverage
    matrix = matrix.reindex(index=bucket_index, columns=product_index)

    # Store index/columns as strings; readers will convert back to timestamps
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    # 5. Save to Parquet
    # -------------------------------------------------------------------------
    output_path.mkdir(parents=True, exist_ok=True)
    fname = output_path / f"vwaps_{current_day:%Y-%m-%d}.parquet"

    matrix.to_parquet(fname)
    logger.info(f" → Saved VWAP matrix to {fname}")


def _get_existing_parquet_days(output_path: PathLike) -> Set[pd.Timestamp.date]:
    """
    Scan the output directory for existing VWAP parquet files and extract the
    delivery dates from filenames of the form 'vwaps_YYYY-MM-DD.parquet'.

    Returns
    -------
    set of datetime.date
        All days for which a VWAP parquet file already exists.
    """
    output_path = Path(output_path)

    if not output_path.is_dir():
        return set()

    existing_days: Set[pd.Timestamp.date] = set()
    for fname in os.listdir(output_path):
        if fname.startswith("vwaps_") and fname.endswith(".parquet"):
            date_str = fname[len("vwaps_") : -len(".parquet")]
            try:
                day = pd.to_datetime(date_str).date()
                existing_days.add(day)
            except Exception:
                # Ignore files that do not match the expected naming convention
                continue
    return existing_days


def precompute_range(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    bucket_size: int,
    min_trades: int,
    output_path: PathLike,
) -> None:
    """
    Precompute VWAP matrices for a range of delivery days.

    Features:
    - Resumable: checks which parquet files already exist and skips those days.
    - Processes days in [start_day, end_day).

    Parameters
    ----------
    start_day : pd.Timestamp
        First delivery day (inclusive), tz-aware (Europe/Berlin recommended).
    end_day : pd.Timestamp
        Last delivery day (exclusive).
    bucket_size : int
        Bucket size in minutes.
    min_trades : int
        Minimum trade count per (bucket_end, deliverystart) to compute a VWAP.
    output_path : str | Path
        Directory where VWAP parquet files are stored.
    """
    output_path = Path(output_path)
    existing_days = _get_existing_parquet_days(output_path)

    start_date = start_day.date()
    end_date = end_day.date()

    # Determine start point, potentially resuming from last existing file
    if existing_days:
        # Consider only existing days within the requested range
        relevant_existing = [d for d in existing_days if start_date <= d < end_date]
        if relevant_existing:
            last_existing = max(relevant_existing)
            resume_date = last_existing + pd.Timedelta(days=1)
            d = pd.Timestamp(resume_date, tz=start_day.tzinfo)
            logger.info(
                "Existing VWAP parquet files detected. "
                f"Starting not at {start_day.date()}, "
                f"but resuming from {d.date()} (last existing file: {last_existing})."
            )
        else:
            d = start_day
    else:
        d = start_day

    # Iterate over days and only compute missing VWAP matrices
    while d < end_day:
        if d.date() in existing_days:
            logger.info(f"Parquet for {d.date()} already exists – skipping.")
        else:
            precompute_vwaps_for_day(d, bucket_size, min_trades, output_path)
        d += pd.Timedelta(days=1)




if __name__ == "__main__":
    from src.shared.config import (
        DATA_START, DATA_END, PRECOMPUTED_VWAP_PATH,
        BUCKET_SIZE, MIN_TRADES
    )

    precompute_range(
        start_day=DATA_START,
        end_day=DATA_END,
        bucket_size=BUCKET_SIZE,
        min_trades=MIN_TRADES,
        output_path=PRECOMPUTED_VWAP_PATH,
    )