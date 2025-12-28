import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ---------- DB connection helpers ----------

def _engine_from_env():
    load_dotenv()
    password = os.getenv("SQL_PASSWORD") or ""
    pw = f":{password}" if password else ""
    db = os.getenv("POSTGRES_DB_NAME")
    user = os.getenv("POSTGRES_USER")
    host = os.getenv("POSTGRES_DB_HOST")
    if not all([db, user, host]):
        raise RuntimeError("POSTGRES_DB_NAME / POSTGRES_USER / POSTGRES_DB_HOST missing in environment.")
    url = f"postgresql://{user}{pw}@{host}/{db}"
    return create_engine(url)

# ---------- SQL blocks ----------

SQL_RECREATE_IDFULL_H_FROM_QH = """
DROP TABLE IF EXISTS public.id_full_h_from_qh;
CREATE TABLE public.id_full_h_from_qh AS
SELECT
  -- Aggregation in UTC-Stunden:
  date_trunc('hour', deliverystart AT TIME ZONE 'UTC')  AS hour_start_utc,
  SUM(weighted_avg_price * volume) / NULLIF(SUM(volume),0) AS id_full_h,
  SUM(volume)                                              AS hour_volume
FROM public.transactions_intraday_de
WHERE deliverystart IS NOT NULL
  AND volume > 0
GROUP BY date_trunc('hour', deliverystart AT TIME ZONE 'UTC')
ORDER BY hour_start_utc;

CREATE UNIQUE INDEX IF NOT EXISTS idx_id_full_h_from_qh
  ON public.id_full_h_from_qh (hour_start_utc);
"""


# ---------- core operations ----------

def rebuild_idfull_table():
    """(Re)creates public.id_full_h_from_qh from public.transactions_intraday_de."""
    eng = _engine_from_env()
    with eng.begin() as con:
        # Run each statement separately
        for stmt in SQL_RECREATE_IDFULL_H_FROM_QH.strip().split(";\n"):
            s = stmt.strip()
            if s:
                con.execute(text(s))


def load_idfull(only_start=None, only_end=None) -> pd.DataFrame:
    """Loads hourly IDFull from DB as tz-aware (UTC) indexed DataFrame."""
    eng = _engine_from_env()
    base = "SELECT hour_start_utc, id_full_h FROM public.id_full_h_from_qh"
    where = []
    params = {}

    if only_start is not None:
        where.append("hour_start_utc >= :start")
        ts = pd.Timestamp(only_start)
    
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        params["start"] = ts

    if only_end is not None:
        where.append("hour_start_utc <= :end")
        ts = pd.Timestamp(only_end)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        params["end"] = ts

    q = base + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY hour_start_utc"

    df = pd.read_sql(text(q), eng, params=params, parse_dates=["hour_start_utc"]).set_index("hour_start_utc")

    # Make sure index is tz-aware UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df

def merge_idfull_into_csv(csv_in: str, csv_out: str = None) -> pd.DataFrame:
    """
    Merges DB hourly IDFull into the given hourly CSV by matching the 'time' column.
    Assumes the CSV 'time' represents UTC (e.g., 2019-01-01 00:00:00+00:00).
    """
    if csv_out is None:
        csv_out = csv_in

    # Load CSV
    df = pd.read_csv(csv_in, parse_dates=["time"]).set_index("time")

    # Make sure index is tz-aware UTC
    if df.index.tz is None:
        # If it has "+00:00" in string, pandas already parsed as tz-aware; if not, localize to UTC
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Load IDFull for same range (optional optimization)
    start, end = df.index.min(), df.index.max()
    idfull = load_idfull(only_start=start, only_end=end)

    if "id_full_h" in df.columns:
        df.drop(columns=["id_full_h"], inplace=True)
        
    # Join
    merged = df.join(idfull, how="left")
    
    # duplicate id_full_h -> id_full_qh (exact copy)
    merged["id_full_qh"] = merged["id_full_h"]

    # Save
    merged.to_csv(csv_out)
    return merged

def run_build_idfull_and_merge(csv_in: str, csv_out: str = None) -> pd.DataFrame:
    """
    1) Rebuilds id_full_h_from_qh in DB
    2) Merges it into the given hourly CSV
    3) Saves as *_with_idfull.csv (or csv_out) and returns the merged DataFrame
    """
    rebuild_idfull_table()
    return merge_idfull_into_csv(csv_in=csv_in, csv_out=csv_out)

# ---------- CLI ----------

def _cli():
    ap = argparse.ArgumentParser(description="Rebuild hourly IDFull from QH trades and merge into hourly CSV.")
    ap.add_argument("--csv-in", required=True, help="Path to hourly CSV (e.g., data/data_2019-01-01_2024-01-01_hourly.csv)")
    ap.add_argument("--csv-out", default=None, help="Optional output path; defaults to *_with_idfull.csv")
    args = ap.parse_args()
    merged = run_build_idfull_and_merge(args.csv_in, args.csv_out)
    # quick sanity prints
    n_missing = merged["id_full_h"].isna().sum()
    print(f"[OK] Merged. Missing id_full_h rows: {n_missing}. Saved to: {args.csv_out or (os.path.splitext(args.csv_in)[0] + '_with_idfull.csv')}")

if __name__ == "__main__":
    _cli()