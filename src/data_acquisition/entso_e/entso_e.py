from time import time
from typing import Optional

from entsoe import EntsoePandasClient
import os
import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from src.data_acquisition.postgres_db.postgres_db_hooks import ThesisDBHook
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from entsoe import EntsoePandasClient
import requests


load_dotenv()
POSTGRES_USERNAME = os.getenv("POSTGRES_USER")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_PASSWORD = os.getenv("SQL_PASSWORD") or os.getenv("POSTGRES_PASSWORD")

COUNTRY_CODE = "DE_LU"


def fill_database_with_entsoe_data(start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    Fetch ENTSO-E data in smaller chunks (weekly) to avoid API point limits.

    Holt:
    - DA 60min Preise
    - DA 15min Preise
    - Load D-1 Forecast
    - VRE (PV / Wind on/offshore) D-1 Forecast

    und schreibt alles direkt in die Postgres-Tabellen.
    """

    entsoe_hook = EntsoeHook(api_key=os.getenv("ENTSOE_API_KEY"))
    thesis_db_hook = ThesisDBHook(
        username=POSTGRES_USERNAME,
        hostname=POSTGRES_DB_HOST,
        port=POSTGRES_PORT,
        password=POSTGRES_PASSWORD,
    )

    # ------------------------------------------------------------
    # ALTER CODE (monatlich) – behalten, aber auskommentieren
    # ------------------------------------------------------------
    """
    months = pd.date_range(start, end, freq="MS")

    for month in months:
        month_end = month + relativedelta(months=1)
        print(f"\n[INFO] Processing {month.date()} → {month_end.date()}")

        # 1) DA auction prices (60min)
        try:
            da_auction_prices = entsoe_hook.get_day_ahead_auction_prices(month, month_end)
            print(f"  DA 60min rows: {len(da_auction_prices)}")
            if len(da_auction_prices):
                thesis_db_hook.upload_entsoe_auction_prices(df=da_auction_prices)
                print("  → inserted DA 60min")
        except Exception as e:
            print(f"  [WARN] DA 60min failed: {e}")

        # 2) 15min day-ahead prices
        try:
            exaa_15min = entsoe_hook.get_exaa_prices(month, month_end)
            print(f"  DA 15min rows: {len(exaa_15min)}")
            if len(exaa_15min):
                thesis_db_hook.upload_entsoe_auction_prices(df=exaa_15min)
                print("  → inserted DA 15min")
        except Exception as e:
            print(f"  [WARN] DA 15min failed: {e}")

        # 3) Demand forecast (D-1)
        try:
            demand_df = entsoe_hook.get_demand_forecast_day_ahead(month, month_end)
            print(f"  Load forecast rows: {len(demand_df)}")
            if len(demand_df):
                thesis_db_hook.upload_entsoe_forecasts(df=demand_df)
                print("  → inserted load forecast")
        except Exception as e:
            print(f"  [WARN] Load forecast failed: {e}")

        # 4) Variable renewables (wind/solar) forecast (D-1)
        try:
            vre_df = entsoe_hook.get_variable_renewables_forecast_day_ahead(month, month_end)
            print(f"  VRE forecast rows: {len(vre_df)}")
            if len(vre_df):
                thesis_db_hook.upload_entsoe_forecasts(df=vre_df)
                print("  → inserted VRE forecast")
        except Exception as e:
            print(f"  [WARN] VRE forecast failed: {e}")
    """

    # ------------------------------------------------------------
    # NEUER CODE: wöchentliche Chunks
    # ------------------------------------------------------------

    current = start

    # sicherstellen, dass beide tz-aware sind (falls nötig)
    if current.tz is None:
        current = current.tz_localize("Europe/Berlin")
    if end.tz is None:
        end = end.tz_localize("Europe/Berlin")

    while current < end:
        chunk_end = current + relativedelta(days=7)
        if chunk_end > end:
            chunk_end = end

        print(f"\n[INFO] Processing chunk {current} → {chunk_end}")

        # 1) DA auction prices (60min)
        try:
            da_auction_prices = entsoe_hook.get_day_ahead_auction_prices(current, chunk_end)
            print(f"  DA 60min rows: {len(da_auction_prices)}")
            if len(da_auction_prices):
                thesis_db_hook.upload_entsoe_auction_prices(df=da_auction_prices)
                print("  → inserted DA 60min")
        except Exception as e:
            print(f"  [WARN] DA 60min failed: {e}")

        # 2) 15min day-ahead prices (EXAA)
        try:
            exaa_15min = entsoe_hook.get_exaa_prices(current, chunk_end)
            print(f"  DA 15min rows: {len(exaa_15min)}")
            if len(exaa_15min):
                thesis_db_hook.upload_entsoe_auction_prices(df=exaa_15min)
                print("  → inserted DA 15min")
        except Exception as e:
            print(f"  [WARN] DA 15min failed: {e}")

        # 3) Demand forecast (D-1)
        try:
            demand_df = entsoe_hook.get_demand_forecast_day_ahead(current, chunk_end)
            print(f"  Load forecast rows: {len(demand_df)}")
            if len(demand_df):
                thesis_db_hook.upload_entsoe_forecasts(df=demand_df)
                print("  → inserted load forecast")
        except Exception as e:
            print(f"  [WARN] Load forecast failed: {e}")

        # 4) Variable renewables (wind/solar) forecast (D-1)
        try:
            vre_df = entsoe_hook.get_variable_renewables_forecast_day_ahead(current, chunk_end)
            print(f"  VRE forecast rows: {len(vre_df)}")
            if len(vre_df):
                thesis_db_hook.upload_entsoe_forecasts(df=vre_df)
                print("  → inserted VRE forecast")
        except Exception as e:
            print(f"  [WARN] VRE forecast failed: {e}")

        # optional: bisschen Pause, um die API zu schonen
        # time.sleep(0.2)

        current = chunk_end  # nächster Block

    print("\n[INFO] Done.")



class EntsoeHook:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: Optional[EntsoePandasClient] = None

    @property
    def client(self):
        if self._client is None:
            # Important: short timeouts to avoid hanging HTTP reads,
            # and fewer retries to fail fast and move on to the next month.
            self._client = EntsoePandasClient(
                api_key=self._api_key,
                retry_count=2,   # reduce default retry attempts
                timeout=30       # seconds: applies to both connect & read
            )

            # Harden the underlying requests session (handles 429/5xx nicely).
            sess: requests.Session = self._client.session
            retries = Retry(
                total=3,                 # overall retry budget
                connect=2,
                read=2,
                backoff_factor=1.0,      # 1s, 2s, 4s between retries
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"]
            )
            adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)

        return self._client

    def get_demand_forecast_day_ahead(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        df = self.client.query_load_forecast(COUNTRY_CODE, start=start, end=end)
        df.index.name = "time"
        df.index = df.index.tz_convert("utc")
        df.rename(
            columns={"Forecasted Load": "load_forecast_d_minus_1_1000_total_de_lu_mw"},
            inplace=True,
        )
        return df.round(3)

    def get_variable_renewables_forecast_day_ahead(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        df = self.client.query_wind_and_solar_forecast(
            COUNTRY_CODE, start=start, end=end
        )
        df.index.name = "time"
        df.index = df.index.tz_convert("utc")
        df.rename(
            columns={
                "Solar": "pv_forecast_d_minus_1_1000_de_lu_mw",
                "Wind Offshore": "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
                "Wind Onshore": "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
            },
            inplace=True,
        )
        return df.round(3)

    def get_day_ahead_auction_prices(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        df = self.client.query_day_ahead_prices(
            COUNTRY_CODE,
            start=start,
            end=end,
            resolution="60min",
        )
        df.index.name = "time"
        df.index = df.index.tz_convert("utc")
        df.name = "epex_spot_60min_de_lu_eur_per_mwh"
        df = pd.DataFrame(df)
        return df.round(3)

    def get_exaa_prices(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        df = self.client.query_day_ahead_prices(
            COUNTRY_CODE,
            start=start,
            end=end,
            resolution="15min",
        )
        df.index.name = "time"
        df.index = df.index.tz_convert("utc")
        df.name = "exaa_15min_de_lu_eur_per_mwh"
        df = pd.DataFrame(df)
        return df.round(3)

