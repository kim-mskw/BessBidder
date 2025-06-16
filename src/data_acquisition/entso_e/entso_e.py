from typing import Optional

from entsoe import EntsoePandasClient
import os
import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from src.data_acquisition.postgres_db.postgres_db_hooks import ThesisDBHook

load_dotenv()
POSTGRES_USERNAME = os.getenv("POSTGRES_USER")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")

COUNTRY_CODE = "DE_LU"


def fill_database_with_entsoe_data(start: pd.Timestamp, end: pd.Timestamp) -> None:
    months = pd.date_range(start, end, freq="MS")
    entsoe_hook = EntsoeHook(api_key=os.getenv("ENTSOE_API_KEY"))
    thesis_db_hook = ThesisDBHook(username=POSTGRES_USERNAME, hostname=POSTGRES_DB_HOST)

    for month in months:
        da_auction_prices = entsoe_hook.get_day_ahead_auction_prices(
            month, month + relativedelta(months=1)
        )
        intraday_auction_prices = entsoe_hook.get_exaa_prices(
            month, month + relativedelta(months=1)
        )
        demand_df = entsoe_hook.get_demand_forecast_day_ahead(
            month, month + relativedelta(months=1)
        )
        vre_df = entsoe_hook.get_variable_renewables_forecast_day_ahead(
            month, month + relativedelta(months=1)
        )
        thesis_db_hook.upload_entsoe_auction_prices(df=da_auction_prices)
        thesis_db_hook.upload_entsoe_auction_prices(df=intraday_auction_prices)
        thesis_db_hook.upload_entsoe_forecasts(df=demand_df)
        thesis_db_hook.upload_entsoe_forecasts(df=vre_df)


class EntsoeHook:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: Optional[EntsoePandasClient] = None

    @property
    def client(self):
        if self._client is None:
            self._client = EntsoePandasClient(api_key=self._api_key)
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
