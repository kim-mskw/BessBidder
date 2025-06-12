"""
Data acquisition and ETL script for energy market data.

This script:
- Loads environmental variables for database and API connections.
- Fetches energy market data from ENTSO-E, EPEX Spot, and EXAA APIs.
- Transforms and stores the data into a PostgreSQL database.
- Resamples the data into hourly granularity and saves it as CSV files.

Requires:
- PostgreSQL credentials and database connection details in .env file.
- ETL functions for EPEX Spot and EXAA data formats.
"""

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.data_acquisition.entso_e.entso_e import fill_database_with_entsoe_data
from src.data_acquisition.epex_sftp.intraday_transactions_new_format import (
    execute_etl_transactions_new_format,
)
from src.data_acquisition.epex_sftp.intraday_transactions_old_format import (
    execute_etl_transactions_old_format,
)
from src.data_acquisition.postgres_db.postgres_db_hooks import ThesisDBHook

from src.shared.config import START, END

# Load environment variables from .env file
load_dotenv()

# Configure pandas plotting backend
pd.options.plotting.backend = "plotly"

# Fetch PostgreSQL credentials from environment variables
POSTGRES_USERNAME = os.getenv("POSTGRES_USER")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")

# Start and end dates for data extraction
start = START
end = END

# -------------------------------------------------------
# WRITE DATA INTO DATABASE

# ETL Transaction data from epex spot sftp (new format)
# execute_etl_transactions_new_format(years=[2021, 2022, 2023])


# ETL Transaction data from epex spot sftp (old format)
# execute_etl_transactions_old_format(years=[2020])


# Add ENTSO-E Data to the database
fill_database_with_entsoe_data(start, end)

# -------------------------------------------------------
#  GET DATA AND STORE IN CSV

# Initialize database connection to fetch auction prices and forecasts
thesis_db_hook = ThesisDBHook(username=POSTGRES_USERNAME, hostname=POSTGRES_DB_HOST)

# Fetch auction prices for EXAA and EPEX Spot markets
exaa_prices = thesis_db_hook.get_auction_prices(
    start=start, end=end, id="exaa_15min_de_lu_eur_per_mwh"
)

# Fetch auction prices for EPEX Spot (60 min resolution)
da_auction_prices_60 = thesis_db_hook.get_auction_prices(
    start=start, end=end, id="epex_spot_60min_de_lu_eur_per_mwh"
)

# Fetch auction prices for EXAA (15 min resolution)
exaa_auction_prices = thesis_db_hook.get_auction_prices(
    start=start, end=end, id="exaa_15min_de_lu_eur_per_mwh"
)

# Fetch demand forecast data
demand_df = thesis_db_hook.get_forecast(
    ids=["load_forecast_d_minus_1_1000_total_de_lu_mw"], start=start, end=end
)

# Fetch VRE (Variable Renewable Energy) forecast data for PV, wind onshore, and offshore
vre_df = thesis_db_hook.get_forecast(
    ids=[
        "pv_forecast_d_minus_1_1000_de_lu_mw",
        "wind_offshore_forecast_d_minus_1_1000_de_lu_mw",
        "wind_onshore_forecast_d_minus_1_1000_de_lu_mw",
    ],
    start=start,
    end=end,
)

# Combine all data into one DataFrame
data = pd.concat([da_auction_prices_60, exaa_auction_prices, demand_df, vre_df], axis=1)

# Resample the data into hourly frequency (mean)
data_hourly = data.resample("h").mean()

# Format the start and end dates for the output file name
data_start = START.tz_convert("Europe/Berlin").date().isoformat()
data_end = END.tz_convert("Europe/Berlin").date().isoformat()

# Set the output path for saving the data
output_path = Path("data")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Save the hourly data as a CSV file
data_hourly.to_csv(Path(output_path, f"data_{data_start}_{data_end}_hourly.csv"))
