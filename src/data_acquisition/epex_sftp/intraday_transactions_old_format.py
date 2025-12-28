import logging
import os
import tempfile
import warnings
import zipfile
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import List, Optional

import numpy as np
import pandas as pd
import paramiko
import pytz
from dotenv import load_dotenv
from sqlalchemy import Engine

from src.data_acquisition.postgres_db.postgres_db_hooks import ThesisDBHook

warnings.simplefilter(action="ignore", category=FutureWarning)

DST_SPRING = [
    pd.Timestamp("2018-03-25 02:00:00"),
    pd.Timestamp("2019-03-31 02:00:00"),
    pd.Timestamp("2020-03-29 02:00:00"),
    pd.Timestamp("2021-03-28 02:00:00"),
    pd.Timestamp("2022-03-27 02:00:00"),
    pd.Timestamp("2023-03-26 02:00:00"),
]

load_dotenv()

TRANSACTION_HISTORICAL_DATA_PATH_PREFIX = PurePosixPath(
    "germany", "Intraday Continuous", "EOD", "Historical", "Transactions"
)
TRANSACTION_ZIP_FILE_NAME_PREFIX = "intraday_transactions_germany"

SFTP_HOST = os.getenv("EPEX_SFTP_HOST")
SFTP_PORT = os.getenv("EPEX_SFTP_PORT")
SFTP_USERNAME = os.getenv("EPEX_SFTP_USER")
SFTP_PASSWORD = os.getenv("EPEX_SFTP_PW")

POSTGRES_USERNAME = os.getenv("POSTGRES_USER")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_PASSWORD = os.getenv("SQL_PASSWORD") or os.getenv("POSTGRES_PASSWORD")

BERLIN_TZ = pytz.timezone("Europe/Berlin")


def sftp_connect() -> Optional[paramiko.SFTPClient]:
    """Establish a connection to the SFTP server and list files."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        logging.debug(f"Connecting to {SFTP_HOST}:{SFTP_PORT} as {SFTP_USERNAME}...")

        ssh.connect(
            hostname=SFTP_HOST,
            port=int(SFTP_PORT),
            username=SFTP_USERNAME,
            password=SFTP_PASSWORD,
            timeout=60,
            allow_agent=False,
            look_for_keys=False,
        )

        sftp = ssh.open_sftp()
        logging.debug(f"Connected to {SFTP_HOST}")
        return sftp

    except paramiko.AuthenticationException as e:
        logging.error("Authentication failed, please verify your credentials.")
    except paramiko.SSHException as sshException:
        logging.error(f"SSH connection failed: {sshException}")
    except paramiko.BadHostKeyException as badHostKeyException:
        logging.error(f"Unable to verify server's host key: {badHostKeyException}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def download_intraday_transaction_zip_archive(
    remote_path: PurePosixPath,
    local_path: Path,
    file_name_prefix: str,
    year: int = 2018,
) -> Path:
    """Download the ZIP file from the SFTP server."""
    sftp = sftp_connect()

    files = sftp.listdir(remote_path.as_posix())
    zip_files = [
        file
        for file in files
        if file.startswith(file_name_prefix)
        and file.endswith(".zip")
        and str(year) in file
    ]

    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found with prefix {file_name_prefix}.")

    zip_file_name = sorted(zip_files)[-1]

    remote_file_path = PurePosixPath(remote_path, zip_file_name)
    local_file_path = Path(local_path, zip_file_name)

    sftp.get(remote_file_path.as_posix(), local_file_path)
    # with sftp.file(remote_file_path.as_posix(), 'rb') as remote_file:
    # with open(local_file_path.as_posix(), 'wb') as local_file:
    #     local_file.write(remote_file.read())

    sftp.close()
    return local_file_path


def unpack_archive(path: Path) -> Path:
    """Unpack the downloaded ZIP archive."""
    with zipfile.ZipFile(path.as_posix(), "r") as archive:
        archive.extractall(path.parent.as_posix())
    return path.parent


def fetch_csv_file_names(path: Path) -> List[str]:
    """Fetch all CSV filenames from the directory."""
    files = os.listdir(path)
    csv_files = [file for file in files if file.endswith(".csv")]
    return csv_files


def extract_data_from_csv_file(path: Path, filename: str) -> pd.DataFrame:
    """Extract data from a specific CSV file."""
    full_path = Path(path, filename)
    df = pd.read_csv(
        full_path,
        skiprows=1,
    )
    # Optionally, you can convert columns to specific datetime formats if needed
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").apply(lambda x: x.date())
    df["Time Stamp"] = pd.DatetimeIndex(
        pd.to_datetime(df["Time Stamp"], format="%d/%m/%Y %H:%M:%S", utc=True)
    ).tz_convert("Europe/Berlin")
    return df


def process_hour_column(df, column_name):
    # Extract the hour and quarter-hour part
    hours = df[column_name].str.extract(r"(\d+)(qh\d)?(A|B)?")

    if column_name == "Hour from":
        qh_minutes = (
            hours[1]
            .replace({"qh1": 0, "qh2": 15, "qh3": 30, "qh4": 45})
            .fillna(0)
            .astype(int)
        )
    elif column_name == "Hour to":
        qh_minutes = (
            hours[1]
            .replace({"qh1": 15, "qh2": 30, "qh3": 45, "qh4": 60})
            .fillna(60)
            .astype(int)
        )
    else:
        raise ValueError("Wrong column name. Only Hour from and Hour to allowed.")

    # Adjust hours based on 'A' (standard time) or 'B' (daylight saving time)
    dst_flags = hours[2].replace({"A": False, "B": True})
    dst_flags = dst_flags.fillna(np.nan)

    # Convert hours to integers and handle edge cases
    hours = hours[0].astype(int) - 1  # Adjust to 0-based index

    # Compute final timestamps by combining date, hour, and minutes
    times = (
        pd.to_datetime(df["Date"])
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(qh_minutes, unit="m")
    )
    times = times.apply(lambda x: x + timedelta(hours=1) if x in DST_SPRING else x)

    return times, dst_flags


def derive_product_name(row):
    hour_from = str(row["Hour to"])
    if "qh" in hour_from:
        return "Intraday_Quarter_Hour_Power"
    else:
        return "Intraday_Hour_Power"


def convert_volume_in_mwh(row):
    hour_from = str(row["Hour to"])
    if "qh" in hour_from:
        return row["volume"] / 4
    else:
        return row["volume"]


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(
        {
            "Trade ID": "tradeid",
            "Volume (MW)": "volume",
            "Price (EUR)": "price",
            "Quantity (MW)": "volume",
            "Time Stamp": "executiontime",
        },
        axis=1,
        inplace=True,
    )

    # Be careful, in this data, there is no distinction between BUY and SELL trades (we have to adapt to the data format)
    df["side"] = "BUY"
    df["deliverystart"], dst_start_flags = process_hour_column(df, "Hour from")
    df["deliveryend"], dst_end_flags = process_hour_column(df, "Hour to")

    df["deliverystart"] = df["deliverystart"].dt.tz_localize(
        BERLIN_TZ, ambiguous=dst_start_flags
    )
    df["deliveryend"] = df["deliveryend"].dt.tz_localize(
        BERLIN_TZ, ambiguous=dst_end_flags
    )

    df["product"] = df.apply(derive_product_name, axis=1)
    df["volume"] = df.apply(convert_volume_in_mwh, axis=1)
    df["volumeunit"] = "MWH"
    df["currency"] = "EUR"

    df.drop(
        ["Date", "Market Area Buy", "Market Area Sell", "Hour from", "Hour to"],
        axis=1,
        inplace=True,
    )

    # Filter the DataFrame based on the conditions
    filtered_df = df[
        (df["product"].isin(["XBID_Quarter_Hour_Power", "Intraday_Quarter_Hour_Power"]))
    ].copy()

    # set seconds of execution time to zero, because we definetly do not need them
    filtered_df["executiontime"] = filtered_df["executiontime"].apply(
        lambda x: x.replace(second=0, microsecond=0)
    )

    # Calculate the weighted average price and group by deliverystart and execution time
    grouped_df = (
        filtered_df.groupby(["deliverystart", "executiontime"])
        .apply(
            lambda x: pd.Series(
                {
                    "weighted_avg_price": (x["price"] * x["volume"]).sum()
                    / x["volume"].sum(),
                    "volume": x["volume"].sum(),
                    "trade_count": x.shape[0],
                    "tradeid": x["tradeid"].iloc[0],
                    "side": x["side"].iloc[0],
                    "deliveryend": x["deliveryend"].iloc[0],
                    "product": x["product"].iloc[0],
                    "volumeunit": x["volumeunit"].iloc[0],
                    "currency": x["currency"].iloc[0],
                }
            )
        )
        .reset_index()
    )

    return grouped_df
    # return df


def load_data(df: pd.DataFrame, database: Engine) -> None:
    conn = database.connect()
    df.to_sql(
        "transactions_intraday_de",
        conn,
        if_exists="append",
        index=False,
    )
    conn.close()


def execute_etl_transactions_old_format(years: List[int]) -> None:
    ## be aware: 2022 there was a change in data format
    # -> file for 2022 incomplete (new files "Continuous_Trades-MA-yyyymmdd-yyyymmddThhmmsssssZ")
    thesis_db_hook = ThesisDBHook(
        username=POSTGRES_USERNAME,
        hostname=POSTGRES_DB_HOST,
        port=POSTGRES_PORT,
        password=POSTGRES_PASSWORD,
    )
    database = thesis_db_hook.sql_alchemy_connection
    for year in years:
        with tempfile.TemporaryDirectory() as tmpdirname:
            transaction_archive_location = download_intraday_transaction_zip_archive(
                remote_path=TRANSACTION_HISTORICAL_DATA_PATH_PREFIX,
                local_path=Path(tmpdirname),
                file_name_prefix=TRANSACTION_ZIP_FILE_NAME_PREFIX,
                year=year,
            )
            unpacked_archive_path = unpack_archive(transaction_archive_location)
            filenames = fetch_csv_file_names(path=unpacked_archive_path)
            for idx, filename in enumerate(filenames):
                logging.info(
                    f"Processing {filename} - File #{idx + 1} from {len(filenames)}"
                )
                df = extract_data_from_csv_file(unpacked_archive_path, filename)

                # Remove the file immediately after processing
                file_path = os.path.join(unpacked_archive_path, filename)
                try:
                    os.remove(file_path)
                    logging.info(f"Removed file: {file_path}")
                except OSError as e:
                    logging.error(f"Error removing file {file_path}: {e}")

                # TODO: aggregate transaction data?
                df_for_db = transform_data(df)
                load_data(df_for_db, database)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    years = [2022]
    execute_etl_transactions_old_format(years)
