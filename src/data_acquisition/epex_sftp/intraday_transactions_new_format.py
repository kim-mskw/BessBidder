import logging
import os
import shutil
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

#warnings.simplefilter(action="ignore", category=FutureWarning)
import platform

if platform.system() == "Darwin":
    os.environ.setdefault("TMPDIR", "/tmp")
    tempfile.tempdir = "/tmp"

load_dotenv()

TRANSACTION_HISTORICAL_DATA_PATH_PREFIX = PurePosixPath(
    "germany", "Intraday Continuous", "EOD", "Historical", "Transactions"
)
TRANSACTION_ZIP_FILE_NAME_PREFIX = "Continuous_Trades-DE"

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
    if sftp is None:
        raise ConnectionError("SFTP connection failed")

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
    """Recursively unpack a ZIP archive, including any nested ZIP files."""
    with zipfile.ZipFile(path.as_posix(), "r") as archive:
        extract_path = path.parent
        archive.extractall(extract_path)

    nested_extract_path = extract_path
    Path(nested_extract_path).mkdir(parents=True, exist_ok=True)


    for extracted_file in extract_path.glob("**/*.zip"):
        with zipfile.ZipFile(extracted_file, "r") as nested_archive:
            nested_archive.extractall(nested_extract_path)
        extracted_file.unlink()

    return extract_path


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

    df["DeliveryStart"] = pd.DatetimeIndex(
        pd.to_datetime(df["DeliveryStart"], utc=True)
    ).tz_convert("Europe/Berlin")
    df["DeliveryEnd"] = pd.DatetimeIndex(
        pd.to_datetime(df["DeliveryEnd"], utc=True)
    ).tz_convert("Europe/Berlin")
    df["ExecutionTime"] = pd.DatetimeIndex(
        pd.to_datetime(df["ExecutionTime"], utc=True)
    ).tz_convert("Europe/Berlin")

    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(
        {
            "TradeId": "tradeid",
            "Side": "side",
            "Product": "product",
            "Volume": "volume",
            "Price": "price",
            "ExecutionTime": "executiontime",
            "DeliveryStart": "deliverystart",
            "DeliveryEnd": "deliveryend",
            "Currency": "currency",
            "VolumeUnit": "volumeunit",
        },
        axis=1,
        inplace=True,
    )

    df.drop(
        [
            "RemoteTradeId",
            "TradePhase",
            "UserDefinedBlock",
            "SelfTrade",
            "OrderID",
            "DeliveryArea",
        ],
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

    # Calculate the weighted average price and group by deliverystart
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


def load_data(df: pd.DataFrame, database: Engine) -> None:
    conn = database.connect()
    df.to_sql(
        "transactions_intraday_de",
        conn,
        chunksize=10000,
        if_exists="append",
        index=False,
    )
    conn.close()


def execute_etl_transactions_new_format(years: List[int]) -> None:
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
        with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmpdirname:
            transaction_archive_location = download_intraday_transaction_zip_archive(
                remote_path=TRANSACTION_HISTORICAL_DATA_PATH_PREFIX,
                local_path=Path(tmpdirname),
                file_name_prefix=TRANSACTION_ZIP_FILE_NAME_PREFIX,
                year=year,
            )
            logging.info(f"[{year}] ZIP downloaded to: {transaction_archive_location}")

            unpacked_archive_path = unpack_archive(transaction_archive_location)
            filenames = fetch_csv_file_names(path=unpacked_archive_path)
            logging.info(f"[{year}] Found {len(filenames)} CSV files in {unpacked_archive_path}")
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

                df_for_db = transform_data(df)
                load_data(df_for_db, database)
            shutil.rmtree(unpacked_archive_path, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    years = [2024]
    execute_etl_transactions_new_format(years)
