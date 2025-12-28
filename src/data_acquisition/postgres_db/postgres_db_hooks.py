import os
from string import Template
from typing import List, Optional

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from loguru import logger
from psycopg2.extensions import connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

load_dotenv()

POSTGRES_URL = Template("postgres://${username}:${password}@${hostname}:${port}/${db_name}")
SQL_ALCHEMY_URL = Template("postgresql://${username}:${password}@${hostname}:${port}/${db_name}")
AFRR_CAPACITY_PRICES_TABLE_NAME = "afrr_capacity_prices"
ENTSOE_FORECASTS_TABLE_NAME = "entsoe_forecasts"
ENTSOE_PRICES_TABLE_NAME = "entsoe_prices"
THESIS_DB_NAME = os.getenv("POSTGRES_DB_NAME")


class ThesisDBHook:
    """This acts as an interface between the PostgreSQL Database and the modules implemented for the thesis.
    Preliminaries for using this hook:
    - A local postgres DB has to be initialized
    - The name of the DB has to be specified in this code"
    """

    def __init__(
        self, username: str, hostname: str, port: str | None = None, password: str | None = None
    ) -> None:
        self._username = username
        self._hostname = hostname
        self._port = port or os.getenv("POSTGRES_PORT", "5432")
        self._password = password or os.getenv("SQL_PASSWORD") or os.getenv("POSTGRES_PASSWORD") or ""
        self._postgres_url = POSTGRES_URL.substitute(
            username=username,
            password=self._password,
            hostname=hostname,
            port=self._port,
            db_name=THESIS_DB_NAME,
        )
        self._sql_alchemy_url = SQL_ALCHEMY_URL.substitute(
            username=username,
            password=self._password,
            hostname=hostname,
            port=self._port,
            db_name=THESIS_DB_NAME,
        )
        self._postgres_connection: Optional = None
        self._sql_alchemy_connection: Optional = None

    @property
    def postgres_connection(self) -> connection:
        if self._postgres_connection is None:
            self._postgres_connection = psycopg2.connect(self._postgres_url)
        return self._postgres_connection

    @property
    def sql_alchemy_connection(self) -> Connection:
        if self._sql_alchemy_connection is None:
            self._sql_alchemy_connection = create_engine(self._sql_alchemy_url)
        return self._sql_alchemy_connection

    def _close_connection(self) -> None:
        """Close the database connection."""
        if self._postgres_connection is not None:
            self._postgres_connection.close()
            self._postgres_connection = None
            logger.info("Database connection closed.")

    def table_exists(self, table_name: str) -> bool:
        query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_name=%s
        );
        """
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (table_name,))
            return cursor.fetchone()[0]

    def column_exists(self, table_name: str, column_name: str) -> bool:
        query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name=%s AND column_name=%s
        );
        """
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(query, (table_name, column_name))
            return cursor.fetchone()[0]

    def _create_table(
        self, table_name: str, columns: str, constraint: str = None
    ) -> None:
        """
        Create a table in the database.

        :param table_name: Name of the table to create.
        :param columns: A string defining the columns and their data types.
                        Example: "id SERIAL PRIMARY KEY, name VARCHAR(100), age INTEGER"
        """
        if self.table_exists(table_name):
            logger.info(f"Table {table_name} already exists.")
            return

        if constraint is None:
            constraint_clause = ""
        else:
            constraint_clause = f", {constraint}"
        create_table_query = (
            f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}{constraint_clause}) ;"
        )
        with self.postgres_connection.cursor() as cursor:
            cursor.execute(create_table_query)
            self.postgres_connection.commit()
            logger.info(f"Table {table_name} created successfully.")

    def _create_afrr_capacity_prices_table(self) -> None:
        self._create_table(
            table_name=AFRR_CAPACITY_PRICES_TABLE_NAME,
            columns="delivery_start TIMESTAMPTZ,"
            "delivery_end TIMESTAMPTZ,"
            "direction TEXT,"
            "capacity_price_90_percentile_eur_per_mw REAL,"
            "capacity_price_90_percentile_eur_per_mw_per_h REAL",
            constraint="UNIQUE (delivery_start, delivery_end, direction)",
        )
        self._close_connection()

    def _upload_multi_index_time_series(
        self, table_name: str, time_series: pd.Series
    ) -> None:
        index_columns = time_series.index.names
        value_column = time_series.name
        cursor = self.postgres_connection.cursor()

        for index_values, value in time_series.items():
            insert_query = f"""
                    INSERT INTO {table_name} ({", ".join(index_columns)}, {value_column})
                    VALUES ({", ".join(["%s"] * len(index_columns))}, %s)
                    ON CONFLICT ({", ".join(index_columns)}) DO UPDATE
                    SET {value_column} = EXCLUDED.{value_column};
                    """
            cursor.execute(insert_query, (*index_values, value))

        self.postgres_connection.commit()
        cursor.close()

    def _upload_single_index_time_series(
        self, table_name: str, time_series: pd.Series
    ) -> None:
        timestamp_column = time_series.index.name
        value_column = time_series.name
        cursor = self.postgres_connection.cursor()
        for timestamp, value in time_series.items():
            insert_query = f"""
            INSERT INTO {table_name} ({timestamp_column}, {value_column})
            VALUES (%s, %s)
            ON CONFLICT ({timestamp_column}) DO UPDATE
            SET {value_column} = EXCLUDED.{value_column};
            """
            cursor.execute(insert_query, (timestamp, value))

        self.postgres_connection.commit()
        cursor.close()

    def upload_afrr_capacity_prices_90_percentile(self, df: pd.DataFrame) -> None:
        self._create_afrr_capacity_prices_table()
        for column in df.columns:
            self._upload_multi_index_time_series(
                table_name=AFRR_CAPACITY_PRICES_TABLE_NAME, time_series=df[column]
            )
        logger.info("aFRR capacity price data has been written to DB successfully.")
        self._close_connection()

    def _create_entsoe_forecasts_table(self) -> None:
        self._create_table(
            table_name=ENTSOE_FORECASTS_TABLE_NAME,
            columns="time TIMESTAMPTZ UNIQUE,"
            "load_forecast_d_minus_1_1000_total_de_lu_mw REAL,"
            "pv_forecast_d_minus_1_1000_de_lu_mw REAL,"
            "wind_offshore_forecast_d_minus_1_1000_de_lu_mw REAL,"
            "wind_onshore_forecast_d_minus_1_1000_de_lu_mw REAL",
        )
        self._close_connection()

    def _create_entsoe_prices_table(self) -> None:
        self._create_table(
            table_name=ENTSOE_PRICES_TABLE_NAME,
            columns="time TIMESTAMPTZ UNIQUE,"
            "epex_spot_60min_de_lu_eur_per_mwh REAL,"
            "exaa_15min_de_lu_eur_per_mwh REAL",
        )
        self._close_connection()

    def upload_entsoe_forecasts(self, df: pd.DataFrame) -> None:
        self._create_entsoe_forecasts_table()
        for column in df.columns:
            self._upload_single_index_time_series(
                time_series=df[column], table_name=ENTSOE_FORECASTS_TABLE_NAME
            )
        logger.info(
            "Entso-e forecast data from {} to {} has been written to DB successfully.",
            df.index.min().isoformat(),
            df.index.max().isoformat(),
        )
        self._close_connection()

    def upload_entsoe_auction_prices(self, df: pd.DataFrame) -> None:
        self._create_entsoe_prices_table()
        for column in df.columns:
            self._upload_single_index_time_series(
                time_series=df[column], table_name=ENTSOE_PRICES_TABLE_NAME
            )
        logger.info(
            "Entso-e price data from {} to {} has been written to DB successfully.",
            df.index.min().isoformat(),
            df.index.max().isoformat(),
        )
        self._close_connection()

    def get_forecast(
        self, ids: List[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        cursor = self.postgres_connection.cursor()
        cursor.execute(
            f"""
                SELECT
                time,
                {",".join(ids)}
                FROM
                {ENTSOE_FORECASTS_TABLE_NAME}
                WHERE
                time BETWEEN '{start}' AND '{end}';
                """
        )
        result = cursor.fetchall()
        cursor.execute("ROLLBACK")
        cursor.close()
        df = pd.DataFrame(result, columns=["time"] + ids)
        df.set_index("time", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        return df

    def get_afrr_capacity_price(
        self, ids: List[str], direction: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        cursor = self.postgres_connection.cursor()
        cursor.execute(
            f"""
                SELECT
                delivery_start,
                delivery_end,
                direction,
                {", ".join(ids)}
                FROM
                {AFRR_CAPACITY_PRICES_TABLE_NAME}
                WHERE
                (delivery_start BETWEEN '{start}' AND '{end}') AND
                (direction = '{direction}');
                """
        )
        result = cursor.fetchall()
        cursor.execute("ROLLBACK")
        cursor.close()
        df = pd.DataFrame(
            result, columns=["delivery_start", "delivery_end", "direction"] + ids
        )
        for name in ids:
            if df[name].isna().all():
                logger.warning(
                    "No data available for id {} between {} and {}",
                    ", ".join(ids),
                    start.isoformat(),
                    end.isoformat(),
                )

        df["delivery_start"] = pd.to_datetime(
            df["delivery_start"], utc=True
        ).dt.tz_convert("Europe/Berlin")
        df["delivery_end"] = pd.to_datetime(df["delivery_end"], utc=True).dt.tz_convert(
            "Europe/Berlin"
        )
        return df

    def get_auction_prices(
        self, id: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        cursor = self.postgres_connection.cursor()
        cursor.execute(
            f"""
                SELECT
                time,
                {id}
                FROM
                {ENTSOE_PRICES_TABLE_NAME}
                WHERE
                time BETWEEN '{start}' AND '{end}';
                """
        )
        result = cursor.fetchall()
        cursor.execute("ROLLBACK")
        cursor.close()
        df = pd.DataFrame(result, columns=["time", id])
        df.set_index("time", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        return df
    

