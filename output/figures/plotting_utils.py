"""
Utility functions and styling for plotting figures in the ACM e-Energy paper.

Includes:
- Custom colour definitions
- Page dimensions for ACM layout
- Plot styling function for LaTeX integration
- Helpers for State-of-Charge (SoC) statistics and alignment
"""

import matplotlib.pyplot as plt
import pandas as pd

# Custom color palette
colors = {
    "pastel_burgundy": "#5EBEC4",
    "light_blue": "#F92C85",
    "pastel_green": "#F67EB7",
    "pastel_yellow": "#9CD4D8",
}

# ACM paper text dimensions (in inches)
page_width = 7.09
page_height = 9.25


def set_acm_plot_style():
    """
    Apply ACM-compatible LaTeX plot styling to matplotlib.

    This function modifies the global matplotlib `rcParams` to use LaTeX
    rendering, Libertine font, and formatting compatible with ACM publications.
    """
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.latex.preamble": r"\usepackage[utf8]{inputenc}\usepackage{libertine}\usepackage{textcomp}\newcommand{\euro}{\texteuro}",
            "font.family": "libertine",
            "font.size": 9,
            "legend.fontsize": 8,
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


def add_soc_statistics(df):
    """
    Add descriptive statistics to a DataFrame containing State-of-Charge (SoC) values.

    Computes the mean and the 20th and 80th percentiles across columns for each row
    and adds them as new columns.

    Args:
        df (pd.DataFrame): DataFrame with SoC values, one column per trial/day.

    Returns:
        pd.DataFrame: Input DataFrame with additional columns:
            - "mean"
            - "0.2 quantile"
            - "0.8 quantile"
    """
    columns = df.columns
    df["mean"] = df[columns].mean(axis=1)
    df["0.2 quantile"] = df[columns].quantile(0.2, axis=1)
    df["0.8 quantile"] = df[columns].quantile(0.8, axis=1)
    return df


def align_to_date(df, target_date):
    """
    Reindex a DataFrame to a specific target date while preserving time-of-day. USed for aligning daily data across 24 hours.

    Args:
        df (pd.DataFrame): DataFrame with a DateTimeIndex.
        target_date (str): Target date in "YYYY-MM-DD" format.

    Returns:
        pd.DataFrame: DataFrame reindexed to the target_date, with the original
        time-of-day preserved. If fewer than 24 timestamps are present, the result
        is padded to 24 hourly values.
    """
    times = df.index.strftime("%H:%M:%S")
    new_index = pd.to_datetime(target_date + " " + times)
    df = df.set_index(new_index)

    if len(times) < 24:
        new_index = pd.date_range(start=new_index.min(), periods=24, freq="H")
        df = df.reindex(new_index)

    return df


def derive_soc_from_trades(
    trade_df: pd.DataFrame, efficiency: float, start_of_day, end_of_day
):
    """
    Derive State-of-Charge (SoC) from trade data.
    This function processes a DataFrame of trades to summarise the net position,
    volume, and price-weighted average (VWAP) for each product over a specified
    time period. It calculates the SoC based on the net charge and discharge
    volumes, and returns a DataFrame indexed by 15-minute intervals.

    Args:
        trade_df (pd.DataFrame): DataFrame containing trade data with columns:
            - "product": timestamp of the product in seconds
            - "side": "buy" or "sell"
            - "quantity": volume of the trade
            - "price": price of the trade
        efficiency (float): Efficiency factor for charge/discharge calculations.
        start_of_day (int): Start of the day in seconds since epoch.
        end_of_day (int): End of the day in seconds since epoch.

    Returns:
        pd.DataFrame: DataFrame indexed by 15-minute intervals with columns:
            - "sell_volume": total volume of sell trades
            - "buy_volume": total volume of buy trades
            - "net_position": net position (buy - sell)
            - "net_discharge": total discharge volume
            - "net_charge": total charge volume
            - "vwap_sell": volume-weighted average price for sell trades
            - "vwap_buy": volume-weighted average price for buy trades
            - "pnl": profit and loss based on VWAPs and volumes
            - "soc": state of charge calculated from net charge/discharge
    """
    trade_df = trade_df.copy()
    trade_df["product"] = [
        pd.to_datetime(x, unit="s", utc=True).tz_convert("Europe/Berlin")
        for x in trade_df["product"]
    ]
    grouped_by_product = trade_df.groupby("product")

    per_product_data = {}

    for idx, product_df in grouped_by_product:
        if product_df.empty:
            continue
        product = product_df["product"].iloc[0]
        sells = product_df[product_df["side"] == "sell"]
        sell_volume = sells["quantity"].sum()
        buys = product_df[product_df["side"] == "buy"]
        buy_volume = buys["quantity"].sum()

        net_position = (-1) * sell_volume + buy_volume

        vwap_sell = (sells["quantity"] * sells["price"]).sum() / sells["quantity"].sum()
        vwap_buy = (buys["quantity"] * buys["price"]).sum() / buys["quantity"].sum()

        pnl = buy_volume * (-1) * vwap_buy + sell_volume * vwap_sell

        net_charge = 0
        net_discharge = 0
        if net_position >= 0:
            net_charge = net_position
        else:
            net_discharge = (-1) * net_position

        per_product_data.update(
            {
                product: {
                    "sell_volume": sell_volume,
                    "buy_volume": buy_volume,
                    "net_position": net_position,
                    "net_discharge": net_discharge,
                    "net_charge": net_charge,
                    "vwap_sell": vwap_sell,
                    "vwap_buy": vwap_buy,
                    "pnl": pnl,
                }
            }
        )

    per_product_data_df = pd.DataFrame.from_dict(per_product_data).T
    start_of_day = pd.to_datetime(start_of_day, unit="s", utc=True).tz_convert(
        "Europe/Berlin"
    )
    end_of_day = pd.to_datetime(end_of_day, unit="s", utc=True).tz_convert(
        "Europe/Berlin"
    ) + pd.Timedelta(minutes=15)

    per_product_data_df = per_product_data_df.reindex(
        pd.date_range(start_of_day, end_of_day, freq="15min"), fill_value=0.0
    )
    per_product_data_df["soc"] = (
        per_product_data_df["net_charge"] + (-1) * per_product_data_df["net_discharge"]
    ).cumsum()
    per_product_data_df = per_product_data_df.shift(1).fillna(0)
    per_product_data_df.index = [x.timestamp() for x in per_product_data_df.index]
    return per_product_data_df
