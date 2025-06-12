"""
Myopic Multi Market profit and State-of-Charge (SoC) plotting script for ACM e-Energy paper.

Generates:
1. A cumulative profit duration plot for Day-Ahead (DA) and Intraday Continuous (IDC) markets.
2. A State-of-Charge (SoC) comparison plot between DA and IDC strategies.


Requires:
- DA and IDC result CSVs for the naive strategy
- VWAP and trade files for SoC reconstruction

The script assumes a specific folder structure and preprocessed CSV data.
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Polygon
from plotting_utils import (
    add_soc_statistics,
    align_to_date,
    colors,
    derive_soc_from_trades,
    page_height,
    page_width,
    set_acm_plot_style,
)

if __name__ == "__main__":
    set_acm_plot_style()

    path_da = Path("output/myopic_multi_market/day_ahead_milp/11-12.2020_ACM.csv")
    path_idc_base = Path(
        "output/myopic_multi_market/rolling_intrinsic_stacked_on_day_ahead_qh/bs15cr1rto0.86mc365mt10"
    )

    # Day-Ahead profit: aggregate daily
    df_da = pd.read_csv(path_da)
    df_da.index = pd.to_datetime(df_da["time"], utc=True).dt.tz_convert("Europe/Berlin")
    df_da = df_da.drop_duplicates(keep="last")
    df_da["profit"] = df_da["discharge_revenues"] + df_da["charge_costs"]
    da_daily = df_da.groupby(df_da.index.date)["profit"].sum()

    # Intraday Continuous profit: clean & aggregate daily
    df_idc = pd.read_csv(path_idc_base / "profit.csv", parse_dates=["day"])
    df_idc = df_idc.drop_duplicates(subset="day", keep="last")

    drop_dates = ["2020-11-15", "2020-12-27", "2020-12-31"]
    df_idc = df_idc[~df_idc["day"].dt.strftime("%Y-%m-%d").isin(drop_dates)]
    da_daily = da_daily[~da_daily.index.astype(str).isin(drop_dates)]
    idc_daily = df_idc.set_index(df_idc["day"].dt.date)["profit"]

    # Merge DA and IDC into one DataFrame
    profits = pd.merge(
        da_daily.sort_index()
        .reset_index()
        .rename(columns={"profit": "DA", "index": "date"}),
        idc_daily.sort_index()
        .reset_index()
        .rename(columns={"profit": "IDC", "day": "date"}),
        on="date",
        how="inner",
    ).set_index("date")

    profits = profits.sort_values(by="IDC", ascending=True)
    index_hours = range(1, len(profits) + 1)

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(page_width / 1.5, page_height / 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # -----------------------------------
    # CUMULATIVE PROFIT PLOT
    # -----------------------------------
    overall_profit_idc = profits["IDC"].sum() - profits["DA"].sum()
    overall_profit_da = profits["DA"].sum()
    overall_profit = profits["IDC"].sum()

    hatch_legend = [
        Patch(
            facecolor="white", edgecolor="white", label=f"SUM: {overall_profit:.2f} €"
        ),
        Patch(
            facecolor=colors["pastel_green"],
            edgecolor="black",
            hatch="//",
            label=f"IDC: {overall_profit_idc:.2f} €",
        ),
        Patch(
            facecolor=colors["pastel_yellow"],
            edgecolor="black",
            hatch="\\",
            label=f"DA: {overall_profit_da:.2f} €",
        ),
    ]

    cumsum_IDC = profits["IDC"].cumsum()
    cumsum_DA = profits["DA"].cumsum()

    axs[0].plot(index_hours, cumsum_IDC, label="IDC Profit", color=colors["light_blue"])
    polygon_positive = Polygon(
        list(zip(index_hours, cumsum_IDC))
        + list(zip(index_hours[::-1], cumsum_DA[::-1])),
        closed=True,
        facecolor=colors["pastel_green"],
        edgecolor="black",
        hatch="//",
        alpha=0.5,
    )
    axs[0].add_patch(polygon_positive)

    axs[0].plot(
        index_hours, cumsum_DA, label="DA Profit", color=colors["pastel_burgundy"]
    )
    polygon_positive_da = Polygon(
        [(index_hours[0], 0)]
        + list(zip(index_hours, cumsum_DA))
        + [(index_hours[-1], 0)],
        closed=True,
        facecolor=colors["pastel_yellow"],
        edgecolor="black",
        hatch="\\",
        alpha=0.5,
    )
    axs[0].add_patch(polygon_positive_da)

    axs[0].set_xlabel("Day number")
    axs[0].set_ylabel("Cumulative profit in €")
    axs[0].legend(
        handles=hatch_legend, loc="upper left", title="Overall Profit", frameon=False
    )
    axs[0].margins(0)
    axs[0].set_ylim(0, 5000)

    # -----------------------------------
    # STATE OF CHARGE (SOC) PLOT
    # -----------------------------------
    target_date = "2024-01-01"
    soc_id_df = pd.DataFrame()
    soc_da_df = pd.DataFrame()

    for i, date in enumerate(idc_daily.index):
        path_trades = path_idc_base / f"trades/trades_{date}.csv"
        trades = pd.read_csv(path_trades, index_col=0)
        trades.index = (
            pd.to_datetime(trades.index, utc=True)
            .tz_convert("Europe/Berlin")
            .astype("int64")
            // 10**9
        )
        trades["product"] = (
            pd.to_datetime(trades["product"], utc=True)
            .dt.tz_convert("Europe/Berlin")
            .astype("int64")
            // 10**9
        )

        path_vwaps = path_idc_base / f"vwap/vwaps_{date}.csv"
        vwaps = pd.read_csv(path_vwaps, index_col=0)
        date_vwaps = pd.to_datetime(vwaps.index[0]).date()
        vwaps.columns = pd.date_range(date_vwaps, periods=96, freq="15T")
        vwaps.index = (
            pd.to_datetime(vwaps.index, utc=True)
            .tz_convert("Europe/Berlin")
            .astype("int64")
            // 10**9
            - 900
        )
        vwaps.columns = (
            pd.to_datetime(vwaps.columns, utc=True)
            .tz_convert("Europe/Berlin")
            .astype("int64")
            // 10**9
        )
        vwaps = vwaps.T.sort_index()

        soc_id = derive_soc_from_trades(trades, 1, vwaps.columns[0], vwaps.columns[-1])[
            "soc"
        ]
        soc_id = pd.DataFrame(soc_id / 4 * 100).iloc[32:]
        soc_id.index = pd.to_datetime(soc_id.index, unit="s", utc=True).tz_convert(
            "Europe/Berlin"
        )
        soc_id = align_to_date(soc_id, target_date)
        soc_id_df[f"soc_{i}"] = soc_id

        soc_da = pd.DataFrame(df_da["soc"][df_da.index.date == date] * 100)
        soc_da = align_to_date(soc_da, target_date)
        soc_da_df[f"soc_{i}"] = soc_da

    soc_id_df = add_soc_statistics(soc_id_df)
    soc_da_df = add_soc_statistics(soc_da_df)

    axs[1].fill_betweenx(
        soc_id_df.index,
        soc_id_df["0.2 quantile"],
        soc_id_df["0.8 quantile"],
        color=colors["pastel_green"],
        alpha=0.5,
        label="Quantiles IDC",
    )
    axs[1].plot(
        soc_id_df["mean"],
        soc_id_df.index,
        label="Mean IDC",
        color=colors["light_blue"],
        linewidth=1.5,
    )

    axs[1].fill_betweenx(
        soc_da_df.index,
        soc_da_df["0.2 quantile"],
        soc_da_df["0.8 quantile"],
        color=colors["pastel_yellow"],
        alpha=0.5,
        label="Quantiles DA",
    )
    axs[1].plot(
        soc_da_df["mean"],
        soc_da_df.index,
        label="Mean DA",
        color=colors["pastel_burgundy"],
        linewidth=1.5,
    )

    axs[1].set_xlabel("State of Charge in %")
    axs[1].set_xticks(np.arange(0, 101, 50))
    axs[1].set_xlim(0, 100)

    date_range = pd.date_range(
        soc_da_df.index[0], soc_da_df.index[-1] + pd.Timedelta(minutes=15), freq="2H"
    )
    axs[1].set_yticks(date_range)
    axs[1].set_yticklabels(date_range.strftime("%H:%M"))
    axs[1].set_ylim(date_range[0], date_range[-1])

    # Clean up duplicated legend entries
    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[1].legend(by_label.values(), by_label.keys(), loc="lower right")

    plt.tight_layout()
    plt.savefig("output/figures/profit_plot_Myopic_Multi_Market.png")
    plt.savefig("output/figures/profit_plot_Myopic_Multi_Market.pdf", format="pdf")
    plt.show()
