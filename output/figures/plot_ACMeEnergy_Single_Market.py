"""
Single Market profit and State-of-Charge (SoC) plotting script for ACM e-Energy paper.


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

# General imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Polygon

# Add parent directories to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from plotting_utils import (
    add_soc_statistics,
    align_to_date,
    colors,
    derive_soc_from_trades,
    page_height,
    page_width,
    set_acm_plot_style,
)
from results_figures_11 import derive_soc_from_trades

# Package-specific imports
from src.shared.evaluate_rl_model import load_data

if __name__ == "__main__":
    # Set ACM-compatible plot style
    set_acm_plot_style()

    # Load day-ahead results
    path_da = Path("output/single_market/day_ahead_milp/11-12.2020_ACM.csv")
    advanced_drl_da_exaa_qh = pd.read_csv(path_da)
    advanced_drl_da_exaa_qh.index = pd.to_datetime(
        advanced_drl_da_exaa_qh["time"], utc=True
    ).dt.tz_convert("Europe/Berlin")
    advanced_drl_da_exaa_qh = advanced_drl_da_exaa_qh.drop_duplicates(keep="last")
    advanced_drl_da_exaa_qh["profit"] = (
        advanced_drl_da_exaa_qh["discharge_revenues"]
        + advanced_drl_da_exaa_qh["charge_costs"]
    )

    # Aggregate daily profit
    advanced_drl_da_exaa_qh_daily = pd.DataFrame(
        advanced_drl_da_exaa_qh.groupby(advanced_drl_da_exaa_qh.index.date)[
            "profit"
        ].sum()
    )

    # Load IDC profit data
    path_idc_base = Path(
        "output/single_market/rolling_intrinsic/ri_basic/qh/2020/bs15cr1rto0.86mc365mt10"
    )
    advanced_drl_ri_qh = pd.read_csv(Path(os.path.join(path_idc_base, "profit.csv")))
    advanced_drl_ri_qh = advanced_drl_ri_qh.drop_duplicates(subset="day", keep="last")

    # Remove days excluded from RL
    exclude_dates = [
        "2020-11-01 00:00:00+01:00",
        "2020-11-15 00:00:00+01:00",
        "2020-12-27 00:00:00+01:00",
        "2020-12-31 00:00:00+01:00",
    ]
    advanced_drl_ri_qh = advanced_drl_ri_qh[
        ~advanced_drl_ri_qh["day"].isin(exclude_dates)
    ]
    dates_to_check = [
        pd.to_datetime(date).date()
        for date in ["2020-11-01", "2020-11-15", "2020-12-27", "2020-12-31"]
    ]
    advanced_drl_da_exaa_qh_daily = advanced_drl_da_exaa_qh_daily[
        ~advanced_drl_da_exaa_qh_daily.index.isin(dates_to_check)
    ]

    # Combine and sort profit data
    profits = pd.DataFrame()
    profits["IDC"] = advanced_drl_ri_qh.sort_values(by="day").reset_index()["profit"]
    profits["DA"] = advanced_drl_da_exaa_qh_daily.sort_index().reset_index()["profit"]
    profits = profits.fillna(0).sort_values(by="IDC", ascending=True)

    index_hours = range(1, len(profits) + 1)

    # Create subplots for profit and SoC
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(page_width / 1.5, page_height / 3.5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # -------------------------------
    # PROFIT DURATION CURVE
    # -------------------------------
    overall_profit_idc = profits["IDC"].sum()
    overall_profit_da = profits["DA"].sum()

    hatch_legend = [
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
    profits = profits.sort_values(by="DA", ascending=True)
    cumsum_DA = profits["DA"].cumsum()

    axs[0].plot(index_hours, cumsum_IDC, label="IDC Profit", color=colors["light_blue"])
    polygon_positive = Polygon(
        [(index_hours[0], 0)]
        + list(zip(index_hours, cumsum_IDC))
        + [(index_hours[-1], 0)],
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
    axs[0].legend(loc="lower right", bbox_to_anchor=(1, 1.02))
    axs[0].legend(
        handles=hatch_legend, loc="upper left", title="Overall Profit", frameon=False
    )
    axs[0].margins(0)
    axs[0].set_ylim(0, 5000)

    # -------------------------------
    # STATE OF CHARGE (SOC) COMPARISON
    # -------------------------------
    target_date = "2024-01-01"
    soc_id_df, soc_da_df = pd.DataFrame(), pd.DataFrame()

    for i, date in enumerate(advanced_drl_da_exaa_qh_daily.index):
        # Load trades
        path = Path(os.path.join(path_idc_base, f"trades/trades_{date}.csv"))
        trades = pd.read_csv(path, index_col=0, header=0)
        trades.index = [
            pd.to_datetime(x, utc=True).tz_convert("Europe/Berlin").timestamp()
            for x in trades.index
        ]
        trades["product"] = [
            pd.to_datetime(x, utc=True).tz_convert("Europe/Berlin").timestamp()
            for x in trades["product"]
        ]

        # Load VWAPs and align timestamps
        path_vwaps = Path(os.path.join(path_idc_base, f"vwap/vwaps_{date}.csv"))
        vwaps = pd.read_csv(path_vwaps, index_col=0, header=0)
        date_vwaps = pd.to_datetime(vwaps.index[0]).date()
        vwaps.columns = pd.date_range(date_vwaps, periods=96, freq="15T")
        vwaps.index = [
            pd.to_datetime(x, utc=True).tz_convert("Europe/Berlin").timestamp() - 900
            for x in vwaps.index
        ]
        vwaps.columns = [
            pd.to_datetime(x, utc=True).tz_convert("Europe/Berlin").timestamp()
            for x in vwaps.columns
        ]
        vwaps = vwaps.T.sort_index()
        products = vwaps.columns

        # Compute and align SoC for IDC
        soc_id = derive_soc_from_trades(trades, 1, products[0], products[-1])["soc"]
        soc_id = pd.DataFrame(soc_id / 4 * 100).iloc[32:]
        soc_id.index = [
            pd.to_datetime(x, unit="s", utc=True).tz_convert("Europe/Berlin")
            for x in soc_id.index
        ]
        soc_id = align_to_date(soc_id, target_date)
        soc_id_df[f"soc_{i}"] = soc_id

        # Compute and align SoC for DA
        soc_da = pd.DataFrame(
            advanced_drl_da_exaa_qh["soc"][advanced_drl_da_exaa_qh.index.date == date]
            * 100
        )
        soc_da = align_to_date(soc_da, target_date)
        soc_da_df[f"soc_{i}"] = soc_da

    print(add_soc_statistics(soc_id_df))
    print(add_soc_statistics(soc_da_df))

    # Plot SoC comparison
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

    axs[1].set_xlabel(r"State of Charge in %")
    axs[1].set_xticks(np.arange(0, 101, 50))
    axs[1].set_xlim(0, 100)

    date_range = pd.date_range(
        soc_da_df.index[0], soc_da_df.index[-1] + pd.Timedelta(minutes=15), freq="2H"
    )
    axs[1].set_yticks(date_range)
    axs[1].set_yticklabels(date_range.strftime("%H:%M"))
    axs[1].set_ylim(date_range[0], date_range[-1])

    # Clean up legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[1].legend(
        by_label.values(), by_label.keys(), loc="lower right", bbox_to_anchor=(1, 1.02)
    )

    plt.tight_layout()
    plt.savefig("output/figures/profit_plot_Single_Market.png")
    plt.savefig("output/figures/profit_plot_Single_Market.pdf", format="pdf")
    plt.show()
