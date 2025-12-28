import os
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Wenn Index noch keine Zeitzone hat: wir nehmen an, dass er in UTC geloggt wurde
    if df.index.tz is None:
        df.index = df.index.tz_localize("utc").tz_convert("Europe/Berlin")
    else:
        # Index ist bereits tz-aware → nur noch nach Europe/Berlin umrechnen
        df.index = df.index.tz_convert("Europe/Berlin")
    df["action_in_mw"] = (df[["action_1"]] - 1) / 1
    df["date"] = df.index.date
    return df



def save_plot(plot_func, df: pd.DataFrame, file_name: str, folder: str) -> None:
    plt.figure(figsize=(12, 6))
    plot_func(df)
    plt.savefig(os.path.join(folder, file_name))
    plt.close()


def _plot_actions_by_timestep(df: pd.DataFrame) -> None:
    sns.boxplot(x="timestep", y="action_in_mw", data=df)
    plt.title("Boxplot of Actions by Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Actions")


def _plot_capacity_by_timestep(df: pd.DataFrame) -> None:
    sns.boxplot(x="timestep", y="capacity_trade", data=df)
    plt.title("Boxplot of Capacity Trade by Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Cap_trade")


def _plot_profit_by_timestep(df: pd.DataFrame) -> None:
    sns.boxplot(x="timestep", y="profit", data=df)
    plt.title("Boxplot of Profit by Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("profit")

    # Calculate the mean values for each timestep
    mean_values = df.groupby("timestep")["profit"].mean().reset_index()

    # Overlay the mean values
    sns.scatterplot(
        x="timestep",
        y="profit",
        data=mean_values,
        color="red",
        marker="D",
        s=100,
        label="Mean",
        zorder=2,
    )


def _plot_reward_by_timestep(df: pd.DataFrame) -> None:
    sns.boxplot(x="timestep", y="reward", data=df)
    plt.title("Boxplot of Reward by Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")


def _plot_enhanced_single(df: pd.DataFrame, file_name: str, folder: str) -> None:
    # Normalize variables for comparison
    df["action_1_normalized"] = (df["action_1"] - df["action_1"].min()) / (
        df["action_1"].max() - df["action_1"].min()
    )
    df["soc_t_normalized"] = (df["obs: soc_t"] - df["obs: soc_t"].min()) / (
        df["obs: soc_t"].max() - df["obs: soc_t"].min()
    )

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # First axis (Primary y-axis)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Action / SoC (Normalized)", color="b")
    ax1.plot(
        df.index,
        df["action_1_normalized"],
        label="Action 1 (Normalized)",
        color="b",
        linestyle="-",
        marker="o",
    )
    ax1.plot(
        df.index,
        df["soc_t_normalized"],
        label="SoC (Normalized)",
        color="r",
        linestyle="-",
        marker="x",
    )
    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left")

    # Second axis (Secondary y-axis for Capacity Trade)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Capacity Trade (MW)", color="orange")
    ax2.bar(
        df.index,
        df["capacity_trade"],
        color="orange",
        alpha=0.5,
        label="Capacity Trade",
        width=0.2,
    )
    ax2.tick_params(axis="y", labelcolor="orange")

    # Third axis (Secondary y-axis for EPEX Spot Price)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Move this axis outward
    ax3.set_ylabel("EPEX Spot Price (Normalized)", color="g")
    ax3.plot(
        df.index,
        df["epex_spot_60min_de_lu_eur_per_mwh"],
        label="EPEX Spot Price (Normalized)",
        color="g",
        linestyle="--",
    )
    ax3.tick_params(axis="y", labelcolor="g")

    # Fourth axis (Secondary y-axis for Reward)
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("outward", 120))  # Move this axis further outward
    ax4.set_ylabel("Reward (€)", color="purple")
    ax4.plot(
        df.index,
        df["reward"],
        label="Reward (€)",
        color="purple",
        linestyle="-.",
        marker="s",
    )
    ax4.tick_params(axis="y", labelcolor="purple")

    # Fifth axis (Secondary y-axis for Remaining Cycles)
    ax5 = ax1.twinx()
    ax5.spines["right"].set_position(("outward", 180))  # Move this axis outward
    ax5.set_ylabel("Remaining Cycles", color="cyan")
    ax5.bar(
        df.index,
        df["remaining_cycles"],
        color="cyan",
        alpha=0.4,
        label="Remaining Cycles",
        width=0.03,
    )
    ax5.tick_params(axis="y", labelcolor="cyan")

    # Title and layout
    plt.title(
        "Enhanced Overview of Actions, SoC, Capacity, EPEX Spot Price, Reward, and Remaining Cycles"
    )
    fig.tight_layout()

    # Save the figure to the folder
    plt.savefig(os.path.join(folder, file_name))
    plt.close()


def evaluate_rl_model(csv_path: str) -> None:
    report_folder = os.path.join(os.path.dirname(csv_path), "report")
    os.makedirs(report_folder, exist_ok=True)

    # Log file to store print outputs
    log_file = os.path.join(report_folder, "report_log.txt")

    with open(log_file, "w") as f:
        df = load_data(csv_path)

        # Save all plots
        save_plot(
            _plot_actions_by_timestep, df, "actions_by_timestep.png", report_folder
        )
        save_plot(
            _plot_capacity_by_timestep, df, "capacity_by_timestep.png", report_folder
        )
        save_plot(_plot_profit_by_timestep, df, "profit_by_timestep.png", report_folder)
        save_plot(_plot_reward_by_timestep, df, "reward_by_timestep.png", report_folder)

        # Group by daily data
        df_grouped = df.groupby("date")
        grouped_dfs_daily = []
        incomplete_episodes = {}

        for idx, group in df_grouped:
            if group.shape != (24, 16):
                incomplete_episodes.update({idx: group})
            temp_df = group
            temp_df.sort_index(inplace=True)
            temp_df.drop("Unnamed: 0", axis=1, inplace=True)
            temp_df["profit_cumsum"] = temp_df.profit.cumsum()
            grouped_dfs_daily.append(temp_df)

        profit_episode_dict = {}
        remaining_cycles_episode_dict = {}
        profit_date_dict = {}
        # Now use a different name, `daily_df`, in the loop to avoid overwriting the main `df`
        for daily_df in grouped_dfs_daily:
            profit_episode_dict.update(
                {daily_df.date.iloc[-1]: daily_df.profit_cumsum.iloc[-1]}
            )
            profit_date_dict.update(
                {daily_df.date.iloc[0]: daily_df.profit_cumsum.iloc[-1]}
            )
            remaining_cycles_episode_dict.update(
                ({daily_df.date.iloc[0]: daily_df.remaining_cycles.iloc[-1]})
            )

        f.write(f"days tested in date: {len(profit_date_dict)}\n")

        # Plot remaining cycles over time
        episodes = list(remaining_cycles_episode_dict.keys())
        rem_cycles = list(remaining_cycles_episode_dict.values())

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rem_cycles, "-")
        plt.title("Remaining Cycles Over Time")
        plt.xlabel("Episodes")
        plt.ylabel("Remaining Cycles")
        plt.savefig(os.path.join(report_folder, "remaining_cycles.png"))
        plt.close()

        remaining_cycles_episode_df = pd.DataFrame(
            remaining_cycles_episode_dict, index=[0]
        ).T
        remaining_cycles_episode_df.columns = ["Remaining Cycles"]

        f.write(
            str(
                remaining_cycles_episode_df[
                    remaining_cycles_episode_df["Remaining Cycles"] != 0
                ]
            )
            + "\n"
        )

        complete_cycles = remaining_cycles_episode_df[
            remaining_cycles_episode_df["Remaining Cycles"] == 0
        ].index

        # Plot revenues over time
        episodes = list(profit_episode_dict.keys())
        revenues = list(profit_episode_dict.values())

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, revenues, "-")
        plt.title("Revenues Over Time")
        plt.xlabel("Episodes")
        plt.ylabel("Revenues")
        plt.savefig(os.path.join(report_folder, "revenues_over_time.png"))
        plt.close()

        profit_date_dict = dict(
            sorted(profit_date_dict.items(), key=lambda x: x[0], reverse=False)
        )
        profit_date_df = pd.DataFrame(profit_date_dict, index=[0]).T
        profit_date_df.columns = ["Profit"]

        avg_profit = profit_date_df.loc[complete_cycles, "Profit"].mean()
        total_profit = profit_date_df.loc[complete_cycles, "Profit"].sum()

        f.write(f"avg Profit: {avg_profit}\n")
        f.write(f"total Profit: {total_profit}\n")

        # Create the enhanced plot for a specific date (e.g., November 19, 2022)
        random_sampled_day = (
            df[df.date == date(2022, 11, 19)]
            .sort_index()[
                [
                    "action_1",
                    "capacity_trade",
                    "epex_spot_60min_de_lu_eur_per_mwh",
                    "reward",
                    "profit",
                    "obs: soc_t",
                    "remaining_cycles",
                ]
            ]
            .copy()
        )
        random_sampled_day["action_1"] = (random_sampled_day["action_1"] - 1) / 1
        random_sampled_day["epex_spot_60min_de_lu_eur_per_mwh"] = (
            random_sampled_day["epex_spot_60min_de_lu_eur_per_mwh"]
            - random_sampled_day["epex_spot_60min_de_lu_eur_per_mwh"].min()
        ) / (
            random_sampled_day["epex_spot_60min_de_lu_eur_per_mwh"].max()
            - random_sampled_day["epex_spot_60min_de_lu_eur_per_mwh"].min()
        )

        _plot_enhanced_single(random_sampled_day, "sample_day.png", report_folder)

        df["month"] = df.index.month
        months_per_episode = df.groupby("date")["month"].last()
        monthly_profit = df.groupby("date")["profit"].sum()
        monthly_daily_profit = pd.concat([months_per_episode, monthly_profit], axis=1)

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=monthly_daily_profit, x="month", y="profit")
        plt.title("Monthly Profit Distribution")
        plt.savefig(os.path.join(report_folder, "monthly_profit_distribution.png"))
        plt.close()

        f.write(f"Monthly avg profit: {monthly_daily_profit.profit.mean()}\n")

        plt.figure(figsize=(10, 6))
        plt.hist(revenues, bins=13)
        plt.title("Revenue Histogram")
        plt.savefig(os.path.join(report_folder, "revenue_histogram.png"))
        plt.close()
