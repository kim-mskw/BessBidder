import os
import warnings
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import gurobipy as gp
from loguru import logger

# Suppress noisy warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Default path for precomputed VWAP matrices
from src.shared.config import PRECOMPUTED_VWAP_PATH



def adjust_prices_block(
    prices_qh: pd.DataFrame,
    execution_time: pd.Timestamp,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Create two vectorized, discounted price columns based on time-to-delivery
    and a continuous discount rate:

    - price_sell_adj: effective price for selling
    - price_buy_adj:  effective price for buying (inverse discounting)

    Rules:
    - Base price column: 'price' (rounded to 2 decimals).
    - Discount factor depends on hours between execution_time and delivery.
    - If absolute time to delivery <= 1 hour: use original price.
    - Negative prices are discounted with inverse sign in exponent
      (to preserve economics of negative pricing).
    """
    out = prices_qh.copy()

    if "price" not in out.columns:
        raise ValueError("prices_qh must contain a 'price' column.")
    out["price"] = out["price"].round(2)

    # Compute time difference (in hours) between execution_time and each product
    idx = out.index.to_numpy(dtype="datetime64[ns]")
    exec_ts = np.datetime64(execution_time.tz_convert(None), "ns")
    hours = (idx - exec_ts) / np.timedelta64(1, "h")  # ndarray[float]

    price = out["price"].to_numpy(dtype=float)
    is_nan = np.isnan(price)

    # Sign rule so negative prices are handled correctly:
    # price < 0 => +1, else -1 in the exponent
    sign = np.where(price < 0, +1.0, -1.0)

    # Continuous discount factor per row
    factor = np.exp((discount_rate / 100.0) * sign * hours)

    # For deliveries within 1 hour: keep original price
    use_orig = hours <= 1.0

    # Sell side: multiply by factor (except within 1 hour)
    price_sell_adj = np.where(use_orig, price, price * factor)

    # Buy side: divide by factor (except within 1 hour)
    price_buy_adj = np.where(use_orig, price, price / factor)

    # Preserve NaNs
    price_sell_adj = np.where(is_nan, np.nan, price_sell_adj)
    price_buy_adj = np.where(is_nan, np.nan, price_buy_adj)

    out["price_sell_adj"] = np.round(price_sell_adj, 2)
    out["price_buy_adj"] = np.round(price_buy_adj, 2)
    return out


def get_net_trades(trades: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Aggregate trades to a full delivery day in 15-minute resolution.

    Returns a DataFrame (index: all quarter-hours of the delivery day) with:
    - sum_buy
    - sum_sell
    - net_buy  (max(sum_buy - sum_sell, 0))
    - net_sell (max(sum_sell - sum_buy, 0))
    """
    # Build full 24h quarter-hour index for the delivery day
    start = end_date - pd.Timedelta(hours=2)
    start = start.replace(hour=0, minute=0)
    end = start.replace(hour=23, minute=45)
    idx = pd.date_range(start, end, freq="15min")

    # Case 1: no trades -> all zeros
    if trades.empty:
        return pd.DataFrame(
            0.0,
            index=idx,
            columns=["sum_buy", "sum_sell", "net_buy", "net_sell"],
        )

    # Case 2: aggregate trades by product and side
    grouped = trades.groupby(["product", "side"])["quantity"].sum().unstack(fill_value=0)

    # Safe access for buy/sell columns
    grouped["sum_buy"] = grouped.get("buy", 0.0)
    grouped["sum_sell"] = grouped.get("sell", 0.0)

    grouped["net_buy"] = grouped["sum_buy"] - grouped["sum_sell"]
    grouped["net_sell"] = grouped["sum_sell"] - grouped["sum_buy"]

    # No negative net volumes
    grouped["net_buy"] = grouped["net_buy"].clip(lower=0.0)
    grouped["net_sell"] = grouped["net_sell"].clip(lower=0.0)

    grouped = grouped[["sum_buy", "sum_sell", "net_buy", "net_sell"]]

    # Reindex to full day, fill missing with 0
    return grouped.reindex(idx, fill_value=0.0)


def load_vwaps_for_day(
    current_day: pd.Timestamp,
    vwaps_base_path: str = PRECOMPUTED_VWAP_PATH,
) -> pd.DataFrame:
    """
    Load precomputed VWAP matrix for a given delivery day.

    - Index: execution_time_end (bucket end times)
    - Columns: delivery start times (product quarter-hours)
    Both in Europe/Berlin timezone.
    """
    fname = os.path.join(vwaps_base_path, f"vwaps_{current_day:%Y-%m-%d}.parquet")

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"VWAP parquet file for {current_day:%Y-%m-%d} not found: {fname}"
        )

    matrix = pd.read_parquet(fname)

    # Index: execution_time_end (bucket end)
    matrix.index = pd.to_datetime(matrix.index, utc=True).tz_convert("Europe/Berlin")

    # Columns: delivery start times
    matrix.columns = pd.to_datetime(matrix.columns, utc=True).tz_convert(
        "Europe/Berlin"
    )

    return matrix


def infer_bucket_size_minutes(vwaps_day: pd.DataFrame) -> int:
    """
    Infer bucket size (in minutes) from the VWAP matrix index spacing.
    """
    idx = vwaps_day.index.sort_values()
    if len(idx) < 2:
        raise ValueError("VWAP matrix has <2 rows; cannot infer bucket size.")

    deltas = idx.to_series().diff().dropna()
    # take the most frequent delta to be robust against missing buckets
    most_common = deltas.value_counts().idxmax()
    minutes = int(most_common.total_seconds() / 60)

    if minutes <= 0:
        raise ValueError(f"Inferred invalid bucket size: {minutes} minutes")

    return minutes


def get_vwap_from_precomputed(
    vwaps_day: pd.DataFrame,
    execution_time_end: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Extract one VWAP row from the daily VWAP matrix and convert it into a
    quarter-hourly price curve for the full delivery day.

    Returns a DataFrame with:
    - index: all quarter-hours of the delivery day
    - column: 'price'
    """
    end_date = end_date.tz_convert("Europe/Berlin")

    start_of_day = end_date - pd.Timedelta(hours=2)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day = start_of_day.replace(hour=23, minute=45)

    product_index = pd.date_range(
        start_of_day, end_of_day, freq="15min", tz="Europe/Berlin"
    )

    if execution_time_end not in vwaps_day.index:
        return pd.DataFrame(index=product_index, columns=["price"], dtype=float)

    row = vwaps_day.loc[execution_time_end]  # Series

    vwap = row.to_frame(name="price")
    vwap = vwap.reindex(product_index)
    return vwap


def build_battery_model(
    T: List[pd.Timestamp],
    cap: float,
    c_rate: float,
    roundtrip_eff: float,
) -> Tuple[gp.Model, Dict[str, Any], Dict[Any, gp.Constr], gp.Constr]:
    """
    Build a persistent Gurobi model for a single battery over a full delivery day.

    Variables (indexed by quarter-hour in T):
    - current_buy_qh, current_sell_qh: buy/sell volumes in each QH
    - battery_soc: state of charge
    - net_buy, net_sell: true charging/discharging flows
    - charge_sign: binary flag to prevent simultaneous charge+discharge
    - z, w: auxiliary variables for piecewise linear / big-M logic

    The model is built once and reused for each execution bucket by
    adjusting RHS and the objective.
    """
    efficiency = roundtrip_eff ** 0.5
    M = cap * c_rate

    m = gp.Model("battery_persistent")
    m.Params.OutputFlag = 0

    current_buy_qh = m.addVars(T, lb=0.0, name="current_buy_qh")
    current_sell_qh = m.addVars(T, lb=0.0, name="current_sell_qh")
    battery_soc = m.addVars(T, lb=0.0, name="battery_soc")

    net_buy = m.addVars(T, lb=0.0, name="net_buy")
    net_sell = m.addVars(T, lb=0.0, name="net_sell")
    charge_sign = m.addVars(T, vtype=gp.GRB.BINARY, name="charge_sign")

    z = m.addVars(T, lb=0.0, name="z")
    w = m.addVars(T, lb=0.0, name="w")

    # SOC dynamics over time
    prev = T[0]
    for i in T[1:]:
        m.addConstr(
            battery_soc[i]
            == battery_soc[prev]
            + net_buy[prev] * efficiency / 4.0
            - net_sell[prev] / 4.0 / efficiency,
            name=f"BatteryBalance_{i}",
        )
        prev = i

    # Start with empty battery
    m.addConstr(battery_soc[T[0]] == 0.0, name="InitialBatterySOC")

    # Time-independent constraints
    for i in T:
        # Capacity and power limits
        m.addConstr(battery_soc[i] <= cap, name=f"Cap_{i}")
        m.addConstr(net_buy[i] <= cap * c_rate, name=f"BuyRate_{i}")
        m.addConstr(net_sell[i] <= cap * c_rate, name=f"SellRate_{i}")

        # Cannot discharge more than current SOC
        m.addConstr(
            net_sell[i] / efficiency / 4.0 <= battery_soc[i],
            name=f"SellVsSOC_{i}",
        )

        # Big-M constraints to prevent simultaneous buy/sell
        m.addConstr(net_buy[i] <= M * charge_sign[i], name=f"NetBuyBigM_{i}")
        m.addConstr(
            net_sell[i] <= M * (1 - charge_sign[i]), name=f"NetSellBigM_{i}"
        )

        # Auxiliary variable z for charging
        m.addConstr(z[i] <= charge_sign[i] * M, name=f"ZUpper_{i}")
        m.addConstr(z[i] <= net_buy[i], name=f"ZNetBuy_{i}")
        m.addConstr(
            z[i] >= net_buy[i] - (1 - charge_sign[i]) * M, name=f"ZLower_{i}"
        )
        m.addConstr(z[i] >= 0.0, name=f"ZNonNeg_{i}")

        # Auxiliary variable w for discharging
        m.addConstr(w[i] <= (1 - charge_sign[i]) * M, name=f"WUpper_{i}")
        m.addConstr(w[i] <= net_sell[i], name=f"WNetSell_{i}")
        m.addConstr(
            w[i] >= net_sell[i] - charge_sign[i] * M, name=f"WLower_{i}"
        )
        m.addConstr(w[i] >= 0.0, name=f"WNonNeg_{i}")

    # Netting constraints:
    # z[i] - w[i] - current_buy_qh[i] + current_sell_qh[i] = RHS
    # RHS will be updated per bucket (prev_net_buy - prev_net_sell)
    netting_constr: Dict[Any, gp.Constr] = {}
    for i in T:
        c = m.addConstr(
            z[i] - w[i] - current_buy_qh[i] + current_sell_qh[i] == 0.0,
            name=f"Netting_{i}",
        )
        netting_constr[i] = c

    # Cycle constraint: sum of charged energy <= allowed_cycles * cap
    # RHS will be updated per bucket via max_cycles_constr.RHS
    max_cycles_constr = m.addConstr(
        gp.quicksum(net_buy[i] * efficiency / 4.0 for i in T) <= 0.0,
        name="MaxCycles",
    )

    m.setObjective(0.0, gp.GRB.MAXIMIZE)
    m.update()

    vars_dict = {
        "current_buy_qh": current_buy_qh,
        "current_sell_qh": current_sell_qh,
        "battery_soc": battery_soc,
        "net_buy": net_buy,
        "net_sell": net_sell,
        "charge_sign": charge_sign,
        "z": z,
        "w": w,
        "efficiency": efficiency,
    }

    return m, vars_dict, netting_constr, max_cycles_constr


def solve_bucket_with_persistent_model(
    m: gp.Model,
    vars: Dict[str, Any],
    netting_constr: Dict[Any, gp.Constr],
    max_cycles_constr: gp.Constr,
    prices_qh: pd.DataFrame,
    execution_time: pd.Timestamp,
    discount_rate: float,
    prev_net_trades: pd.DataFrame,
    allowed_cycles: float,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, float]:
    """
    Solve one intraday bucket using a persistent battery model.

    - Updates RHS of netting and cycle constraints based on previous trades.
    - Rebuilds the objective from discounted prices.
    - Returns:
        results: quarter-hourly decision variables
        trades:  realized trades in this bucket
        objval:  objective value (profit)
      If no optimal solution is found, returns (None, empty_trades, 0.0).
    """
    T = list(prices_qh.index)
    current_buy_qh = vars["current_buy_qh"]
    current_sell_qh = vars["current_sell_qh"]
    net_buy = vars["net_buy"]
    net_sell = vars["net_sell"]
    charge_sign = vars["charge_sign"]

    # 1) Prepare discounted prices
    prices_qh_adj_all = adjust_prices_block(prices_qh, execution_time, discount_rate)
    prev_net_trades = prev_net_trades.reindex(prices_qh.index).fillna(0.0)

    # Small spread/tolerance parameter
    eps = 0.01

    # 2) Set RHS of cycle constraint
    # (Note: cap=1 in calling code; multiply by cap if needed)
    max_cycles_constr.RHS = allowed_cycles * 1.0

    # 3) Update netting RHS and variable bounds for NaNs
    for i in T:
        prev_nb = prev_net_trades.loc[i, "net_buy"]
        prev_ns = prev_net_trades.loc[i, "net_sell"]

        # RHS = prev_nb - prev_ns (net position carried into this bucket)
        netting_constr[i].RHS = float(prev_nb - prev_ns)

        price = prices_qh.loc[i, "price"]
        if pd.isna(price):
            # Disable trading for missing price
            current_buy_qh[i].UB = 0.0
            current_sell_qh[i].UB = 0.0
        else:
            # Allow trading without an explicit upper bound here
            current_buy_qh[i].UB = gp.GRB.INFINITY
            current_sell_qh[i].UB = gp.GRB.INFINITY

    # 4) Rebuild objective function
    obj = gp.LinExpr()
    for i in T:
        price = prices_qh.loc[i, "price"]
        if pd.isna(price):
            continue

        prev_nb = prev_net_trades.loc[i, "net_buy"]
        prev_ns = prev_net_trades.loc[i, "net_sell"]

        price_sell_adj = prices_qh_adj_all.loc[i, "price_sell_adj"]
        price_buy_adj = prices_qh_adj_all.loc[i, "price_buy_adj"]

        # Slightly different spread if there was no previous position
        if prev_nb < eps and prev_ns < eps:
            term = (
                current_sell_qh[i] * (price_sell_adj - 0.1 / 2 - eps)
                - current_buy_qh[i] * (price_buy_adj + 0.1 / 2 + eps)
            ) / 4.0
        else:
            term = (
                current_sell_qh[i] * (price_sell_adj - eps)
                - current_buy_qh[i] * (price_buy_adj + eps)
            ) / 4.0

        obj += term

    m.setObjective(obj, gp.GRB.MAXIMIZE)

    # 5) Solve
    m.optimize()

    if m.status != gp.GRB.OPTIMAL:
        # No optimal solution found -> return empty trades
        logger.warning("No optimal solution found for current bucket.")
        empty_trades = pd.DataFrame(
            columns=["execution_time", "side", "quantity", "price", "product", "profit"]
        )
        return None, empty_trades, 0.0

    # 6) Collect results
    results = pd.DataFrame(
        index=prices_qh.index,
        columns=[
            "current_buy_qh",
            "current_sell_qh",
            "net_buy",
            "net_sell",
            "charge_sign",
            "battery_soc",
        ],
    )

    trade_rows = []
    for i in T:
        cb = current_buy_qh[i].X
        cs = current_sell_qh[i].X

        if cb is not None and cb > 0:
            trade_rows.append(
                (
                    execution_time,
                    "buy",
                    cb,
                    prices_qh.loc[i, "price"],
                    i,
                    -cb * prices_qh.loc[i, "price"] / 4.0,
                )
            )
        if cs is not None and cs > 0:
            trade_rows.append(
                (
                    execution_time,
                    "sell",
                    cs,
                    prices_qh.loc[i, "price"],
                    i,
                    cs * prices_qh.loc[i, "price"] / 4.0,
                )
            )

        results.loc[i, "current_buy_qh"] = cb
        results.loc[i, "current_sell_qh"] = cs
        results.loc[i, "net_buy"] = net_buy[i].X
        results.loc[i, "net_sell"] = net_sell[i].X
        results.loc[i, "charge_sign"] = charge_sign[i].X
        results.loc[i, "battery_soc"] = vars["battery_soc"][i].X

    trades = pd.DataFrame(
        trade_rows,
        columns=["execution_time", "side", "quantity", "price", "product", "profit"],
    )

    return results, trades, m.ObjVal


def simulate_days_stacked_quarterhourly_products(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    discount_rate: float,
    #bucket_size: int,
    c_rate: float,
    roundtrip_eff: float,
    max_cycles: float,
    min_trades: int,
    vwaps_base_path: str = PRECOMPUTED_VWAP_PATH,
    drl_output: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Main simulation function for intraday trading on quarter-hourly products.

    Workflow per delivery day:
    - Build the trading horizon and battery model (persistent Gurobi model).
    - Load precomputed VWAP matrices.
    - Optionally include day-ahead DRL trades.
    - Loop over execution buckets (bucket_size minutes):
        - Extract VWAP curve for this bucket.
        - Aggregate existing trades -> net positions.
        - Solve a battery optimization bucket.
    - Build a quarter-hourly reporting DataFrame with:
        'net_quantity', 'vwap', 'total_profit'.

    Note: Currently simulates a single delivery day (start_day).
    """
    log_message = (
        "Running rolling intrinsic QH simulation with parameters:\n"
        f"Start Day: {start_day}\n"
        f"End Day: {end_day}\n"
        f"Discount Rate: {discount_rate}\n"
        #f"Bucket Size: {bucket_size}\n"
        f"C Rate: {c_rate}\n"
        f"Roundtrip Efficiency: {roundtrip_eff}\n"
        f"Max Cycles: {max_cycles}\n"
        f"Min Trades: {min_trades}"
    )
    logger.info(log_message)

    current_day = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
    logger.info(f"Current delivery day: {current_day}")

    all_trades = pd.DataFrame(
        columns=["execution_time", "side", "quantity", "price", "product", "profit"]
    )

    # Trading window: from 8h before delivery day to end of delivery day
    trading_start = current_day - pd.Timedelta(hours=8)
    trading_end = current_day + pd.Timedelta(days=1)

    logger.info(f"Trading start: {trading_start}")
    logger.info(f"Trading end:   {trading_end}")

    # Day-ahead trades (optional, from DRL output)
    gate_closure_day_ahead = (
        current_day - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
    )
    day_ahead_trades_drl = None

    if drl_output is not None:
        df_da = drl_output.copy()

        if "product" not in df_da.columns:
            logger.warning(
                f"drl_output has no 'product' column for {current_day:%Y-%m-%d} – "
                "skipping day-ahead trades."
            )
        else:
            # Ensure 'product' is a Timestamp
            if not np.issubdtype(df_da["product"].dtype, np.datetime64):
                df_da["product"] = pd.to_datetime(df_da["product"])

            # Only trades for the current delivery day
            mask = df_da["product"].dt.date == current_day.date()
            day_ahead_trades_drl = df_da.loc[mask].copy()

            if day_ahead_trades_drl.empty:
                logger.warning(
                    f"No DRL day-ahead trades found for {current_day:%Y-%m-%d} – "
                    "skipping day-ahead."
                )
                day_ahead_trades_drl = None
            else:
                logger.info(
                    f"Day-ahead trades for {current_day:%Y-%m-%d}: "
                    f"{len(day_ahead_trades_drl)} rows"
                )
                all_trades = pd.concat(
                    [all_trades, day_ahead_trades_drl], ignore_index=True
                )

    # Build Gurobi battery model
    start_of_day = trading_end - pd.Timedelta(hours=2)
    start_of_day = start_of_day.replace(hour=0, minute=0)
    end_of_day = start_of_day.replace(hour=23, minute=45)

    T = list(
        pd.date_range(
            start_of_day, end_of_day, freq="15min", tz="Europe/Berlin"
        )
    )

    m, vars_dict, netting_constr, max_cycles_constr = build_battery_model(
        T, cap=1, c_rate=c_rate, roundtrip_eff=roundtrip_eff
    )

    # Load precomputed VWAP matrix for the day
    vwaps_day = load_vwaps_for_day(current_day, vwaps_base_path)

    bucket_size = infer_bucket_size_minutes(vwaps_day)

    execution_time_start = trading_start
    execution_time_end = trading_start + pd.Timedelta(minutes=bucket_size)

    allowed_cycles = max_cycles

    # Intraday simulation loop over execution buckets
    while execution_time_end < trading_end:
        vwap = get_vwap_from_precomputed(
            vwaps_day,
            execution_time_end,
            trading_end,
        )

        net_trades = get_net_trades(all_trades, trading_end)

        # If bucket has no prices at all, skip to next bucket
        if vwap["price"].isnull().all():
            execution_time_start = execution_time_end
            execution_time_end = (
                execution_time_start + pd.Timedelta(minutes=bucket_size)
            )
            continue

        try:
            _, trades, profit = solve_bucket_with_persistent_model(
                m,
                vars_dict,
                netting_constr,
                max_cycles_constr,
                vwap,
                execution_time_start,
                discount_rate,
                net_trades,
                allowed_cycles,
            )
            all_trades = pd.concat([all_trades, trades])
        except ValueError as e:
            logger.error(
                f"Error in optimization at execution_time_start={execution_time_start}: {e}"
            )

        execution_time_start = execution_time_end
        execution_time_end = (
            execution_time_start + pd.Timedelta(minutes=bucket_size)
        )

    # Reporting: optionally drop day-ahead trades (if DRL DA exists)
    if day_ahead_trades_drl is not None:
        all_trades_reporting = all_trades[
            all_trades["execution_time"] != gate_closure_day_ahead
        ]
    else:
        all_trades_reporting = all_trades

    reporting = create_quarterhourly_reporting(
        all_trades=all_trades_reporting,
        start_day=current_day,
    )

    return reporting


def create_quarterhourly_reporting(
    all_trades: pd.DataFrame, start_day: pd.Timestamp
) -> pd.DataFrame:
    """
    Create a quarter-hourly reporting DataFrame with:

    - net_quantity: net position per QH (MWh)
    - vwap:         volume-weighted average price per QH
    - total_profit: sum of profits per QH

    Missing products are filled with zeros, and the index covers the
    full delivery day in 15-minute resolution.
    """
    df = all_trades.copy()

    if df.empty:
        complete_index = pd.date_range(
            start_day,
            start_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
            freq="15min",
        )
        complete_df = pd.DataFrame(
            index=complete_index,
            columns=["net_quantity", "vwap", "total_profit"],
        )
        complete_df.index.name = "product"
        complete_df.fillna(0, inplace=True)
        return complete_df

    # Net quantity: buys are negative, sells positive
    df["net_quantity"] = df.apply(
        lambda x: -x["quantity"] if x["side"] == "buy" else x["quantity"],
        axis=1,
    )
    net_quantity = df.groupby("product")["net_quantity"].sum() / 4

    # VWAP: price * volume, aggregated and normalized
    df["vwap"] = df["price"] * df["quantity"] / 4
    vwap = df.groupby("product").apply(
        lambda x: x["vwap"].sum() / x["quantity"].sum() / 4
    )

    # Total profit per product
    total_profit = df.groupby("product")["profit"].sum()

    summary = pd.DataFrame(
        {
            "net_quantity": net_quantity,
            "vwap": vwap,
            "total_profit": total_profit,
        }
    )

    # Reindex to full 24h quarter-hourly delivery day
    summary = summary.reindex(
        index=pd.date_range(
            start_day,
            start_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
            freq="15min",
        )
    )
    summary.index.name = "product"
    summary.fillna(0, inplace=True)

    return summary