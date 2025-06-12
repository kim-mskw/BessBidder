import os
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from src.shared.config import TEST_CSV_NAME, TRAIN_CSV_NAME

SEED = 42
np.random.seed(SEED)

PERIOD_LENGTH = 24
# EPEX_DA_MIN_PRICE = np.float32(-500.0)
# EPEX_DA_MAX_PRICE = np.float32(4000.0)
PENALTY_SOC_LIMITS = 0.5


class BasicBatteryDAM(gym.Env):
    def __init__(
        self,
        modus: str,
        logging_path: str,
        input_data: dict[str, dict[str, np.array]],
        power: np.float32 = 1.0,
        capacity: np.float32 = 1.0,
        round_trip_efficiency: np.float32 = 1.0,
        start_end_soc: np.float32 = 0.0,
    ):
        self._episode_id = 0
        self._modus = modus
        self._logging_path = logging_path
        self._input_data = input_data
        self._days_left = np.array(list(self._input_data.keys()), dtype=str)

        self._reinitialize_input_data_after_reset()

        self._power = power
        self._capacity = capacity
        self._efficiency = round_trip_efficiency**0.5
        # TODO: implement start end restriction for the storage
        self._start_end_soc = start_end_soc
        self._current_soc = self._start_end_soc
        self._remaining_cycles = 1
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(-1, 1, shape=(42,), dtype=np.float32)
        self._current_time_step = 0
        self._realized_quantity_t_minus_1 = 0
        self._total_profit = 0.0

    def _get_obs(self):
        # Calculate sine and cosine of the current time step
        sin_time_step = np.sin(2 * np.pi * self._current_time_step / 24)
        cos_time_step = np.cos(2 * np.pi * self._current_time_step / 24)

        # Calculate sine and cosine of month
        sin_month = np.sin(2 * np.pi * self._date_month / 12)
        cos_month = np.sin(2 * np.pi * self._date_month / 12)

        return np.concatenate(
            (
                self._realized_quantity_t_minus_1,
                self._current_soc,
                self._remaining_cycles,
                # encoded time
                sin_time_step,
                cos_time_step,
                # hourly forecasts
                self._residual_load_forecast_scaled[self._current_time_step],
                self._forecasted_price_vector_scaled,
                # include avergae change per hour in forecasts
                self._delta_load_forecast[self._current_time_step],
                self._delta_pv_forecast_scaled[self._current_time_step],
                self._delta_wind_onshore_forecast_scaled[self._current_time_step],
                sin_month,
                cos_month,
                self._day_of_week,
                # get daily RE statistics
                self._wind_forecast_daily_mean,
                self._wind_forecast_daily_std,
                self._spread_id_full_da_mean,
                self._spread_id_full_da_std,
                self._spread_id_full_da_min,
                self._spread_id_full_da_max,
            ),
            axis=None,
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._reinitialize_input_data_after_reset()
        self._current_time_step = 0
        self._realized_quantity_t_minus_1 = 0
        self._current_soc = self._start_end_soc
        self._remaining_cycles = 1
        self._total_profit = 0.0
        observation = self._get_obs()
        return observation, {}  # empty info dict

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """

        :param action:
        :return:
        """
        action_continuous = self._map_discrete_action_to_continuous(action)
        quantity = np.clip(
            action_continuous, -1.0, 1.0
        )  # Ensure it's within the valid range

        clearing_price = self._realized_price_vector[self._current_time_step]
        clearing_price_scaled = self._realized_price_vector_scaled[
            self._current_time_step
        ]

        if quantity > 0:
            sell_decision = min(self._current_soc, quantity, 2 * self._remaining_cycles)
            realized_quantity = sell_decision
        elif quantity < 0:
            # buy_decision = quantity
            buy_decision = min(
                self._capacity - self._current_soc,
                (-1) * quantity,
                2 * self._remaining_cycles,
            )
            realized_quantity = (-1) * buy_decision
        else:
            realized_quantity = 0

        reward = clearing_price * realized_quantity / (85 / 24)

        delta_soc = 0
        delta_soc = (-1) * realized_quantity

        profit = 0
        if realized_quantity < 0:
            profit = clearing_price * realized_quantity * (1 / self._efficiency)
        elif realized_quantity > 0:
            profit = clearing_price * realized_quantity * self._efficiency

        self._current_soc += delta_soc
        delta_cycles = abs(delta_soc) / (2 * self._capacity)
        self._remaining_cycles -= delta_cycles

        self._realized_quantity_t_minus_1 = realized_quantity

        game_over = False
        if round(self._remaining_cycles, 2) <= 0:
            game_over = True

        self._total_profit += profit

        info = self._get_info()
        observation = self._get_obs()

        self._last_time_step = self._current_time_step
        self._current_time_step += 1
        # terminated = True if (self._current_time_step == PERIOD_LENGTH) or game_over else False
        terminated = (
            True if (self._current_time_step == PERIOD_LENGTH) or game_over else False
        )

        if terminated:
            self._episode_id += 1

            # check if we have capacity left in battery
            if self._current_soc > 0:
                # penalty because missed profit
                penalty = self._current_soc * (85 / 24)
                reward = -penalty

        self.log_data(
            modus=self._modus,
            timestamp=self._timestamps[self._last_time_step],
            episode_id=self._episode_id,
            timestep=self._last_time_step,
            observations=observation,
            action=action,
            reward=reward,
            dam_price_forecast=self._forecasted_price_vector[self._last_time_step],
            dam_price=clearing_price,
            price_bid=np.nan,
            capacity_bid=quantity,
            capacity_trade=realized_quantity,
            delta_soc=delta_soc,
            remaining_cycles=self._remaining_cycles,
            # remaining_cycles=np.nan,
            profit=profit,
        )

        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

    def _get_info(self):
        return {
            "timestamp": self._timestamps[self._current_time_step].astype("int64"),
            "position": self._realized_quantity_t_minus_1,
            "clearing_price": self._realized_price_vector[self._current_time_step],
            "scaling_max_price": self._max_price_realized,
            "scaling_min_price": self._min_price_realized,
        }

    def close(self):
        pass

    def _map_discrete_action_to_continuous(self, action: int) -> float:
        # Map discrete action index (0 to 6) to continuous value (-1.0 to 1.0)
        if action <= 4:
            return round(-1 + (action / 3), 2)
        else:
            return round((action - 3) / 3, 2)

    @staticmethod
    def calculate_soc_penalty(current_soc: float) -> float:
        """
        Calculate penalty based on deviation from SOC limits (quadratic penalty).

        Parameters:
            current_soc (float): Current state of charge (SOC) of the battery.

        Returns:
            float: Penalty value.
        """
        if current_soc < 0.0:
            penalty = (-1) * current_soc
        elif current_soc > 1.0:
            penalty = current_soc - 1
        else:
            penalty = 0.0

        return penalty

    @staticmethod
    def _map_action_to_obs_space(
        action_normalized: np.float32, min_value: np.float32, max_value: np.float32
    ) -> np.float32:
        """
        Maps the agent's standardized action to a value within the specified range.

        Parameters:
            action_normalized (float): The standardized action taken by the agent, ranging from -1 to 1.
            min_value (float): The minimum value in the desired range.
            max_value (float): The maximum value in the desired range.

        Returns:
            float: The corresponding observation within the specified range.
        """
        return (action_normalized + 1) * (max_value - min_value) / 2 + min_value

    def _sample_random_day(self) -> str:
        return str(np.random.choice(self._days_left, 1, replace=False)[0])

    def _reinitialize_input_data_after_reset(self) -> None:
        self._random_day = self._sample_random_day()
        self._forecasted_price_vector = self._input_data[self._random_day][
            "price_forecast"
        ]
        self._realized_price_vector = self._input_data[self._random_day][
            "price_realized"
        ]
        self._min_price_realized = np.min(self._realized_price_vector)
        self._max_price_realized = np.max(self._realized_price_vector)
        self._min_price_forecasted = np.min(self._forecasted_price_vector)
        self._max_price_forecasted = np.max(self._forecasted_price_vector)
        self._forecasted_price_vector_scaled = (
            self._input_data[self._random_day]["price_forecast"]
            - self._min_price_forecasted
        ) / (self._max_price_forecasted - self._min_price_forecasted)
        self._realized_price_vector_scaled = (
            self._input_data[self._random_day]["price_realized"]
            - self._min_price_realized
        ) / (self._max_price_realized - self._min_price_realized)
        self._date_month = self._input_data[self._random_day]["date_month"][0]
        self._day_of_week = self._input_data[self._random_day]["day_of_week"][0]
        self._wind_forecast_daily_mean = self._input_data[self._random_day][
            "wind_forecast_daily_mean"
        ][0]
        self._wind_forecast_daily_std = self._input_data[self._random_day][
            "wind_forecast_daily_std"
        ][0]
        self._spread_id_full_da_mean = self._input_data[self._random_day][
            "spread_id_full_da_mean"
        ][0]
        self._spread_id_full_da_std = self._input_data[self._random_day][
            "spread_id_full_da_std"
        ][0]
        self._spread_id_full_da_min = self._input_data[self._random_day][
            "spread_id_full_da_min"
        ][0]
        self._spread_id_full_da_max = self._input_data[self._random_day][
            "spread_id_full_da_max"
        ][0]

        self._timestamps = self._input_data[self._random_day]["timestamps"]

        self._pv_forecast = self._input_data[self._random_day][
            "pv_forecast_d_minus_1_1000_de_lu_mw"
        ]

        self._wind_onshore_forecast = self._input_data[self._random_day][
            "wind_onshore_forecast_d_minus_1_1000_de_lu_mw"
        ]
        self._load_forecast = self._input_data[self._random_day][
            "load_forecast_d_minus_1_1000_total_de_lu_mw"
        ]

        self._residual_load_forecast = (
            self._load_forecast
            - self._wind_onshore_forecast
            - self._pv_forecast
            - self._pv_forecast
        )

        self._min_residual_load_forecast = np.min(self._residual_load_forecast)
        self._max_residual_load_forecast = np.max(self._residual_load_forecast)

        denominator = (
            self._max_residual_load_forecast - self._min_residual_load_forecast
        )
        if denominator != 0:
            self._residual_load_forecast_scaled = (
                self._residual_load_forecast - self._min_residual_load_forecast
            ) / denominator
        else:
            self._residual_load_forecast_scaled = np.zeros(24)

        self._min_pv_forecast = np.min(self._pv_forecast)
        self._max_pv_forecast = np.max(self._pv_forecast)

        denominator = self._max_pv_forecast - self._min_pv_forecast
        if denominator != 0:
            self._pv_forecast_scaled = (
                self._pv_forecast - self._min_pv_forecast
            ) / denominator
        else:
            self._pv_forecast_scaled = np.zeros(24)

        self._pv_forecast_daily_mean = np.mean(self._pv_forecast)
        self._pv_forecast_daily_std = np.std(self._pv_forecast)

        self._min_wind_onshore_forecast = np.min(self._wind_onshore_forecast)
        self._max_wind_onshore_forecast = np.max(self._wind_onshore_forecast)

        denominator = self._max_wind_onshore_forecast - self._min_wind_onshore_forecast
        if denominator != 0:
            self._wind_onshore_forecast_scaled = (
                self._wind_onshore_forecast - self._min_wind_onshore_forecast
            ) / denominator
        else:
            self._wind_onshore_forecast_scaled = np.zeros(24)

        self._min_load_forecast = np.min(self._load_forecast)
        self._max_load_forecast = np.max(self._load_forecast)

        denominator = self._max_load_forecast - self._min_load_forecast
        if denominator != 0:
            self._load_forecast_scaled = (
                self._load_forecast - self._min_load_forecast
            ) / denominator
        else:
            self._load_forecast_scaled = np.zeros(24)

        self._load_forecast_daily_mean = np.mean(self._load_forecast)
        self._load_forecast_daily_std = np.std(self._load_forecast)

        # calculate load gradient of forecast
        self._delta_load_forecast = np.append(np.diff(self._load_forecast_scaled), 0)
        self._delta_pv_forecast_scaled = np.append(np.diff(self._pv_forecast_scaled), 0)
        self._delta_wind_onshore_forecast_scaled = np.append(
            np.diff(self._wind_onshore_forecast_scaled), 0
        )

    def log_data(
        self,
        modus: str,
        timestamp,
        episode_id,
        timestep,
        observations,
        action,
        reward,
        dam_price_forecast,
        dam_price,
        price_bid,
        capacity_bid,
        capacity_trade,
        delta_soc,
        remaining_cycles,
        profit,
    ):
        """
        datetime: datetime object
        market: dam or brm
        observations: observations
        actions: actions
        reward: reward
        dam_price: dam_price
        price_bid: aFRR_price_bid or dam_price_bid
        capacity_bid: aFRR_capacity_bid or dam_capacity_bid
        capacity_trade: actually traded capacity on either market
        prod_cost: prod_cost
        profit: profit

        every log will write 4 rows in the log file
        """

        if modus.startswith("train"):
            path = os.path.join(self._logging_path, TRAIN_CSV_NAME)
        elif modus.startswith("test"):
            path = os.path.join(self._logging_path, TEST_CSV_NAME)
        else:
            raise ValueError("Unknown mode {}", modus)

        # TODO: log episode
        # write to dataframe
        log_df = pd.DataFrame(
            columns=[
                "time",
                "episode_id",
                "timestep",
                "dam_price_forecast",
                "epex_spot_60min_de_lu_eur_per_mwh",  # spare market info
                "action_1",  # actions
                # "action_2",
                "reward",
                "price_bid",
                "capacity_bid",
                "capacity_trade",
                "obs: soc_t",
                "delta_soc",
                "remaining_cycles",
                "profit",
            ]
        )

        # columns for observation

        log_df.loc[0, "obs: soc_t"] = observations[1]
        log_df.loc[0, "time"] = pd.to_datetime(timestamp, utc=True).isoformat()
        log_df.loc[0, "episode_id"] = episode_id
        log_df.loc[0, "timestep"] = timestep
        log_df.loc[0, "action_1"] = action
        # log_df.loc[0, "action_2"] = actions[1]
        log_df.loc[0, "dam_price_forecast"] = dam_price_forecast
        log_df.loc[0, "epex_spot_60min_de_lu_eur_per_mwh"] = dam_price
        log_df.loc[0, "reward"] = reward
        log_df.loc[0, "price_bid"] = price_bid
        log_df.loc[0, "capacity_bid"] = capacity_bid
        log_df.loc[0, "capacity_trade"] = capacity_trade
        log_df.loc[0, "profit"] = profit
        log_df.loc[0, "delta_soc"] = delta_soc
        log_df.loc[0, "remaining_cycles"] = remaining_cycles

        if not os.path.isfile(path):
            log_df.to_csv(path, header=True)
        else:  # else it exists so append without writing the header
            log_df.to_csv(path, mode="a", header=False)
