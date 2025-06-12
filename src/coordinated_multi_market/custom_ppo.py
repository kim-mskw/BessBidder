import concurrent.futures

import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO

from src.coordinated_multi_market.rolling_intrinsic.training_rolling_intrinsic_h_intelligent_stacking import (
    simulate_period_hourly_products,
)
from src.coordinated_multi_market.rolling_intrinsic.training_rolling_intrinsic_qh_intelligent_stacking import (
    simulate_period_quarterhourly_products,
)
from src.shared.config import BUCKET_SIZE, C_RATE, MAX_CYCLES_PER_DAY, MIN_TRADES, RTE


class CustomPPO(PPO):
    def __init__(self, *args, intraday_product_type: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.intraday_product_type = intraday_product_type
        self.current_step = 0
        self.lambda_val = 0.5
        self._last_ri_reward_per_euro = 0

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        timestamp_buffer = np.zeros(2048)
        position_buffer = np.zeros(2048)
        clearing_price_buffer = np.zeros(2048)
        # TODO: Add revenue buffer for scaling the rewards

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            if n_steps > 1:
                timestamp_buffer[n_steps - 1] = infos[0]["timestamp"]
                position_buffer[n_steps - 1] = infos[0]["position"]
                clearing_price_buffer[n_steps - 1] = infos[0]["clearing_price"]
                scaling_max_price = infos[0]["scaling_max_price"]
                scaling_min_price = infos[0]["scaling_min_price"]

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        complete_periods = self._derive_period_lenghts_from_episode_starts_array(
            rollout_buffer.episode_starts.flatten()
        )
        for row_start, num_rows in complete_periods.items():
            period_timestamps = pd.to_datetime(
                timestamp_buffer[row_start : row_start + num_rows], utc=True
            ).tz_convert("Europe/Berlin")
            period_volumes = position_buffer[row_start : row_start + num_rows]
            if not self._check_if_complete_cycle(period_volumes):
                continue

            if (
                self.num_timesteps >= 200_000
            ):  # in case you want to first train only on DA for 240_000 -> set num_timesteps to 240_000)
                period_clearing_prices = clearing_price_buffer[
                    row_start : row_start + num_rows
                ]
                da_trades = self._derive_day_ahead_trades(
                    timestamps=period_timestamps,
                    volumes=period_volumes,
                    clearing_prices=period_clearing_prices,
                    intraday_product_type=self.intraday_product_type,
                )

                if self.intraday_product_type == "H":
                    (
                        rolling_intrinsic_results_stacked,
                        rolling_intrinsic_results_non_stacked,
                    ) = self.run_simulations_hourly_products_in_parallel(
                        period_timestamps, da_trades
                    )
                elif self.intraday_product_type == "QH":
                    (
                        rolling_intrinsic_results_stacked
                        # rolling_intrinsic_results_non_stacked,
                    ) = self.run_simulations_quarterhourly_products_in_parallel(
                        period_timestamps, da_trades
                    )
                else:
                    raise ValueError(
                        "Unsupported intraday product type %s. Only QH or H supported"
                        % self.intraday_product_type
                    )

                # TODO: Change this scaling it is way too confusing and I need hourly mapping of RI profit

                da_profit = da_trades.profit.sum()
                da_reward_sum = (
                    rollout_buffer.rewards[row_start : row_start + num_rows]
                    .flatten()
                    .sum()
                )

                ri_stacked_profit = rolling_intrinsic_results_stacked[
                    "total_profit"
                ].sum()

                ri_reward_per_scaled = ri_stacked_profit / (85) / num_rows

                # just scale by average reward made in IDM in naive case
                rolling_intrinsic_rewards = np.repeat(ri_reward_per_scaled, num_rows)

                da_rewards = rollout_buffer.rewards[
                    row_start : row_start + num_rows
                ].flatten()

                combined_rewards = (
                    self.lambda_val * da_rewards
                    + (1 - self.lambda_val) * rolling_intrinsic_rewards
                )
                rollout_buffer.rewards[row_start : row_start + num_rows] = (
                    combined_rewards.reshape(-1, 1)
                )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    @staticmethod
    def run_simulations_quarterhourly_products_in_parallel(
        period_timestamps, da_trades
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_stacked = executor.submit(
                simulate_period_quarterhourly_products,
                start_day=period_timestamps[0],
                end_day=period_timestamps[0] + pd.Timedelta(days=1),
                threshold=0,
                threshold_abs_min=0,
                discount_rate=0,
                bucket_size=BUCKET_SIZE,
                c_rate=C_RATE,
                roundtrip_eff=RTE,
                max_cycles=MAX_CYCLES_PER_DAY,
                min_trades=MIN_TRADES,
                day_ahead_trades_drl=da_trades,
            )

            # future_non_stacked = executor.submit(
            #    simulate_period_quarterhourly_products,
            #   start_day=period_timestamps[0],
            #   end_day=period_timestamps[0] + pd.Timedelta(days=1),
            #    threshold=0,
            #   threshold_abs_min=0,
            #    discount_rate=0,
            #    bucket_size=BUCKET_SIZE,
            #    c_rate=C_RATE,
            #    roundtrip_eff=RTE,
            #    max_cycles=MAX_CYCLES_PER_DAY,
            #    min_trades=MIN_TRADES,
            # )

            rolling_intrinsic_results_stacked = future_stacked.result()
            # rolling_intrinsic_results_non_stacked = future_non_stacked.result()
        # rolling_intrinsic_results_stacked = future_stacked.copy()

        return rolling_intrinsic_results_stacked

    @staticmethod
    def run_simulations_hourly_products_in_parallel(period_timestamps, da_trades):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_stacked = executor.submit(
                simulate_period_hourly_products,
                start_day=period_timestamps[0],
                end_day=period_timestamps[0] + pd.Timedelta(days=1),
                threshold=0,
                threshold_abs_min=0,
                discount_rate=0,
                bucket_size=BUCKET_SIZE,
                c_rate=C_RATE,
                roundtrip_eff=RTE,
                max_cycles=MAX_CYCLES_PER_DAY,
                min_trades=MIN_TRADES,
                day_ahead_trades_drl=da_trades,
            )

            future_non_stacked = executor.submit(
                simulate_period_hourly_products,
                start_day=period_timestamps[0],
                end_day=period_timestamps[0] + pd.Timedelta(days=1),
                threshold=0,
                threshold_abs_min=0,
                discount_rate=0,
                bucket_size=BUCKET_SIZE,
                c_rate=C_RATE,
                roundtrip_eff=RTE,
                max_cycles=MAX_CYCLES_PER_DAY,
                min_trades=MIN_TRADES,
            )

            rolling_intrinsic_results_stacked = future_stacked.result()
            rolling_intrinsic_results_non_stacked = future_non_stacked.result()

        return rolling_intrinsic_results_stacked, rolling_intrinsic_results_non_stacked

    @staticmethod
    def _derive_period_lenghts_from_episode_starts_array(episode_starts: np.ndarray):
        # Find the indices where episodes start
        episode_indices = np.where(episode_starts == 1)[0]
        # Calculate episode lengths by finding the difference between consecutive start indices
        episode_lengths = np.diff(episode_indices)
        # Add the length of the last episode (from the last start to the end of the array)
        final_length = len(episode_starts) - 1 - episode_indices[-1]
        episode_lengths = np.append(episode_lengths, final_length)
        episode_dict = {
            index: length
            for index, length in zip(episode_indices, episode_lengths)
            # if length == 24
        }
        return episode_dict

    @staticmethod
    def _check_if_complete_cycle(period_volumes):
        traded_volume = abs(period_volumes).sum()
        return traded_volume / (2 * 1) == 1

    @staticmethod
    def _derive_day_ahead_trades(
        timestamps, volumes, clearing_prices, intraday_product_type: str
    ):
        day_ahead_trades = {}
        for idx in range(len(timestamps)):
            if volumes[idx] == 0:
                continue

            side = "buy" if volumes[idx] < 0 else "sell"
            net_volume = abs(volumes[idx])
            price = clearing_prices[idx]
            profit = volumes[idx] * clearing_prices[idx]

            day_ahead_market_clearing = (timestamps[0] - pd.Timedelta(days=1)).replace(
                hour=13
            )

            if intraday_product_type == "H":
                day_ahead_trades.update(
                    {
                        timestamps[idx]: {
                            "execution_time": day_ahead_market_clearing,
                            "side": side,
                            "quantity": net_volume,
                            "price": price,
                            "product": timestamps[idx],
                            "profit": profit,
                        }
                    }
                )
            elif intraday_product_type == "QH":
                product_indexes = pd.date_range(
                    timestamps[idx], periods=4, freq="15min"
                )
                for product_index in product_indexes:
                    day_ahead_trades.update(
                        {
                            product_index: {
                                "execution_time": day_ahead_market_clearing,
                                "side": side,
                                "quantity": net_volume,
                                "price": price,
                                "product": timestamps[idx],
                                "profit": profit / 4,
                            }
                        }
                    )
            else:
                raise ValueError(
                    "Wrong intraday product type %s. Only QH or H allowed."
                    % intraday_product_type
                )

        return pd.DataFrame(day_ahead_trades).T.reset_index(drop=True)
