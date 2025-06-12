"""
Train PPO Agent for Coordinated Multi-Market Battery Dispatch

This script:
- Loads and preprocesses spot market data.
- Sets up a custom Stable-Baselines3 PPO environment (`BasicBatteryDAM`).
- Trains the agent using PPO with custom architecture and logging.
- Saves the model, logs, and scaler using versioned output folders.

Requires:
- Training data via `load_input_data()`
- `BasicBatteryDAM` environment for DRL coordination
- Stable-Baselines3 and PyTorch
"""

import os
import pandas as pd
import torch

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.coordinated_multi_market.basic_battery_dam_env import BasicBatteryDAM
from src.coordinated_multi_market.custom_ppo import CustomPPO
from src.coordinated_multi_market.learning_utils import (
    load_input_data,
    prepare_input_data,
    linear_schedule,
    orthogonal_weight_init,
    CustomPPO,
)

from src.shared.folder_versioning import create_new_dir_version
from src.shared.config import (
    COORDINATED_MODEL_NAME_QH,
    LOGGING_PATH_COORDINATED,
    MODEL_OUTPUT_PATH_COORDINATED,
    RTE,
    SCALER_OUTPUT_PATH_COORDINATED,
    SEED,
    TENSORBOARD_PATH_INTELLIGENT,
    TRAINING_STEPS_INTELLIGENT,
)

if __name__ == "__main__":
    # Ensure output folders exist
    os.makedirs(LOGGING_PATH_COORDINATED, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_PATH_COORDINATED, exist_ok=True)
    os.makedirs(SCALER_OUTPUT_PATH_COORDINATED, exist_ok=True)

    # Create versioned output folders
    versioned_log_path = create_new_dir_version(LOGGING_PATH_COORDINATED)
    versioned_model_path = create_new_dir_version(MODEL_OUTPUT_PATH_COORDINATED)
    versioned_scaler_path = create_new_dir_version(SCALER_OUTPUT_PATH_COORDINATED)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare training data
    df_spot_train, df_spot_test = load_input_data(write_test=True)

    # Drop problematic days known to break RI algorithm
    df_spot_train = df_spot_train[
        ~df_spot_train.index.date.isin(
            [pd.Timestamp("2020-11-15").date(), pd.Timestamp("2020-12-27").date()]
        )
    ]

    # Apply preprocessing and feature scaling
    input_data_train = prepare_input_data(df_spot_train, versioned_scaler_path)

    # Initialize training environment
    env = BasicBatteryDAM(
        modus="train",
        logging_path=versioned_log_path,
        input_data=input_data_train,
        round_trip_efficiency=RTE,
    )

    # Validate custom environment (optional)
    check_env(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Callback to save intermediate model checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=versioned_model_path,
        name_prefix="ppo_stacked_checkpoint",
    )

    # Define custom policy architecture
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64, 64], vf=[64, 64, 64, 64]),
        log_std_init=-0.5,
        # init_fn=orthogonal_weight_init,  # Optional: Orthogonal weight init
    )

    # Load pretrained model if continuing training
    # model = CustomPPO.load(
    #     path=os.path.join('output/multi_market_engine/models/80', 'ppo_stacked_checkpoint_220000_steps'),
    # )
    # model.set_env(env)

    # Instantiate a new PPO model
    model = CustomPPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=TENSORBOARD_PATH_INTELLIGENT,
        device=device,
        seed=SEED,
        intraday_product_type="QH",
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        n_steps=512,
        clip_range=0.4,
        batch_size=128,
        vf_coef=0.4,
        learning_rate=linear_schedule(0.003),
    )

    # Train the model
    model.learn(
        total_timesteps=TRAINING_STEPS_INTELLIGENT,
        callback=checkpoint_callback,
    )

    # Save the final model
    model.save(os.path.join(versioned_model_path, COORDINATED_MODEL_NAME_QH))

    print("Finished training!")
