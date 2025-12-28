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
import warnings

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
warnings.simplefilter(action="ignore", category=FutureWarning)



RESUME_TRAINING = False      # set to TRUE, if training should be continued from a checkpoint

# Only relevant if RESUME_TRAINING = True
MODEL_NUMBER = "3"  
MODEL_CHECKPOINT = "ppo_stacked_checkpoint_280000_steps"


if __name__ == "__main__":
    
    # Ensure output folders exist
    os.makedirs(LOGGING_PATH_COORDINATED, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_PATH_COORDINATED, exist_ok=True)
    os.makedirs(SCALER_OUTPUT_PATH_COORDINATED, exist_ok=True)


    if RESUME_TRAINING:
        # vorhandene Version weiterverwenden
        versioned_log_path = os.path.join(LOGGING_PATH_COORDINATED, MODEL_NUMBER)
        versioned_model_path = os.path.join(MODEL_OUTPUT_PATH_COORDINATED, MODEL_NUMBER)
        versioned_scaler_path = os.path.join(SCALER_OUTPUT_PATH_COORDINATED, MODEL_NUMBER)
    else:
        # neue Version anlegen
        versioned_log_path = create_new_dir_version(LOGGING_PATH_COORDINATED)
        versioned_model_path = create_new_dir_version(MODEL_OUTPUT_PATH_COORDINATED)
        versioned_scaler_path = create_new_dir_version(SCALER_OUTPUT_PATH_COORDINATED)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare input data
    df_spot_train, df_spot_val, df_spot_test = load_input_data(write_test=False)

    # Fürs Training nur df_spot_train verwenden
    input_data_train = prepare_input_data(df_spot_train, versioned_scaler_path, fit_scaler=True)

    # Reward components logging path
    reward_log_path = os.path.join(versioned_log_path, "reward_components.csv")
    print(f"[Train Script] Reward components CSV: {reward_log_path}")

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

    if RESUME_TRAINING:
        # -> entspricht dem, was du im Testscript machst, nur für Training
        load_path = os.path.join(
            versioned_model_path,
            MODEL_CHECKPOINT + ".zip",
        )
        print(f"Resuming training from: {load_path}")

        model = CustomPPO.load(load_path, device=device)
        # Wichtig: neues Env anhängen (du hast weiter oben ein frisches env gebaut)
        model.set_env(env)

        # Bei weiterem Training: Schrittzähler nicht zurücksetzen
        reset_num_timesteps = False
    else:
        print("Starting training from scratch.")

        model = CustomPPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log=TENSORBOARD_PATH_INTELLIGENT,
            device=device,
            seed=SEED,
            intraday_product_type="QH",
            reward_log_path=os.path.join(versioned_log_path, "reward_components.csv"),
            policy_kwargs=policy_kwargs,
            ent_coef=0.05,
            n_steps=512,
            clip_range=0.4,
            batch_size=128,
            vf_coef=0.4,
            learning_rate=linear_schedule(0.003),
        )
        reset_num_timesteps = True

    # Train the model
    model.learn(
        total_timesteps=TRAINING_STEPS_INTELLIGENT,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps
    )

    # Save the final model
    model.save(os.path.join(versioned_model_path, COORDINATED_MODEL_NAME_QH))

    print("Finished training!")
