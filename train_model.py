"""
train_model.py — Training Pipeline for FocusManagement DQN Agent v2
=====================================================================
Chạy file này để huấn luyện model:
    python train_model.py

Sau khi train xong sẽ có:
    dqn_focus_agent_v2.zip   — DQN model weights
    scaler_v2.pkl            — StandardScaler để reuse khi evaluate

Lưu ý:
    - session_id  → chỉ dùng để group sessions thành episodes, KHÔNG vào state
    - timestamp   → derive work_duration_min, KHÔNG vào state trực tiếp
    - age, hour, occupation → inject ngẫu nhiên per episode khi training
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from focus_env import (
    CONT_COLS,
    FPS,
    N_ACTIONS,
    STATE_DIM,
    ACTION_NAMES,
    FocusManagementEnv,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CSV_FILE    = "focus_dataset.csv"
MODEL_PATH  = "dqn_focus_agent_v2"
SCALER_PATH = "scaler_v2.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING CALLBACK
# ──────────────────────────────────────────────────────────────────────────────


class TrainingLogger(BaseCallback):
    """
    Callback ghi lại episode reward và in tóm tắt mỗi N episodes.
    """

    def __init__(self, print_freq: int = 5, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards: List[float] = []
        self._current_ep_reward: float = 0.0
        self._ep_count: int = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done   = self.locals.get("dones", [False])[0]
        self._current_ep_reward += float(reward)

        if done:
            self._ep_count += 1
            self.episode_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0.0

            if self._ep_count % self.print_freq == 0:
                last_n = self.episode_rewards[-self.print_freq:]
                print(
                    f"  [Episode {self._ep_count:4d}]  "
                    f"mean_reward={np.mean(last_n):+7.2f}  "
                    f"min={np.min(last_n):+7.2f}  "
                    f"max={np.max(last_n):+7.2f}"
                )
        return True


# ──────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────────────


def repair_session_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair broken session data where timestamp and/or consecutive_frames
    are all zero. This is common when the data collection pipeline didn't
    record these fields.

    Fixes applied per session:
        - timestamp: If all zeros, reconstruct from frame index at ~3 FPS
        - consecutive_frames: If all zeros, recompute from is_distracted_label
          as a running count of consecutive distracted frames

    Returns:
        DataFrame with repaired columns.
    """
    df = df.copy()
    repaired = 0

    for sid, idx in df.groupby("session_id").groups.items():
        group = df.loc[idx]

        # Fix timestamps
        if group["timestamp"].nunique() <= 1 and group["timestamp"].iloc[0] == 0.0:
            n_frames = len(group)
            df.loc[idx, "timestamp"] = np.arange(n_frames) / FPS
            repaired += 1
            print(f"    [REPAIR] Session {sid}: reconstructed timestamps "
                  f"({n_frames} frames at {FPS} FPS)")

        # Fix consecutive_frames
        if group["consecutive_frames"].nunique() <= 1 and group["consecutive_frames"].iloc[0] == 0:
            labels = group["is_distracted_label"].values
            consec = np.zeros(len(labels), dtype=np.int64)
            for i in range(len(labels)):
                if labels[i] == 1:
                    consec[i] = (consec[i - 1] + 1) if i > 0 else 1
                else:
                    consec[i] = 0
            df.loc[idx, "consecutive_frames"] = consec
            max_c = consec.max()
            repaired += 1
            print(f"    [REPAIR] Session {sid}: recomputed consecutive_frames "
                  f"(max={max_c})")

    if repaired == 0:
        print("    [OK] No repairs needed")

    return df


def prepare_data(csv_path: str) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Load và tiền xử lý dữ liệu từ CSV.

    Pipeline:
        1. Load CSV
        2. Repair broken sessions (timestamp=0, consecutive_frames=0)
        3. Normalise continuous features (CONT_COLS only)
        4. Fill NaN with 0

    Returns:
        df     — DataFrame đã normalised
        scaler — fitted StandardScaler (lưu lại để dùng khi evaluate)
    """
    print(f"[*] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"    Rows:     {len(df):,}")
    print(f"    Sessions: {df['session_id'].nunique()}")
    print(f"    Distraction rate: {df['is_distracted_label'].mean():.1%}")
    print(f"    Avg session length: {len(df) / df['session_id'].nunique():.0f} frames")

    # Repair broken sessions before normalisation
    print("[*] Checking data quality...")
    df = repair_session_data(df)

    # Normalise raw continuous features only
    scaler = StandardScaler()
    df[CONT_COLS] = scaler.fit_transform(df[CONT_COLS])
    df.fillna(0, inplace=True)

    print(f"[+] Preprocessing done.  State dim = {STATE_DIM}, Actions = {N_ACTIONS}")
    return df, scaler


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────


def run_training_pipeline(
    total_timesteps: int = 50_000,
    net_arch: Optional[List[int]] = None,
    learning_rate: float = 5e-4,
    buffer_size: int = 50_000,
    batch_size: int = 128,
    learning_starts: int = 2_000,
    target_update_interval: int = 500,
    exploration_fraction: float = 0.25,
    exploration_final_eps: float = 0.05,
) -> tuple[DQN, StandardScaler]:
    """
    Pipeline đầy đủ:
        1. Load & preprocess data
        2. Khởi tạo environment (window-based, context injection)
        3. Khởi tạo DQN agent với tuned hyperparameters
        4. Train với callback logging
        5. Lưu model + scaler

    Args:
        total_timesteps:         Số bước train tổng cộng
        net_arch:                Kiến trúc mạng [hidden1, hidden2, ...]
        learning_rate:           Learning rate cho Adam optimizer
        buffer_size:             Kích thước replay buffer
        batch_size:              Mini-batch size
        learning_starts:         Số random steps trước khi bắt đầu học
        target_update_interval:  Update target network mỗi N steps
        exploration_fraction:    Fraction of training với epsilon decay
        exploration_final_eps:   Epsilon cuối cùng (min exploration)

    Returns:
        model  — trained DQN model
        scaler — fitted StandardScaler
    """
    if net_arch is None:
        net_arch = [256, 256]

    # ── 1. Data ───────────────────────────────────────────────────────────
    df, scaler = prepare_data(CSV_FILE)

    # ── 2. Environment ────────────────────────────────────────────────────
    env = FocusManagementEnv(df, randomize_context=True)
    env = Monitor(env)

    # ── 3. Model ──────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=net_arch)

    print(f"\n[*] Initialising DQN Agent")
    print(f"    State dim     : {STATE_DIM}")
    print(f"    Action dim    : {N_ACTIONS}  {list(ACTION_NAMES.values())}")
    print(f"    Net arch      : {net_arch}")
    print(f"    Buffer size   : {buffer_size:,}")
    print(f"    Batch size    : {batch_size}")
    print(f"    Learning rate : {learning_rate}")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate            = learning_rate,
        buffer_size              = buffer_size,
        learning_starts          = learning_starts,
        batch_size               = batch_size,
        gamma                    = 0.99,
        tau                      = 1.0,
        target_update_interval   = target_update_interval,
        train_freq               = 4,
        gradient_steps           = 1,
        exploration_fraction     = exploration_fraction,
        exploration_final_eps    = exploration_final_eps,
        policy_kwargs            = policy_kwargs,
        verbose                  = 1,
    )

    # ── 4. Training ───────────────────────────────────────────────────────
    callback = TrainingLogger(print_freq=5)
    print(f"\n[*] Training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps = total_timesteps,
        callback        = callback,
        progress_bar    = True,
    )

    # Summary
    if callback.episode_rewards:
        print(f"\n[+] Training Summary")
        print(f"    Episodes completed : {len(callback.episode_rewards)}")
        print(f"    Final-20 mean reward: "
              f"{np.mean(callback.episode_rewards[-20:]):+.2f}")
        print(f"    Best episode reward : {max(callback.episode_rewards):+.2f}")

    # ── 5. Save ───────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[+] Model saved  : {MODEL_PATH}.zip")
    print(f"[+] Scaler saved : {SCALER_PATH}")
    print("[+] Training complete!\n")

    return model, scaler


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_training_pipeline(
        total_timesteps          = 50_000,
        net_arch                 = [256, 256],
        learning_rate            = 5e-4,
        buffer_size              = 50_000,
        batch_size               = 128,
        learning_starts          = 2_000,
        target_update_interval   = 500,
        exploration_fraction     = 0.25,
        exploration_final_eps    = 0.05,
    )
