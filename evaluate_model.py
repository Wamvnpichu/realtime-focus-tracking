"""
evaluate_model.py — Evaluation & Testing Pipeline for FocusManagement DQN Agent v2
=====================================================================================
Chạy file này để test model đã train:
    python evaluate_model.py

    # Hoặc với context cố định:
    python evaluate_model.py --age 35 --hour 9 --occupation office --episodes 100

Output:
    - In báo cáo chi tiết ra terminal
    - Lưu biểu đồ: evaluation_results.png

Evaluation Metrics:
    Overall:
        mean_reward, std_reward, median_reward, % positive episodes
    Decision Quality:
        oracle_accuracy       — % action đúng theo rule-based oracle
        false_alarm_rate      — % lần nudge khi user đang focused
        miss_rate             — % lần im lặng khi xao nhãng nặng
        early_detection_rate  — % lần can thiệp proactive ở early-warning zone
    Action Analysis:
        action_distribution   — tỷ lệ mỗi action
        reward_per_action     — mean reward của mỗi action
    Context Analysis:
        by_age_group          — performance theo nhóm tuổi
        by_time_of_day        — performance theo giờ trong ngày
        by_occupation         — performance theo nghề nghiệp
    Error Analysis:
        confusion_matrix      — agent action vs oracle action
        classification_report — precision, recall, F1 per action class
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from stable_baselines3 import DQN

from focus_env import (
    CONT_COLS,
    FPS,
    N_ACTIONS,
    ACTION_NAMES,
    OCCUPATIONS,
    WINDOW_SIZE,
    STATE_DIM,
    FocusManagementEnv,
)
from train_model import repair_session_data

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH  = "dqn_focus_agent_v2"
SCALER_PATH = "scaler_v2.pkl"
CSV_FILE    = "focus_dataset.csv"
PLOT_PATH   = "evaluation_results.png"

# ──────────────────────────────────────────────────────────────────────────────
# ORACLE
# ──────────────────────────────────────────────────────────────────────────────


def oracle_action(distraction_ratio: float, consec_max_norm: float) -> int:
    """
    Rule-based oracle action — ground-truth baseline for accuracy measurement.

    This encodes the "ideal" domain policy:
        Focused      (< 0.20)  → Do Nothing (0)
        Early warning(0.20–0.45) → Play Focus Music (5)  [gentle, proactive]
        Moderate     (0.45–0.75) → Suggest Short Break (3)
        Severe       (≥ 0.75)  → Sound Alert (2)  [strong intervention]
    """
    if distraction_ratio < 0.20:
        return 0
    elif distraction_ratio < 0.45:
        return 5
    elif distraction_ratio < 0.75:
        return 3
    else:
        return 2


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """Record of one environment step."""
    action:            int
    oracle:            int
    reward:            float
    distraction_ratio: float
    consec_max_norm:   float
    early_signal:      bool
    age:               int
    hour:              float
    occupation:        str
    work_duration_min: float


@dataclass
class EpisodeRecord:
    """Aggregated record of one full episode."""
    total_reward: float = 0.0
    n_steps:      int   = 0
    steps: List[StepRecord] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION RUNNER
# ──────────────────────────────────────────────────────────────────────────────


def run_evaluation(
    model: DQN,
    env:   FocusManagementEnv,
    n_episodes: int = 100,
) -> List[EpisodeRecord]:
    """
    Run the trained model for n_episodes and collect detailed step records.

    Returns:
        List of EpisodeRecord objects with full step-by-step data.
    """
    episodes: List[EpisodeRecord] = []

    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        episode = EpisodeRecord()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            oracle = oracle_action(
                info["distraction_ratio"], info["consec_max_norm"]
            )

            episode.total_reward += reward
            episode.n_steps      += 1
            episode.steps.append(StepRecord(
                action            = int(action),
                oracle            = oracle,
                reward            = reward,
                distraction_ratio = info["distraction_ratio"],
                consec_max_norm   = info["consec_max_norm"],
                early_signal      = bool(info["early_signal"]),
                age               = info["age"],
                hour              = info["hour"],
                occupation        = info["occupation"],
                work_duration_min = info["work_duration_min"],
            ))

        episodes.append(episode)

        if (ep_idx + 1) % 10 == 0:
            recent = [e.total_reward for e in episodes[-10:]]
            print(
                f"  Ep {ep_idx+1:3d}/{n_episodes}  |  "
                f"last-10 mean reward = {np.mean(recent):+.2f}"
            )

    return episodes


# ──────────────────────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(episodes: List[EpisodeRecord]) -> Dict[str, Any]:
    """
    Compute all evaluation metrics from episode records.

    Returns:
        Dictionary with all computed metrics.
    """
    all_steps = [s for ep in episodes for s in ep.steps]
    ep_rewards = [ep.total_reward for ep in episodes]

    actions = [s.action for s in all_steps]
    oracles = [s.oracle for s in all_steps]

    m: Dict[str, Any] = {}

    # ── Overall Performance ───────────────────────────────────────────────
    m["n_episodes"]        = len(episodes)
    m["n_steps"]           = len(all_steps)
    m["mean_reward"]       = float(np.mean(ep_rewards))
    m["std_reward"]        = float(np.std(ep_rewards))
    m["median_reward"]     = float(np.median(ep_rewards))
    m["min_reward"]        = float(np.min(ep_rewards))
    m["max_reward"]        = float(np.max(ep_rewards))
    m["positive_ep_rate"]  = float(np.mean([r > 0 for r in ep_rewards]))
    m["episode_rewards"]   = ep_rewards

    # ── Action Distribution ───────────────────────────────────────────────
    counts = np.bincount(actions, minlength=N_ACTIONS)
    m["action_counts"]       = counts.tolist()
    m["action_distribution"] = (counts / len(actions)).tolist()

    # ── Oracle Accuracy ───────────────────────────────────────────────────
    m["oracle_accuracy"] = float(
        np.mean([a == o for a, o in zip(actions, oracles)])
    )

    # ── False Alarm Rate: nudge when user is focused ──────────────────────
    focused = [s for s in all_steps if s.distraction_ratio < 0.20]
    m["false_alarm_rate"] = (
        float(np.mean([s.action != 0 for s in focused])) if focused else 0.0
    )

    # ── Miss Rate: silent when severely distracted ────────────────────────
    severe = [s for s in all_steps if s.distraction_ratio >= 0.75]
    m["miss_rate"] = (
        float(np.mean([s.action == 0 for s in severe])) if severe else 0.0
    )

    # ── Early Detection Rate ──────────────────────────────────────────────
    early = [s for s in all_steps if s.early_signal]
    m["early_detection_rate"] = (
        float(np.mean([s.action != 0 for s in early])) if early else 0.0
    )
    m["n_early_signal_steps"] = len(early)

    # ── Mean Reward per Action ────────────────────────────────────────────
    m["reward_per_action"] = {}
    for a in range(N_ACTIONS):
        a_rews = [s.reward for s in all_steps if s.action == a]
        m["reward_per_action"][ACTION_NAMES[a]] = (
            float(np.mean(a_rews)) if a_rews else 0.0
        )

    # ── Mean Step Reward by Distraction Zone ─────────────────────────────
    zones = {
        "Focused (< 0.20)":          lambda s: s.distraction_ratio < 0.20,
        "Early (0.20-0.45)":         lambda s: 0.20 <= s.distraction_ratio < 0.45,
        "Moderate (0.45-0.75)":      lambda s: 0.45 <= s.distraction_ratio < 0.75,
        "Severe (>= 0.75)":          lambda s: s.distraction_ratio >= 0.75,
    }
    m["reward_by_zone"] = {}
    m["n_steps_by_zone"] = {}
    for name, fn in zones.items():
        grp = [s.reward for s in all_steps if fn(s)]
        m["reward_by_zone"][name]  = float(np.mean(grp)) if grp else 0.0
        m["n_steps_by_zone"][name] = len(grp)

    # -- Performance by Age Group ------------------------------------------
    age_groups = {
        "18-29":  lambda s: 18 <= s.age < 30,
        "30-44":  lambda s: 30 <= s.age < 45,
        "45-59":  lambda s: 45 <= s.age < 60,
        "60+":    lambda s: s.age >= 60,
    }
    m["reward_by_age"] = {}
    for grp, fn in age_groups.items():
        vals = [s.reward for s in all_steps if fn(s)]
        m["reward_by_age"][grp] = float(np.mean(vals)) if vals else 0.0

    # -- Performance by Time of Day ----------------------------------------
    hour_groups = {
        "Morning (6-12)":    lambda s: 6  <= s.hour < 12,
        "Afternoon (12-17)": lambda s: 12 <= s.hour < 17,
        "Evening (17-22)":   lambda s: 17 <= s.hour < 22,
        "Night (22-6)":      lambda s: s.hour >= 22 or s.hour < 6,
    }
    m["reward_by_hour"] = {}
    for grp, fn in hour_groups.items():
        vals = [s.reward for s in all_steps if fn(s)]
        m["reward_by_hour"][grp] = float(np.mean(vals)) if vals else 0.0

    # -- Performance by Occupation -----------------------------------------
    m["reward_by_occupation"] = {}
    for occ in OCCUPATIONS:
        vals = [s.reward for s in all_steps if s.occupation == occ]
        m["reward_by_occupation"][occ] = float(np.mean(vals)) if vals else 0.0

    # -- Confusion Matrix & Classification Report --------------------------
    m["confusion_matrix"] = confusion_matrix(
        oracles, actions, labels=list(range(N_ACTIONS))
    )
    m["classification_report"] = classification_report(
        oracles, actions,
        labels      = list(range(N_ACTIONS)),
        target_names= [ACTION_NAMES[i] for i in range(N_ACTIONS)],
        zero_division= 0,
    )

    # -- Reward rolling average (per-episode) ------------------------------
    window = min(10, len(ep_rewards))
    m["reward_rolling_avg"] = [
        float(np.mean(ep_rewards[max(0, i - window + 1): i + 1]))
        for i in range(len(ep_rewards))
    ]

    return m


# ------------------------------------------------------------------------------
# REPORT PRINTING
# ------------------------------------------------------------------------------


def print_report(m: Dict[str, Any]) -> None:
    """Print a formatted evaluation report to stdout."""
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  EVALUATION REPORT — FocusManagement DQN Agent v2")
    print(SEP)

    print("\n[*] OVERALL PERFORMANCE")
    print(f"    Episodes evaluated  : {m['n_episodes']}")
    print(f"    Total steps         : {m['n_steps']:,}")
    print(f"    Mean episode reward : {m['mean_reward']:+.3f}  "
          f"+/- {m['std_reward']:.3f}")
    print(f"    Median reward       : {m['median_reward']:+.3f}")
    print(f"    Range               : [{m['min_reward']:+.1f}, {m['max_reward']:+.1f}]")
    print(f"    Positive episodes   : {m['positive_ep_rate']:.1%}")

    print("\n[*] DECISION QUALITY")
    print(f"    Oracle accuracy     : {m['oracle_accuracy']:.1%}")
    print(f"    False alarm rate    : {m['false_alarm_rate']:.1%}"
          f"  <- nudge when focused")
    print(f"    Miss rate           : {m['miss_rate']:.1%}"
          f"  <- silent when severely distracted")
    print(f"    Early detect rate   : {m['early_detection_rate']:.1%}"
          f"  <- proactive at early-warning ({m['n_early_signal_steps']:,} steps)")

    print("\n[*] ACTION DISTRIBUTION")
    for i, (prob, cnt) in enumerate(
        zip(m["action_distribution"], m["action_counts"])
    ):
        bar = "#" * int(prob * 35)
        print(f"    {i} {ACTION_NAMES[i]:<26} {prob:5.1%}  "
              f"({cnt:5,})  {bar}")

    print("\n[*] MEAN REWARD PER ACTION")
    for name, r in m["reward_per_action"].items():
        flag = "[OK]" if r > 0 else "[--]"
        print(f"    {flag} {name:<26} {r:+.3f}")

    print("\n[*] MEAN REWARD BY DISTRACTION ZONE")
    for zone, r in m["reward_by_zone"].items():
        n = m["n_steps_by_zone"][zone]
        print(f"    {zone:<28} {r:+.3f}  ({n:,} steps)")

    print("\n[*] PERFORMANCE BY AGE GROUP")
    for grp, r in m["reward_by_age"].items():
        print(f"    {grp:<10} mean step reward: {r:+.3f}")

    print("\n[*] PERFORMANCE BY TIME OF DAY")
    for grp, r in m["reward_by_hour"].items():
        grp_clean = grp.replace("\n", " ")
        print(f"    {grp_clean:<22} mean step reward: {r:+.3f}")

    print("\n[*] PERFORMANCE BY OCCUPATION")
    for occ, r in m["reward_by_occupation"].items():
        print(f"    {occ:<12} mean step reward: {r:+.3f}")

    print("\n[*] ORACLE vs AGENT --- CLASSIFICATION REPORT")
    print(m["classification_report"])

    print(SEP + "\n")


# ------------------------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------------------------


def plot_results(m: Dict[str, Any], save_path: str = PLOT_PATH) -> None:
    """
    Generate a comprehensive 9-panel evaluation dashboard and save to PNG.

    Panels:
        Row 1: Episode Reward Distribution | Rolling Avg Reward | Action Distribution
        Row 2: Key Metrics               | Reward per Action   | Reward by Age
        Row 3: Reward by Time of Day     | Reward by Zone      | Confusion Matrix
    """
    plt.rcParams.update({
        "font.family":  "DejaVu Sans",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "figure.facecolor":   "#f8f9fa",
        "axes.facecolor":     "#ffffff",
    })

    fig = plt.figure(figsize=(22, 17))
    fig.suptitle(
        "FocusManagement DQN Agent v2 — Evaluation Dashboard",
        fontsize=17, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    action_colors = plt.cm.tab10(np.linspace(0, 0.6, N_ACTIONS))

    # -- Panel 1: Episode Reward Histogram ---------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ep_rewards = m["episode_rewards"]
    ax1.hist(ep_rewards, bins=max(15, len(ep_rewards) // 5),
             color="steelblue", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.axvline(m["mean_reward"], color="crimson", linestyle="--",
                linewidth=2, label=f"Mean: {m['mean_reward']:+.1f}")
    ax1.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax1.set_title("Episode Reward Distribution", fontweight="bold")
    ax1.set_xlabel("Total Episode Reward")
    ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=9)

    # -- Panel 2: Rolling Average Reward -----------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    rolling = m["reward_rolling_avg"]
    ax2.plot(rolling, color="darkorange", linewidth=2.0)
    ax2.fill_between(range(len(rolling)), rolling, alpha=0.18, color="orange")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax2.set_title("Rolling Average Episode Reward (window=10)", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean Reward")

    # -- Panel 3: Action Distribution --------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    dist = m["action_distribution"]
    bars = ax3.bar(range(N_ACTIONS), dist, color=action_colors,
                   edgecolor="white", linewidth=0.8)
    ax3.set_title("Action Distribution", fontweight="bold")
    ax3.set_xlabel("Action")
    ax3.set_ylabel("Proportion")
    ax3.set_xticks(range(N_ACTIONS))
    ax3.set_xticklabels(
        [f"{i}\n{ACTION_NAMES[i][:10]}" for i in range(N_ACTIONS)], fontsize=8
    )
    for bar, val in zip(bars, dist):
        if val > 0.01:
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.008,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    # -- Panel 4: Key Metrics Bar ------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    key = {
        "Oracle\nAccuracy":    m["oracle_accuracy"],
        "False\nAlarm (low)": m["false_alarm_rate"],
        "Miss\nRate (low)":   m["miss_rate"],
        "Early\nDetection":   m["early_detection_rate"],
        "Positive\nEpisodes": m["positive_ep_rate"],
    }
    key_colors = ["#2ecc71", "#e74c3c", "#e74c3c", "#2ecc71", "#2ecc71"]
    bars4 = ax4.bar(key.keys(), key.values(), color=key_colors,
                    alpha=0.80, edgecolor="white")
    ax4.set_ylim(0, 1.18)
    ax4.set_title("Key Performance Metrics", fontweight="bold")
    ax4.set_ylabel("Rate")
    for bar, val in zip(bars4, key.values()):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.025,
                 f"{val:.1%}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    # -- Panel 5: Mean Reward per Action -----------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    rpa_vals = [m["reward_per_action"][ACTION_NAMES[i]] for i in range(N_ACTIONS)]
    rpa_colors = ["#27ae60" if r >= 0 else "#e74c3c" for r in rpa_vals]
    bars5 = ax5.bar(range(N_ACTIONS), rpa_vals, color=rpa_colors,
                    alpha=0.85, edgecolor="white")
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.set_title("Mean Reward per Action", fontweight="bold")
    ax5.set_xlabel("Action")
    ax5.set_ylabel("Mean Step Reward")
    ax5.set_xticks(range(N_ACTIONS))
    ax5.set_xticklabels(
        [f"{i}\n{ACTION_NAMES[i][:10]}" for i in range(N_ACTIONS)], fontsize=8
    )
    for bar, val in zip(bars5, rpa_vals):
        offset = 0.05 if val >= 0 else -0.25
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + offset,
                 f"{val:+.2f}", ha="center", va="bottom", fontsize=8)

    # -- Panel 6: Reward by Age Group --------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    age_labels = list(m["reward_by_age"].keys())
    age_vals   = list(m["reward_by_age"].values())
    ax6.bar(age_labels, age_vals, color="#9b59b6", alpha=0.82, edgecolor="white")
    ax6.axhline(0, color="black", linewidth=0.8)
    ax6.set_title("Mean Step Reward by Age Group", fontweight="bold")
    ax6.set_xlabel("Age Group")
    ax6.set_ylabel("Mean Step Reward")

    # -- Panel 7: Reward by Time of Day ------------------------------------
    ax7 = fig.add_subplot(gs[2, 0])
    hour_labels = list(m["reward_by_hour"].keys())
    hour_vals   = list(m["reward_by_hour"].values())
    hour_palette = ["#f39c12", "#3498db", "#e67e22", "#2c3e50"]
    ax7.bar(hour_labels, hour_vals, color=hour_palette, alpha=0.85, edgecolor="white")
    ax7.axhline(0, color="black", linewidth=0.8)
    ax7.set_title("Mean Step Reward by Time of Day", fontweight="bold")
    ax7.set_xlabel("Period")
    ax7.set_ylabel("Mean Step Reward")
    ax7.tick_params(axis="x", labelsize=8)

    # -- Panel 8: Reward by Distraction Zone -------------------------------
    ax8 = fig.add_subplot(gs[2, 1])
    zone_labels = [z.split(" ")[0] for z in m["reward_by_zone"].keys()]
    zone_full   = list(m["reward_by_zone"].keys())
    zone_vals   = list(m["reward_by_zone"].values())
    zone_colors = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c"]
    bars8 = ax8.bar(zone_labels, zone_vals, color=zone_colors,
                    alpha=0.85, edgecolor="white")
    ax8.axhline(0, color="black", linewidth=0.8)
    ax8.set_title("Mean Step Reward by Distraction Zone", fontweight="bold")
    ax8.set_xlabel("Zone")
    ax8.set_ylabel("Mean Step Reward")
    for bar, zf in zip(bars8, zone_full):
        n = m["n_steps_by_zone"][zf]
        ax8.text(bar.get_x() + bar.get_width() / 2,
                 ax8.get_ylim()[0] + 0.05 * (ax8.get_ylim()[1] - ax8.get_ylim()[0]),
                 f"n={n:,}", ha="center", va="bottom", fontsize=7, color="gray")

    # -- Panel 9: Confusion Matrix -----------------------------------------
    ax9 = fig.add_subplot(gs[2, 2])
    cm = m["confusion_matrix"].astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, where=row_sums > 0)
    short_names = [f"{i}:{ACTION_NAMES[i][:9]}" for i in range(N_ACTIONS)]
    sns.heatmap(
        cm_norm, ax=ax9,
        annot=m["confusion_matrix"], fmt="d",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Proportion"},
    )
    ax9.set_title(
        "Confusion Matrix\n(Oracle = rows, Agent = cols, normalised by row)",
        fontweight="bold", fontsize=9,
    )
    ax9.set_xlabel("Agent Action", fontsize=9)
    ax9.set_ylabel("Oracle Action", fontsize=9)
    ax9.tick_params(axis="x", rotation=35, labelsize=7)
    ax9.tick_params(axis="y", rotation=0,  labelsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[+] Evaluation plot saved: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────


def main(
    model_path:       str   = MODEL_PATH,
    scaler_path:      str   = SCALER_PATH,
    csv_path:         str   = CSV_FILE,
    n_episodes:       int   = 100,
    fixed_age:        Optional[int]   = None,
    fixed_hour:       Optional[float] = None,
    fixed_occupation: Optional[str]   = None,
    plot:             bool  = True,
    save_path:        str   = PLOT_PATH,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline.

    Args:
        model_path:       Path to saved DQN model (without .zip extension)
        scaler_path:      Path to saved StandardScaler pickle
        csv_path:         Path to focus_dataset.csv
        n_episodes:       Number of evaluation episodes to run
        fixed_age:        Fix user age (None = random per episode)
        fixed_hour:       Fix hour of day (None = random per episode)
        fixed_occupation: Fix occupation (None = random per episode)
        plot:             Whether to generate the visualisation dashboard
        save_path:        Output path for the PNG plot

    Returns:
        Dictionary of all computed metrics.
    """
    print(f"[*] Loading model   : {model_path}.zip")
    model = DQN.load(model_path)

    print(f"[*] Loading scaler  : {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"[*] Loading dataset : {csv_path}")
    df = pd.read_csv(csv_path)
    df = repair_session_data(df)
    df[CONT_COLS] = scaler.transform(df[CONT_COLS])
    df.fillna(0, inplace=True)

    randomize = (fixed_age is None)
    env = FocusManagementEnv(
        df,
        randomize_context = randomize,
        fixed_age         = fixed_age        or 25,
        fixed_hour        = fixed_hour       or 9.0,
        fixed_occupation  = fixed_occupation or "office",
    )

    print(f"\n[*] Evaluating for {n_episodes} episodes "
          f"({'randomized' if randomize else 'fixed'} context)...\n")

    episodes = run_evaluation(model, env, n_episodes=n_episodes)
    metrics  = compute_metrics(episodes)

    print_report(metrics)

    if plot:
        plot_results(metrics, save_path=save_path)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate FocusManagement DQN Agent v2"
    )
    parser.add_argument("--model",      default=MODEL_PATH,  help="Model path (no .zip)")
    parser.add_argument("--scaler",     default=SCALER_PATH, help="Scaler pickle path")
    parser.add_argument("--csv",        default=CSV_FILE,    help="Dataset CSV path")
    parser.add_argument("--episodes",   type=int,   default=100,    help="Number of eval episodes")
    parser.add_argument("--age",        type=int,   default=None,   help="Fixed user age")
    parser.add_argument("--hour",       type=float, default=None,   help="Fixed hour of day (0-23)")
    parser.add_argument("--occupation", type=str,   default=None,
                        choices=list(OCCUPATIONS.keys()), help="Fixed occupation")
    parser.add_argument("--no-plot",    action="store_true",         help="Skip visualisation")
    parser.add_argument("--output",     default=PLOT_PATH,           help="Output plot path")

    args = parser.parse_args()

    main(
        model_path       = args.model,
        scaler_path      = args.scaler,
        csv_path         = args.csv,
        n_episodes       = args.episodes,
        fixed_age        = args.age,
        fixed_hour       = args.hour,
        fixed_occupation = args.occupation,
        plot             = not args.no_plot,
        save_path        = args.output,
    )
