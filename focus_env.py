"""
focus_env.py - FocusManagementEnv v2
====================================
Window-based Reinforcement Learning Environment for Focus Monitoring.

Architecture Details:
  1. State = sliding window (W frames) -> statistical + trend features (early detection).
  2. Context features in state: age, hour, occupation, work duration.
  3. Action space: 6 actions (0: Do Nothing, 1: Soft Nudge, etc).
  4. Reward function: context-aware based on user profile and fatigue.

State Dimensions (33 total):
  - Window Stats (Means/Stds): Pitch, Yaw, Roll, EAR, MAR, Brow, Delta_Yaw, Delta_Pitch, Delta_EAR, Gaze
  - Ratios & Trends: Person detected ratio, phone ratio, distraction ratio.
  - Context Features: Hour (sin/cos), Age, Work Duration.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE: int = 10            # Number of frames per state (~3.3 s at 3 fps)
FPS: float = 3.0                 # Assumed video frame rate
MAX_CONSECUTIVE: float = 300.0   # Max observed consecutive_frames (repaired data)
MAX_WORK_SEC: float = 14400.0    # 4 hours → normalise work duration (from timestamp)

# Continuous feature columns in the CSV (used for scaler + window stats)
CONT_COLS: List[str] = [
    "head_pitch", "head_yaw", "head_roll",
    "ear_score",  "mar_score", "brow_dist",
    "delta_yaw", "delta_pitch", "delta_ear", "gaze_ratio"
]

# Columns used for trend (slope) computation — core early-detection signals
TREND_COLS: List[str] = ["head_pitch", "head_yaw", "ear_score"]

# State dimension:
#   10 means + 10 stds + 1 person_ratio + 1 phone_ratio
#   + 3 trends + 1 distraction_ratio + 1 consec_max_norm + 1 fatigue_score
#   + 5 context
STATE_DIM: int = len(CONT_COLS) * 2 + 2 + len(TREND_COLS) + 3 + 5  # = 33

# ──────────────────────────────────────────────────────────────────────────────
# ACTION SPACE
# ──────────────────────────────────────────────────────────────────────────────

ACTION_NAMES: Dict[int, str] = {
    0: "Do Nothing",
    1: "Gentle Visual Reminder",
    2: "Sound Alert",
    3: "Suggest Short Break",
    4: "Dim Screen",
    5: "Play Focus Music",
}
N_ACTIONS: int = len(ACTION_NAMES)

# ──────────────────────────────────────────────────────────────────────────────
# OCCUPATION ENCODING
# ──────────────────────────────────────────────────────────────────────────────

OCCUPATIONS: Dict[str, float] = {
    "student":   0.00,
    "office":    0.25,
    "creative":  0.50,
    "technical": 0.75,
    "other":     1.00,
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def hour_to_cyclical(hour: float) -> Tuple[float, float]:
    """
    Encode hour-of-day (0–23) as (sin, cos) pair.
    Cyclical encoding preserves continuity at the 23h → 0h boundary.
    """
    angle = 2.0 * np.pi * float(hour) / 24.0
    return float(np.sin(angle)), float(np.cos(angle))


def encode_context(
    age: int,
    hour: float,
    occupation: str,
    work_duration_sec: float,
) -> np.ndarray:
    """
    Encode runtime contextual metadata into a normalised 5-dim vector.

    Args:
        age:               User age in years (18–65)
        hour:              Hour of day (0–23)
        occupation:        One of OCCUPATIONS keys
        work_duration_sec: Seconds elapsed in current session (from timestamp col)

    Returns:
        np.ndarray shape (5,): [hour_sin, hour_cos, age_norm, occ_code, dur_norm]
    """
    h_sin, h_cos = hour_to_cyclical(hour)
    age_norm = float(np.clip((age - 18) / (65 - 18), 0.0, 1.0))
    occ_code = float(OCCUPATIONS.get(occupation, OCCUPATIONS["other"]))
    dur_norm = float(np.clip(work_duration_sec / MAX_WORK_SEC, 0.0, 1.0))
    return np.array([h_sin, h_cos, age_norm, occ_code, dur_norm], dtype=np.float32)


def extract_window_features(window: pd.DataFrame) -> np.ndarray:
    """
    Extract a feature vector from a sliding window of frames.
    Implements Focus-and-Suppress logic (Idea A): if head is very still,
    we amplify the attention (weight) given to eye features (EAR, Gaze)
    while suppressing head features, because when static, blinking is the primary distraction signal.
    """
    n = len(window)
    feats: List[float] = []
    
    # FOCUS-AND-SUPPRESS ATTENTION
    std_yaw = window["delta_yaw"].std(ddof=0)
    std_pitch = window["delta_pitch"].std(ddof=0)
    head_motion = (std_yaw + std_pitch) / 2.0
    
    eye_weight = 1.3 if head_motion < 0.1 else 1.0
    head_weight = 0.7 if head_motion < 0.1 else 1.0

    # 1. Means
    means = window[CONT_COLS].mean().values.copy()
    means[0:3] *= head_weight
    means[3:6] *= eye_weight
    means[6:8] *= head_weight
    means[8:10] *= eye_weight
    feats.extend(means.tolist())

    # 2. Within-window standard deviations
    stds = window[CONT_COLS].std(ddof=0).fillna(0.0).values.copy()
    stds[0:3] *= head_weight
    stds[3:6] *= eye_weight
    stds[6:8] *= head_weight
    stds[8:10] *= eye_weight
    feats.extend(stds.tolist())

    # 3. Binary presence ratios
    feats.append(float(window["person_detected"].mean()))
    feats.append(float(window["phone_count"].clip(upper=1).mean()))

    # 4. Trend slopes via least-squares linear regression
    x = np.arange(n, dtype=np.float64)
    for col in TREND_COLS:
        y = window[col].values.astype(np.float64)
        slope = float(np.polyfit(x, y, 1)[0]) if n > 1 else 0.0
        if 'pitch' in col or 'yaw' in col:
            slope *= head_weight
        elif 'ear' in col:
            slope *= eye_weight
        feats.append(slope)

    # 5. Distraction ratio in window
    feats.append(float(window["is_distracted_label"].mean()))

    # 6. Max consecutive frames (normalised)
    feats.append(float(window["consecutive_frames"].max() / MAX_CONSECUTIVE))
    
    # 7. Fatigue score (Idea D)
    feats.append(float(window["fatigue_score"].iloc[-1]))

    return np.array(feats, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTION
# ──────────────────────────────────────────────────────────────────────────────


def compute_reward(
    action: int,
    distraction_ratio: float,
    consec_max_norm: float,
    pitch_trend: float,
    yaw_trend: float,
    ear_trend: float,
    age: int,
    hour: float,
    work_duration_min: float,
    spam_count: int,
    fatigue_score: float,
    person_ratio: float,
) -> float:
    early_signal: bool = False
    consecutive_frames = consec_max_norm * MAX_CONSECUTIVE
    
    if consecutive_frames < 50:
        if pitch_trend > 1.00 or ear_trend < -0.02:
            early_signal = True
            
    person_missing = person_ratio < 0.5
    age_factor = np.clip((age - 18) / (65 - 18), 0.0, 1.0)
    gentle_bonus = 0.05 + 0.10 * age_factor
    hard_penalty = -(0.05 + 0.10 * age_factor)
    is_night = hour >= 22 or hour <= 6
    context_scale = 1.10 if is_night else 0.90
    break_bonus = 0.10 if work_duration_min > 45.0 else 0.0
    severity = min(1.0, consecutive_frames / 150.0)

    if person_missing:
        core = {0: -2.00, 1: +0.20, 2: +2.00, 3: +0.50, 4: -0.20, 5: -0.20}
    elif consecutive_frames < 50 and not early_signal:
        if fatigue_score > 0.8:
            core = {0: +1.00, 1: -0.50, 2: -1.00, 3: +2.00, 4: -0.50, 5: +0.50}
        else:
            core = {0: +1.00, 1: -0.50, 2: -1.00, 3: -0.50, 4: -0.50, 5: -0.50}
    elif early_signal or (50 <= consecutive_frames < 100):
        core = {0: -1.50, 1: +1.50, 2: -0.50, 3: +0.80, 4: +0.80, 5: -0.50}
    else:
        # SEVERE DISTRACTION
        core = {0: -2.50, 1: +0.50, 2: +2.00, 3: +1.50, 4: +0.50, 5: -1.00}

    base_reward = core[action]
    
    if action == 2:
        base_reward += hard_penalty
    elif action in (1, 4, 5):
        base_reward += gentle_bonus
    elif action == 3:
        base_reward += gentle_bonus * 0.5

    base_reward *= context_scale

    if spam_count >= 2 and action > 0:
        penalty = min(2.0, 0.40 * (spam_count - 1))
        base_reward -= penalty

    return float(np.clip(base_reward, -3.0, 3.0))


# ──────────────────────────────────────────────────────────────────────────────
# GYMNASIUM ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────


class FocusManagementEnv(gym.Env):
    """
    FocusManagementEnv v2 — Window-based Focus Monitoring RL Environment.

    Observation space (24-dim Box):
        Window-based physiological features (19-dim) + contextual metadata (5-dim).
        See module docstring for full breakdown.

    Action space (Discrete(6)):
        0: Do Nothing
        1: Gentle Visual Reminder
        2: Sound Alert
        3: Suggest Short Break
        4: Dim Screen
        5: Play Focus Music

    Episode structure:
        One episode = one session (a continuous recording segment).
        Step  = advance the sliding window by one frame.
        Reset = choose a random session; sample/fix context metadata.

    Context injection (per episode, NOT per frame):
        Training:   age, hour, occupation sampled randomly for diversity.
        Evaluation: fixed values supplied via constructor args.
        work_duration derived from the `timestamp` CSV column (real data).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        randomize_context: bool = True,
        fixed_age: int = 25,
        fixed_hour: float = 9.0,
        fixed_occupation: str = "office",
    ) -> None:
        """
        Args:
            df:                 Preprocessed DataFrame (CONT_COLS already normalised).
            randomize_context:  True during training; False during evaluation.
            fixed_age:          Age used when randomize_context=False.
            fixed_hour:         Hour used when randomize_context=False.
            fixed_occupation:   Occupation used when randomize_context=False.
        """
        super().__init__()

        self.df = df
        self.randomize_context = randomize_context
        self.fixed_age = fixed_age
        self.fixed_hour = fixed_hour
        self.fixed_occupation = fixed_occupation

        # Group into per-session DataFrames (session_id used ONLY here)
        self.sessions: List[pd.DataFrame] = [
            grp.reset_index(drop=True)
            for _, grp in df.groupby("session_id", sort=False)
        ]

        # Spaces
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )

        # Episode state (initialised in reset)
        self.current_session: Optional[pd.DataFrame] = None
        self.current_step: int = 0
        self.context: Dict[str, Any] = {}
        self.spam_count: int = 0
        self.last_action: int = 0

    # ── Private helpers ───────────────────────────────────────────────────

    def _sample_context(self) -> Dict[str, Any]:
        """Sample or return fixed context for one episode."""
        if self.randomize_context:
            age = int(np.random.randint(18, 66))
            hour = float(np.random.uniform(0.0, 24.0))
            occupation = str(
                np.random.choice(list(OCCUPATIONS.keys()))
            )
        else:
            age = self.fixed_age
            hour = self.fixed_hour
            occupation = self.fixed_occupation
        return {"age": age, "hour": hour, "occupation": occupation}

    def _current_window(self) -> pd.DataFrame:
        """Return the W-frame window ending at current_step."""
        start = max(0, self.current_step - WINDOW_SIZE + 1)
        return self.current_session.iloc[start: self.current_step + 1]

    def _get_observation(self) -> np.ndarray:
        """Concatenate window features and context features into one state vector."""
        window = self._current_window()
        window_feats = extract_window_features(window)

        # Derive work_duration from the raw timestamp column (seconds)
        work_sec = float(
            self.current_session.iloc[self.current_step]["timestamp"]
        )
        context_feats = encode_context(
            self.context["age"],
            self.context["hour"],
            self.context["occupation"],
            work_sec,
        )
        return np.concatenate([window_feats, context_feats]).astype(np.float32)

    def _window_stats(self) -> Dict[str, float]:
        """Compute reward-relevant statistics for the current window."""
        window = self._current_window()
        n = len(window)
        x = np.arange(n, dtype=np.float64)

        def _slope(col: str) -> float:
            y = window[col].values.astype(np.float64)
            return float(np.polyfit(x, y, 1)[0]) if n > 1 else 0.0

        return {
            "distraction_ratio": float(window["is_distracted_label"].mean()),
            "consec_max_norm":   float(
                window["consecutive_frames"].max() / MAX_CONSECUTIVE
            ),
            "pitch_trend": _slope("head_pitch"),
            "yaw_trend":   _slope("head_yaw"),
            "ear_trend":   _slope("ear_score"),
            "fatigue_score": float(window["fatigue_score"].iloc[-1]),
            "person_ratio": float(window["person_detected"].mean()),
        }

    # ── Gym interface ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        idx = int(np.random.randint(len(self.sessions)))
        self.current_session = self.sessions[idx]
        # Start after the first full window
        self.current_step = WINDOW_SIZE - 1
        self.spam_count = 0
        self.last_action = 0
        self.context = self._sample_context()

        return self._get_observation(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        stats = self._window_stats()
        ctx = self.context
        work_sec = float(
            self.current_session.iloc[self.current_step]["timestamp"]
        )
        work_duration_min = work_sec / 60.0

        # ── Spam tracking ─────────────────────────────────────────────────
        if action > 0:
            self.spam_count += 1
        else:
            self.spam_count = 0
        self.last_action = int(action)

        # ── Reward ────────────────────────────────────────────────────────
        reward = compute_reward(
            action            = int(action),
            distraction_ratio = stats["distraction_ratio"],
            consec_max_norm   = stats["consec_max_norm"],
            pitch_trend       = stats["pitch_trend"],
            yaw_trend         = stats["yaw_trend"],
            ear_trend         = stats["ear_trend"],
            age               = ctx["age"],
            hour              = ctx["hour"],
            work_duration_min = work_duration_min,
            spam_count        = self.spam_count,
            fatigue_score     = stats["fatigue_score"],
            person_ratio      = stats["person_ratio"],
        )

        # ── Advance ───────────────────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= len(self.current_session) - 1
        obs = (
            self._get_observation()
            if not terminated
            else np.zeros(STATE_DIM, dtype=np.float32)
        )

        info: Dict[str, Any] = {
            "action":            int(action),
            "action_name":       ACTION_NAMES[int(action)],
            "reward":            reward,
            "distraction_ratio": stats["distraction_ratio"],
            "consec_max_norm":   stats["consec_max_norm"],
            "pitch_trend":       stats["pitch_trend"],
            "ear_trend":         stats["ear_trend"],
            "early_signal":      (
                stats["distraction_ratio"] < 0.35 and (
                    stats["pitch_trend"]   >  1.00 or
                    stats["ear_trend"]     < -0.02 or
                    stats["consec_max_norm"] > 0.10
                )
            ),
            "age":               ctx["age"],
            "hour":              ctx["hour"],
            "occupation":        ctx["occupation"],
            "work_duration_min": work_duration_min,
            "spam_count":        self.spam_count,
        }

        return obs, reward, terminated, False, info


# ──────────────────────────────────────────────────────────────────────────────
# REAL-TIME INFERENCE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

from collections import deque
import pandas as pd
from stable_baselines3 import DQN
import pickle
import time

class FocusAgentInference:
    """
    Wrapper that encapsulates all RL model logic for real-time inference (e.g., webcam demo).
    It maintains the sliding window, applies scaling, extracts features, and runs the DQN model.
    """
    def __init__(self, model_path: str, scaler_path: str):
        self.rl_model = DQN.load(model_path)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.window_buffer = deque(maxlen=WINDOW_SIZE)
        self.consecutive_frames = 0
        self.consecutive_looking_down = 0
        self.start_time = time.time()
        
        self.last_yaw = 0.0
        self.last_pitch = 0.0
        self.last_ear = 0.0
        
    def process_frame(self, head_pitch, head_yaw, head_roll, ear_score, mar_score, brow_dist, person_detected, phone_count, user_age, user_hour, user_occupation, gaze_ratio):
        """
        Takes raw metrics from a single frame, updates internal state, and returns the agent's action and status.
        """
        # Track looking down duration
        if head_pitch > 20 and abs(head_yaw) < 20 and ear_score >= 0.22:
            self.consecutive_looking_down += 1
        else:
            self.consecutive_looking_down = 0

        # 1. Distraction Logic
        # Exception for note-taking: looking down (positive pitch), not looking away (low yaw), eyes open, no phone, and not stuck looking down for > 30s
        is_note_taking = (head_pitch > 20) and (abs(head_yaw) < 20) and (phone_count == 0) and (ear_score >= 0.20) and (self.consecutive_looking_down < 300)
        
        is_distracted = 1 if (
            (ear_score < 0.20) or
            (abs(head_yaw) > 40) or
            (abs(head_pitch) > 30 and not is_note_taking) or
            (head_pitch < -25) or  # looking up
            (phone_count > 0) or
            (person_detected == 0)
        ) else 0
        
        if is_distracted:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0
            
        # 2. Pack Features
        # Compute deltas
        delta_yaw = head_yaw - self.last_yaw
        delta_pitch = head_pitch - self.last_pitch
        delta_ear = ear_score - self.last_ear
        
        self.last_yaw = head_yaw
        self.last_pitch = head_pitch
        self.last_ear = ear_score
        
        # Calculate Fatigue
        session_elapsed_sec = time.time() - self.start_time
        norm_time = min(1.0, session_elapsed_sec / 3600.0) # max 1 hour
        fatigue_score = min(1.0, norm_time * (0.4 - min(ear_score, 0.4)) * 2.5)
        
        metrics = {
            'head_pitch': head_pitch,
            'head_yaw': head_yaw,
            'head_roll': head_roll,
            'ear_score': ear_score,
            'mar_score': mar_score,
            'brow_dist': brow_dist,
            'delta_yaw': delta_yaw,
            'delta_pitch': delta_pitch,
            'delta_ear': delta_ear,
            'gaze_ratio': gaze_ratio,
            'person_detected': person_detected,
            'phone_count': phone_count,
            'consecutive_frames': self.consecutive_frames,
            'is_distracted_label': is_distracted,
            'fatigue_score': fatigue_score,
            'age': user_age,
            'hour': user_hour,
            'occupation': user_occupation
        }
        
        self.window_buffer.append(metrics)
        
        # Default empty action if window is not full
        action_name = ""
        dist_ratio = is_distracted
        
        # 3. Agent Prediction
        if len(self.window_buffer) == WINDOW_SIZE:
            df_window = pd.DataFrame(list(self.window_buffer))
            df_scaled = df_window.copy()
            df_scaled[CONT_COLS] = self.scaler.transform(df_scaled[CONT_COLS])
            
            win_features = extract_window_features(df_scaled)
            
            session_elapsed_sec = time.time() - self.start_time
            ctx_features = encode_context(user_age, user_hour, user_occupation, session_elapsed_sec)
            
            state = np.concatenate([win_features, ctx_features])
            
            action, _ = self.rl_model.predict(state, deterministic=True)
            action_name = ACTION_NAMES[int(action)]
            
            # OVERRIDE: Prevent noisy interventions during perfect focus
            if self.consecutive_frames < 50:
                # Only allow break suggestion if heavily fatigued
                if fatigue_score > 0.8 and action_name == "Suggest Short Break":
                    pass
                else:
                    action_name = "Do Nothing"
            
            # FALLBACK OVERRIDE: Force actions if distracted for too long
            if self.consecutive_frames >= 900:
                action_name = "Sound Alert"
            elif self.consecutive_frames >= 90 and action_name == "Do Nothing":
                action_name = "Gentle Visual Reminder"
                
            # DOWNGRADE OVERRIDE: Prevent hard nudges from triggering too quickly
            if action_name in ["Sound Alert", "Suggest Short Break"] and self.consecutive_frames < 900:
                if self.consecutive_frames >= 90:
                    action_name = "Gentle Visual Reminder"
                else:
                    action_name = "Do Nothing"
            
            dist_ratio = df_window['is_distracted_label'].mean()
            
        return metrics, dist_ratio, action_name
