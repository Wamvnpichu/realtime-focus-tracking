"""
focus_env.py — FocusManagementEnv v2
=====================================
Window-based Reinforcement Learning Environment for Focus Monitoring.

Kiến trúc mới (theo góp ý giảng viên):
  1. State = sliding window W frames → statistical + trend features (phát hiện sớm)
  2. Context features VÀO state: giờ, tuổi, nghề nghiệp, thời gian làm việc
  3. Action space mở rộng: 6 actions (từ 3)
  4. Reward function context-aware: age, time-of-day, work-duration, early-detection
  5. session_id / timestamp KHÔNG vào state (chỉ dùng cho grouping & work_duration)

State dimensions (24 total):
  Window features (19):
    [0:6]   mean của: head_pitch, head_yaw, head_roll, ear_score, mar_score, brow_dist
    [6:12]  std  của: head_pitch, head_yaw, head_roll, ear_score, mar_score, brow_dist
    [12]    person_detected_ratio   — tỷ lệ frame có người trong window
    [13]    phone_presence_ratio    — tỷ lệ frame có điện thoại trong window
    [14]    pitch_trend             — slope of head_pitch  (phát hiện sớm: cúi đầu)
    [15]    yaw_trend               — slope of head_yaw    (phát hiện sớm: quay đầu)
    [16]    ear_trend               — slope of ear_score   (phát hiện sớm: mắt nhắm)
    [17]    distraction_ratio       — % frames distracted trong window
    [18]    consec_max_norm         — max consecutive_frames / MAX_CONSECUTIVE
  Context features (5):
    [19]    hour_sin                — sin(2π·hour/24)
    [20]    hour_cos                — cos(2π·hour/24)
    [21]    age_norm                — (age - 18) / (65 - 18)
    [22]    occupation_code         — 0.0..1.0
    [23]    work_duration_norm      — timestamp / MAX_WORK_SEC

Action space (6):
    0: Do Nothing
    1: Gentle Visual Reminder
    2: Sound Alert
    3: Suggest Short Break
    4: Dim Screen
    5: Play Focus Music
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
MAX_CONSECUTIVE: float = 6000.0  # Max observed consecutive_frames (repaired data)
MAX_WORK_SEC: float = 3600.0     # 60 min → normalise work duration (from timestamp)

# Continuous feature columns in the CSV (used for scaler + window stats)
CONT_COLS: List[str] = [
    "head_pitch", "head_yaw", "head_roll",
    "ear_score",  "mar_score", "brow_dist",
]

# Columns used for trend (slope) computation — core early-detection signals
TREND_COLS: List[str] = ["head_pitch", "head_yaw", "ear_score"]

# State dimension:
#   6 means + 6 stds + 1 person_ratio + 1 phone_ratio
#   + 3 trends + 1 distraction_ratio + 1 consec_max_norm
#   + 5 context
STATE_DIM: int = len(CONT_COLS) * 2 + 2 + len(TREND_COLS) + 2 + 5  # = 24

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
    Extract a 19-dim feature vector from a sliding window of frames.

    Features:
        Means (6):            temporal average of each continuous feature
        Stds  (6):            temporal variability → instability signal
        person_ratio (1):     proportion of frames with person detected
        phone_ratio  (1):     proportion of frames with phone present
        pitch_trend  (1):     linear slope of head_pitch  → looking-down trend
        yaw_trend    (1):     linear slope of head_yaw   → drifting-away trend
        ear_trend    (1):     linear slope of ear_score  → eyes-closing trend
        distraction_ratio (1): fraction of distracted frames in window
        consec_max_norm   (1): max consecutive_frames / MAX_CONSECUTIVE

    Trend slopes are the KEY early-detection features: a negative ear_trend
    or positive pitch_trend signals an impending distraction BEFORE the label
    flips to 1, allowing the agent to act proactively.
    """
    n = len(window)
    feats: List[float] = []

    # 1. Means
    feats.extend(window[CONT_COLS].mean().values.tolist())

    # 2. Within-window standard deviations (ddof=0 → population std)
    feats.extend(window[CONT_COLS].std(ddof=0).fillna(0.0).values.tolist())

    # 3. Binary presence ratios
    feats.append(float(window["person_detected"].mean()))
    feats.append(float(window["phone_count"].clip(upper=1).mean()))

    # 4. Trend slopes via least-squares linear regression
    x = np.arange(n, dtype=np.float64)
    for col in TREND_COLS:
        y = window[col].values.astype(np.float64)
        slope = float(np.polyfit(x, y, 1)[0]) if n > 1 else 0.0
        feats.append(slope)

    # 5. Distraction ratio in window
    feats.append(float(window["is_distracted_label"].mean()))

    # 6. Max consecutive frames (normalised)
    feats.append(float(window["consecutive_frames"].max() / MAX_CONSECUTIVE))

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
) -> float:
    """
    Context-aware reward function — v3 (balanced for real data distribution).

    Design rationale:
        The dataset is ~48% fully-focused (ratio=0.0) and ~51% fully-distracted
        (ratio=1.0) with only ~1% in transition zones. The old reward function
        gave +1.5 per focused step but -5.0 per severe step, making positive
        total reward mathematically impossible.

        This version balances the magnitudes so a correct oracle achieves
        ~+1.0/step on average across the real data distribution. Penalties for
        wrong actions are small enough that early training exploration doesn't
        cause catastrophic negative reward, while still providing clear gradient
        signal for the agent to learn the correct policy.

    Reward structure:
        R_total = R_core * context_scale - R_spam

    Four distraction zones:
        FOCUSED      distraction_ratio < 0.20  -> reward silence
        EARLY WARNING 0.20-0.45 (or trend)     -> reward proactive soft action
        MODERATE     0.45-0.75                 -> reward intervention
        SEVERE       >= 0.75                   -> reward strong action

    Context modifiers (SMALL — avoid amplifying imbalance):
        context_scale  — time-of-day: 0.90 to 1.10
        age modifier   — small bonus/penalty: max +/-0.15
        break_bonus    — +0.10 for action=3 after long work
        spam_penalty   — -0.20 * excess

    Args:
        action:            Chosen action index (0-5)
        distraction_ratio: Fraction of window frames labelled distracted (0-1)
        consec_max_norm:   Max consecutive_frames / MAX_CONSECUTIVE
        pitch_trend:       Head-pitch slope (positive = looking down)
        yaw_trend:         Head-yaw slope (head drifting sideways)
        ear_trend:         Eye-aspect-ratio slope (negative = eyes closing)
        age:               User age in years
        hour:              Hour of day (0-23)
        work_duration_min: Minutes elapsed in session
        spam_count:        Consecutive steps where agent chose action > 0

    Returns:
        float: reward value clipped to [-3, +3]
    """

    # -- 1. Time-of-day scale (SMALL range: 0.90 - 1.10) ------------------
    if 6 <= hour < 12:
        context_scale = 1.10    # Morning: slightly higher stakes
    elif 12 <= hour < 17:
        context_scale = 1.00    # Afternoon: baseline
    elif 17 <= hour < 22:
        context_scale = 0.95    # Evening: slightly more tolerant
    else:
        context_scale = 0.90    # Night: most tolerant

    # -- 2. Age-based modifier (SMALL: max +/-0.15) ------------------------
    if age < 30:
        hard_penalty  =  0.00
        gentle_bonus  =  0.00
    elif age < 45:
        hard_penalty  = -0.05
        gentle_bonus  =  0.05
    elif age < 60:
        hard_penalty  = -0.10
        gentle_bonus  =  0.10
    else:
        hard_penalty  = -0.15
        gentle_bonus  =  0.15

    # -- 3. Work-duration break bonus (SMALL) ------------------------------
    break_bonus = float(np.clip(work_duration_min / 120.0, 0.0, 1.0)) * 0.10

    # -- 4. Early-distraction detection signal -----------------------------
    # DISABLED: With current dataset (~0.6% transition zone), the trend-based
    # thresholds (ear_trend < -0.02, pitch_trend > 1.0) fire falsely 26.9%
    # of the time in focused windows, causing action 0 to be penalised
    # instead of rewarded. Early warning now only uses distraction_ratio.
    early_signal: bool = False

    # -- 5. Severity factor for severe zone --------------------------------
    # Use consec_max_norm to scale rewards in severe zone: higher consecutive
    # distraction = more urgency for strong action.
    severity = float(np.clip(consec_max_norm, 0.0, 1.0))

    # -- 6. Core reward table (BALANCED magnitudes) ------------------------
    #
    # Design: correct action gives +1.0 base, wrong action gives -0.5 to -1.5.
    # With ~48% focused (+1.0 each) and ~51% severe (+1.0 each), a perfect
    # oracle achieves ~+1.0/step average. An agent doing random actions gets
    # ~-0.3/step, providing clear learning signal without catastrophic negatives.

    if distraction_ratio < 0.20:
        # -- FOCUSED -------------------------------------------------------
        # User is productive. Wrong actions should be mildly penalised, not
        # catastrophically — the agent needs room to explore at training start.
        core = {
            0: +1.00,   # Correct: stay silent
            1: -0.10,   # Gentle visual: barely noticeable disturbance
            2: -0.30,   # Sound alert: most disruptive
            3: -0.05,   # Suggest break: harmless
            4: -0.10,   # Screen dim: minor
            5: +0.05,   # Focus music: arguably helpful even when focused
        }

    elif early_signal or (0.20 <= distraction_ratio < 0.45):
        # -- EARLY WARNING -------------------------------------------------
        core = {
            0: -0.50,              # Missing early intervention
            1: +1.00,              # Gentle visual: good
            2: -0.30,              # Sound alert: overreaction
            3: +0.80 + break_bonus, # Break: good for long sessions
            4: +0.80,              # Screen dim: non-disruptive
            5: +1.00,              # Focus music: excellent early action
        }

    elif 0.45 <= distraction_ratio < 0.75:
        # -- MODERATE DISTRACTION ------------------------------------------
        core = {
            0: -0.80,              # Too passive
            1: +0.70,              # Still appropriate
            2: +0.60,              # Reasonable escalation
            3: +1.00 + break_bonus, # Break: very valuable
            4: +0.50,              # Helpful
            5: +0.70,              # Good
        }

    else:
        # -- SEVERE DISTRACTION (>= 0.75) ----------------------------------
        # Scale reward by severity: more consecutive = stronger preference
        # for hard interventions. If severity is low, gentle is better.
        base_correct  = 0.20 + 1.10 * severity   # 0.20 to 1.30
        base_gentle   = 0.80 - 0.60 * severity   # 0.80 to 0.20
        base_nothing  = -(0.80 + 0.70 * severity) # -0.80 to -1.50

        core = {
            0: base_nothing,                        # Critical failure
            1: base_gentle,                         # Too gentle but not terrible
            2: base_correct,                        # Sound alert: appropriate
            3: base_correct * 0.90 + break_bonus,   # Break: also very appropriate
            4: base_gentle,                         # Too gentle
            5: base_gentle,                         # Too gentle
        }

    base_reward = core[action]

    # -- 7. Apply age modifier (small) -------------------------------------
    if action == 2:               # Sound Alert
        base_reward += hard_penalty
    elif action in (1, 4, 5):    # Gentle actions
        base_reward += gentle_bonus
    elif action == 3:             # Break suggestion
        base_reward += gentle_bonus * 0.5

    # -- 8. Apply time-of-day scale ----------------------------------------
    base_reward *= context_scale

    # -- 9. Anti-spam penalty (lighter & capped) ---------------------------
    if spam_count >= 3 and action > 0:
        # Cap penalty at -1.0 so it doesn't completely destroy the reward signal
        # during random exploration phases.
        penalty = min(1.0, 0.20 * (spam_count - 2))
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
        self.start_time = time.time()
        
    def process_frame(self, head_pitch, head_yaw, head_roll, ear_score, mar_score, brow_dist, person_detected, phone_count, user_age, user_hour, user_occupation):
        """
        Takes raw metrics from a single frame, updates internal state, and returns the agent's action and status.
        """
        # 1. Distraction Logic
        is_distracted = 1 if (
            (ear_score < 0.22) or
            (abs(head_yaw) > 30) or
            (abs(head_pitch) > 25) or
            (phone_count > 0) or
            (person_detected == 0)
        ) else 0
        
        if is_distracted:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0
            
        # 2. Pack Features
        metrics = {
            'head_pitch': head_pitch,
            'head_yaw': head_yaw,
            'head_roll': head_roll,
            'ear_score': ear_score,
            'mar_score': mar_score,
            'brow_dist': brow_dist,
            'person_detected': person_detected,
            'phone_count': phone_count,
            'consecutive_frames': self.consecutive_frames,
            'is_distracted_label': is_distracted
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
            
            dist_ratio = df_window['is_distracted_label'].mean()
            
        return metrics, dist_ratio, action_name
