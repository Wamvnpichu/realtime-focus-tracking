# Focus Tracker AI (v2.1)

A real-time desktop app using computer vision and Reinforcement Learning (RL) to track focus and provide intelligent productivity nudges.

## Features (v2.1)

- **Standalone GUI:** Packaged Windows executable (`FocusTracker.exe`) with camera feed and system tray support.
- **Glasses-Friendly AI:** Lenient vision thresholds to prevent false positives for users with glasses.
- **Smooth State Transitions:**
  - `Focused`: 0-3s of distraction.
  - `Early Warning`: 3-9s of distraction.
  - `Distracted`: >9s of distraction.
- **Smart Nudging:**
  - *Soft Nudge*: Unobtrusive notification after 9s of distraction.
  - *Hard Nudge*: Full-screen lock requiring acknowledgment after 90s.
- **Note-Taking Mode:** Looking down with eyes open is treated as "Focused" for up to 30s.
- **Built-in Music:** Integrated offline LoFi player.

## Usage

1. Open `dist/FocusTracker/`.
2. Run `FocusTracker.exe`.
3. *(Optional)* Add a Gemini API key in `llm_agent.py` for dynamic AI messages.
4. Click 'X' to minimize to tray, or right-click the tray icon to quit.

## Structure

- `gui_app.py`: UI & camera feed (Tkinter + OpenCV).
- `focus_env.py`: RL environment & nudge rules.
- `vision_utils.py`: Extracts head pose and eye metrics via Mediapipe.
- `llm_agent.py`: Gemini LLM integration.
- `build_exe.py`: PyInstaller build script.