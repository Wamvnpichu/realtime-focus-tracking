import argparse
import cv2
import time
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from stable_baselines3 import DQN
from collections import deque
import datetime
import os
import sys

try:
    from focus_env import (
        CONT_COLS, WINDOW_SIZE, STATE_DIM, ACTION_NAMES, 
        extract_window_features, encode_context, FocusAgentInference
    )
except ImportError:
    print("Error: Could not import from focus_env.py. Ensure it is in the same directory.")
    sys.exit(1)

# Feature Extraction Functions
def estimate_head_pose(face_landmarks, frame_shape):
    h, w = frame_shape[:2]
    
    # 3D model points (standard face model)
    model_points = np.array([
        [0.0, 0.0, 0.0],         # Nose tip
        [0.0, -330.0, -65.0],    # Chin
        [-225.0, 170.0, -135.0], # Left eye corner
        [225.0, 170.0, -135.0],  # Right eye corner
        [-150.0, -150.0, -125.0],# Left mouth corner
        [150.0, -150.0, -125.0]  # Right mouth corner
    ], dtype=np.float64)
    
    # 2D image points from landmarks
    # Indices: nose_tip=1, chin=152, left_eye=263, right_eye=33, left_mouth=287, right_mouth=57
    landmark_indices = [1, 152, 263, 33, 287, 57]
    image_points = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h]
        for i in landmark_indices
    ], dtype=np.float64)
    
    # Camera matrix
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.decomposeProjectionMatrix(
        np.hstack((rotation_matrix, translation_vector.reshape(3, 1)))
    )[6]
    
    pitch = float(angles[0][0])
    yaw = float(angles[1][0])
    roll = float(angles[2][0])
    return pitch, yaw, roll


def calculate_ear(face_landmarks):
    # Right eye landmarks
    right_eye = [33, 160, 158, 133, 153, 144]
    # Left eye landmarks  
    left_eye = [263, 387, 385, 362, 380, 373]
    
    def eye_aspect_ratio(eye_indices):
        p = [face_landmarks[i] for i in eye_indices]
        # Vertical distances
        v1 = ((p[1].x - p[5].x)**2 + (p[1].y - p[5].y)**2)**0.5
        v2 = ((p[2].x - p[4].x)**2 + (p[2].y - p[4].y)**2)**0.5
        # Horizontal distance
        h = ((p[0].x - p[3].x)**2 + (p[0].y - p[3].y)**2)**0.5
        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0


def calculate_mar(face_landmarks):
    # Mouth landmarks
    mouth = [61, 291, 13, 14, 78, 308]
    p = [face_landmarks[i] for i in mouth]
    
    # Vertical: upper lip to lower lip
    v = ((p[2].x - p[3].x)**2 + (p[2].y - p[3].y)**2)**0.5
    # Horizontal: left corner to right corner
    h = ((p[0].x - p[1].x)**2 + (p[0].y - p[1].y)**2)**0.5
    if h == 0:
        return 0.0
    return v / h


def calculate_brow_dist(face_landmarks, frame_shape):
    h, w = frame_shape[:2]
    # Left brow: landmark 105, Left eye: landmark 159
    brow = face_landmarks[105]
    eye = face_landmarks[159]
    dist = ((brow.x - eye.x)**2 * w**2 + (brow.y - eye.y)**2 * h**2)**0.5
    return dist

def draw_ui_overlay(frame, metrics, dist_ratio, action_history, fps, session_time, current_action_name):
    h, w = frame.shape[:2]
    
    # Colors (BGR)
    COLOR_GREEN = (83, 200, 0)       # #00C853
    COLOR_RED = (68, 23, 255)        # #FF1744
    COLOR_ORANGE = (0, 145, 255)     # #FF9100
    COLOR_TEXT = (255, 255, 255)
    COLOR_PANEL = (0, 0, 0)
    
    # Draw semi-transparent panels
    overlay = frame.copy()
    
    # Left sidebar
    cv2.rectangle(overlay, (0, 0), (320, h), COLOR_PANEL, -1)
    # Top bar
    cv2.rectangle(overlay, (320, 0), (w, 40), COLOR_PANEL, -1)
    # Bottom bar
    cv2.rectangle(overlay, (320, h - 50), (w, h), COLOR_PANEL, -1)
    
    # Apply alpha blending
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # ================= TOP BAR =================
    cv2.putText(frame, f"FPS: {fps:.1f}", (340, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    
    # Format session time
    session_str = str(datetime.timedelta(seconds=int(session_time)))
    cv2.putText(frame, f"Session Time: {session_str}", (w - 250, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    
    # ================= LEFT SIDEBAR =================
    y_offset = 40
    cv2.putText(frame, "FOCUS TRACKER AI", (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_TEXT, 2)
    cv2.line(frame, (20, y_offset + 10), (300, y_offset + 10), (100, 100, 100), 1)
    
    y_offset += 40
    cv2.putText(frame, "HEAD POSE", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 25
    cv2.putText(frame, f"Pitch: {metrics['head_pitch']:.1f}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    y_offset += 25
    cv2.putText(frame, f"Yaw:   {metrics['head_yaw']:.1f}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    y_offset += 25
    cv2.putText(frame, f"Roll:  {metrics['head_roll']:.1f}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    y_offset += 40
    cv2.putText(frame, "EYES & MOUTH", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 25
    
    ear = metrics['ear_score']
    ear_status = "Closed" if ear < 0.20 else ("Closing" if ear < 0.25 else "Open")
    ear_color = COLOR_RED if ear_status == "Closed" else (COLOR_ORANGE if ear_status == "Closing" else COLOR_GREEN)
    cv2.putText(frame, f"EAR: {ear:.2f} ({ear_status})", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 1)
    
    y_offset += 25
    cv2.putText(frame, f"MAR: {metrics['mar_score']:.2f}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    y_offset += 40
    cv2.putText(frame, "ENVIRONMENT", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 25
    
    phone_text = "Detected" if metrics['phone_count'] > 0 else "Not detected"
    phone_color = COLOR_RED if metrics['phone_count'] > 0 else COLOR_GREEN
    cv2.putText(frame, f"Phone: {phone_text}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 1)
    
    y_offset += 40
    cv2.line(frame, (20, y_offset), (300, y_offset), (100, 100, 100), 1)
    y_offset += 40
    
    # Calculate window-based distraction ratio for stable UI
    if dist_ratio < 0.20:
        status_text = "FOCUSED"
        status_color = COLOR_GREEN
    elif dist_ratio < 0.45:
        status_text = "EARLY WARNING"
        status_color = COLOR_ORANGE
    else:
        status_text = "DISTRACTED"
        status_color = COLOR_RED
        
    cv2.putText(frame, status_text, (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
    
    y_offset += 35
    cv2.putText(frame, f"Consecutive Frames: {metrics['consecutive_frames']}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    y_offset += 50
    cv2.putText(frame, "AGENT RECOMMENDATION", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 35
    
    if current_action_name:
        cv2.putText(frame, current_action_name, (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_ORANGE, 2)
    else:
        cv2.putText(frame, "Waiting for data...", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

    # ================= BOTTOM BAR =================
    cv2.putText(frame, "Action History:", (340, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    hist_x = 480
    for act in action_history:
        if act == "Do Nothing":
            color = COLOR_GREEN
        elif act in ("Sound Alert", "Suggest Short Break"):
            color = COLOR_RED
        else:
            color = COLOR_ORANGE
        cv2.rectangle(frame, (hist_x, h - 35), (hist_x + 20, h - 15), color, -1)
        cv2.rectangle(frame, (hist_x, h - 35), (hist_x + 20, h - 15), (255, 255, 255), 1)
        hist_x += 30

    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time Focus Tracking Demo")
    parser.add_argument("--model", type=str, default="dqn_focus_agent_v2", help="Path to DQN model (without .zip)")
    parser.add_argument("--scaler", type=str, default="scaler_v2.pkl", help="Path to scaler")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--age", type=int, default=25, help="User age")
    parser.add_argument("--hour", type=int, default=-1, help="Hour override (-1 for system time)")
    parser.add_argument("--occupation", type=str, default="student", help="User occupation")
    parser.add_argument("--yolo", type=str, default="yolov8n.pt", help="YOLO model name")
    args = parser.parse_args()

    print("Loading models...")
    # Load YOLO
    yolo_model = YOLO(args.yolo)
    
    # Load MediaPipe FaceLandmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Load RL Model & Scaler via Inference Wrapper
    model_file = args.model if args.model.endswith('.zip') else args.model + '.zip'
    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} not found.")
        sys.exit(1)
    if not os.path.exists(args.scaler):
        print(f"Error: Scaler file {args.scaler} not found.")
        sys.exit(1)
        
    agent = FocusAgentInference(model_file, args.scaler)

    # Initialize video capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)

    print("Starting webcam feed (Press 'q' to quit)...")
    
    target_fps = 10
    frame_time = 1.0 / target_fps
    
    window_buffer = deque(maxlen=WINDOW_SIZE)
    action_history = deque(maxlen=5)
    
    consecutive_frames = 0
    current_action_name = ""
    
    start_time = time.time()
    last_frame_time = time.time()
    fps_display = 0.0

    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. MediaPipe Inference
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        # 2. YOLO Inference
        results = yolo_model(frame, verbose=False)
        phone_count = sum(1 for box in results[0].boxes if int(box.cls) == 67 and float(box.conf) > 0.5)
        
        # 3. Feature Extraction
        person_detected = 0
        pitch = yaw = roll = ear = mar = brow = 0.0
        
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            person_detected = 1
            pitch, yaw, roll = estimate_head_pose(face_landmarks, frame.shape)
            ear = calculate_ear(face_landmarks)
            mar = calculate_mar(face_landmarks)
            brow = calculate_brow_dist(face_landmarks, frame.shape)
        
        # 4. Agent Inference 
        current_hour = args.hour if args.hour != -1 else datetime.datetime.now().hour
        metrics, dist_ratio, action_name = agent.process_frame(
            pitch, yaw, roll, ear, mar, brow, person_detected, phone_count,
            args.age, current_hour, args.occupation
        )
        
        if action_name:
            current_action_name = action_name
            action_history.append(current_action_name)
            
        # 5. Render UI Overlay
        session_elapsed = time.time() - start_time
        frame = draw_ui_overlay(frame, metrics, dist_ratio, action_history, fps_display, session_elapsed, current_action_name)
        
        cv2.imshow("Focus Tracker AI", frame)
        
        # 7. FPS Control & Quit
        process_time = time.time() - loop_start
        sleep_time = max(0, frame_time - process_time)
        time.sleep(sleep_time)
        
        fps_display = 1.0 / (time.time() - loop_start)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo terminated.")

if __name__ == "__main__":
    main()
