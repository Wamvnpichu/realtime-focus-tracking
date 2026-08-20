import cv2
import time
import argparse
import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import os

from focus_env import FocusAgentInference
from demo_webcam import estimate_head_pose, calculate_ear, calculate_mar, calculate_brow_dist, calculate_gaze_ratio
from llm_agent import FocusLLMAgent

def draw_llm_overlay(frame, metrics, action_history, fps, session_time, current_action_name, llm_msg):
    h, w = frame.shape[:2]
    
    # Colors
    COLOR_GREEN = (83, 200, 0)
    COLOR_RED = (68, 23, 255)
    COLOR_ORANGE = (0, 145, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PANEL = (0, 0, 0)
    
    overlay = frame.copy()
    
    # Left sidebar
    cv2.rectangle(overlay, (0, 0), (320, h), COLOR_PANEL, -1)
    # Bottom bar (Thicker for LLM message)
    cv2.rectangle(overlay, (0, h - 80), (w, h), COLOR_PANEL, -1)
    
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Text info
    cv2.putText(frame, "LLM FOCUS TRACKER", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_TEXT, 1)
    cv2.line(frame, (20, 45), (300, 45), (100, 100, 100), 1)
    
    y_offset = 80
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    y_offset += 30
    cv2.putText(frame, f"Time: {int(session_time//60)}m {int(session_time%60)}s", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
    
    y_offset += 40
    cv2.putText(frame, "ENVIRONMENT", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 25
    
    phone_text = "Detected" if metrics.get('phone_count', 0) > 0 else "Not detected"
    phone_color = COLOR_RED if metrics.get('phone_count', 0) > 0 else COLOR_GREEN
    cv2.putText(frame, f"Phone: {phone_text}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 1)
    
    y_offset += 40
    cv2.line(frame, (20, y_offset), (300, y_offset), (100, 100, 100), 1)
    y_offset += 40
    
    cf = metrics.get('consecutive_frames', 0)
    person = metrics.get('person_detected', 1)
    
    if person == 0:
        status_text = "USER MISSING"
        status_color = COLOR_RED
    elif cf < 50:
        status_text = "FOCUSED"
        status_color = COLOR_GREEN
    elif cf < 100:
        status_text = "EARLY WARNING"
        status_color = COLOR_ORANGE
    else:
        status_text = "DISTRACTED"
        status_color = COLOR_RED
        
    cv2.putText(frame, status_text, (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
    
    y_offset += 50
    cv2.putText(frame, "AGENT RECOMMENDATION", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 35
    
    if current_action_name:
        cv2.putText(frame, current_action_name, (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_ORANGE, 2)
    
    # LLM MESSAGE AT BOTTOM
    cv2.putText(frame, "AI Assistant:", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, llm_msg, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', type=int, default=25)
    parser.add_argument('--occupation', type=str, default='office')
    parser.add_argument('--hour', type=float, default=-1)
    args = parser.parse_args()

    agent = FocusAgentInference(
        model_path="dqn_focus_agent_v2.zip",
        scaler_path="scaler_v2.pkl"
    )

    llm = FocusLLMAgent()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FPS, 10)
    frame_time = 1.0 / 10.0

    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    yolo_model = YOLO("yolov8n.pt")

    action_history = []
    fps_display = 10.0
    start_time = time.time()
    
    current_action_name = "Waiting for data..."
    last_llm_action = None

    print("Starting LLM-powered WebCam Demo (Press 'q' to quit)...")
    if not llm.client:
        print("[WARNING] GEMINI_API_KEY is not set or google-genai is missing. LLM messages will be simulated.")

    while True:
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        results = yolo_model(frame, verbose=False)
        phone_count = sum(1 for box in results[0].boxes if int(box.cls) == 67 and float(box.conf) > 0.5)
        yolo_person = 1 if sum(1 for box in results[0].boxes if int(box.cls) == 0 and float(box.conf) > 0.5) > 0 else 0
        
        person_detected = 0
        pitch = yaw = roll = ear = mar = brow = gaze_ratio = 0.0
        
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            person_detected = 1
            pitch, yaw, roll = estimate_head_pose(face_landmarks, frame.shape)
            ear = calculate_ear(face_landmarks)
            mar = calculate_mar(face_landmarks)
            brow = calculate_brow_dist(face_landmarks, frame.shape)
            gaze_ratio = calculate_gaze_ratio(face_landmarks)
            
        # If face is lost but YOLO still sees a person (e.g. turned around)
        if person_detected == 0 and yolo_person > 0:
            person_detected = 1
        
        current_hour = args.hour if args.hour != -1 else datetime.datetime.now().hour
        metrics, dist_ratio, action_name = agent.process_frame(
            pitch, yaw, roll, ear, mar, brow, person_detected, phone_count,
            args.age, current_hour, args.occupation, gaze_ratio
        )
        
        if action_name:
            current_action_name = action_name
            action_history.append(current_action_name)
            if len(action_history) > 10:
                action_history.pop(0)
                
            # Trigger LLM update if action changes and it's not "Do Nothing"
            if current_action_name != last_llm_action and current_action_name != "Do Nothing":
                last_llm_action = current_action_name
                llm.update_context(current_action_name, metrics)
            
            # Reset LLM message when focused
            if current_action_name == "Do Nothing":
                last_llm_action = None
                llm.reset_message()
            
        session_elapsed = time.time() - start_time
        
        frame = draw_llm_overlay(
            frame, metrics, action_history, fps_display, session_elapsed, 
            current_action_name, llm.get_current_message()
        )
        
        cv2.imshow("LLM Focus Tracker AI", frame)
        
        process_time = time.time() - loop_start
        sleep_time = max(0, frame_time - process_time)
        time.sleep(sleep_time)
        
        fps_display = 1.0 / (time.time() - loop_start)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    llm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Demo terminated.")

if __name__ == "__main__":
    main()
