import os
import sys

# --- HARD NUDGE CLI MODE ---
# This runs when subprocess calls background_app.exe --nudge
if len(sys.argv) > 1 and sys.argv[1] == '--nudge':
    import tkinter as tk
    msg = sys.argv[2] if len(sys.argv) > 2 else "You seem highly distracted. Time to focus!"
    
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.attributes('-fullscreen', True)
    root.configure(bg='black')
    
    def on_click():
        root.destroy()
        
    lbl = tk.Label(root, text="HARD NUDGE", fg="red", bg="black", font=("Arial", 50, "bold"))
    lbl.pack(pady=(150, 20))
    
    lbl2 = tk.Label(root, text=msg + "\n\nClick below to acknowledge and continue.", 
                   fg="white", bg="black", font=("Arial", 25))
    lbl2.pack(expand=True)
    
    btn = tk.Button(root, text="I will focus now", font=("Arial", 20), bg="red", fg="white", command=on_click)
    btn.pack(pady=100)
    
    root.mainloop()
    sys.exit(0)

# --- MAIN APP MODE ---
import cv2
import time
import threading
import datetime
import pystray
from PIL import Image, ImageDraw
import subprocess
from plyer import notification
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import win32event
import win32api
import winerror
import pygame

import mediapipe as mp
from ultralytics import YOLO

from focus_env import FocusAgentInference
from demo_webcam import estimate_head_pose, calculate_ear, calculate_mar, calculate_brow_dist, calculate_gaze_ratio
from llm_agent import FocusLLMAgent

# 1. SINGLE INSTANCE LOCK
mutex = win32event.CreateMutex(None, 1, 'FocusTrackerMutex')
if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
    notification.notify(title="Focus Tracker", message="App is already running in the background!", timeout=3)
    sys.exit(0)

app_running = True
nudge_active = False

# 2. LOCAL MUSIC SETUP
pygame.mixer.init()
music_playing = False
MUSIC_FILE = "focus_music.mp3" # User can replace this file

def toggle_music(icon, item):
    global music_playing
    if not music_playing:
        if os.path.exists(MUSIC_FILE):
            try:
                pygame.mixer.music.load(MUSIC_FILE)
                pygame.mixer.music.play(-1) # Loop indefinitely
                music_playing = True
            except Exception as e:
                notification.notify(title="Music Error", message=str(e), timeout=3)
        else:
            # Tell user to put mp3
            notification.notify(title="Missing Music File", message="Please place a 'focus_music.mp3' file in the dist/background_app folder.", timeout=5)
    else:
        pygame.mixer.music.stop()
        music_playing = False
        
    icon.update_menu()

def create_tray():
    img = Image.new('RGB', (64, 64), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([16, 16, 48, 48], fill=(0, 255, 0))
    
    def on_quit(icon, item):
        global app_running
        app_running = False
        icon.stop()

    # Dynamic Menu
    menu = pystray.Menu(
        pystray.MenuItem(lambda text: "Stop Local Music" if music_playing else "Play Local Music", toggle_music),
        pystray.MenuItem("Exit", on_quit)
    )
    icon = pystray.Icon("FocusTracker", img, "Focus Tracker AI", menu)
    icon.run()

def show_hard_nudge(msg):
    global nudge_active
    nudge_active = True
    # Spawn a new process to handle Tkinter safely without crashing the main thread
    subprocess.run([sys.executable, "--nudge", msg])
    nudge_active = False

def main():
    global app_running
    
    agent = FocusAgentInference(model_path="dqn_focus_agent_v2.zip", scaler_path="scaler_v2.pkl")
    llm = FocusLLMAgent()
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    yolo_model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(0)
    
    last_llm_action = None
    last_notified_msg = ""
    last_toast_time = 0
    
    print("Background App Running. Check system tray.")
    
    while app_running:
        if nudge_active:
            time.sleep(1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        person_detected = 0
        phone_count = 0
        
        yolo_results = yolo_model(frame, stream=True, verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    person_detected = 1
                elif cls_id == 67:
                    phone_count += 1
                    
        pitch, yaw, roll, ear, mar, brow, gaze_ratio = 0, 0, 0, 0, 0, 0, 0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            pitch, yaw, roll = estimate_head_pose(face_landmarks, frame.shape)
            ear = calculate_ear(face_landmarks)
            mar = calculate_mar(face_landmarks)
            brow = calculate_brow_dist(face_landmarks)
            gaze_ratio = calculate_gaze_ratio(face_landmarks, frame)
            person_detected = 1
            
        current_hour = datetime.datetime.now().hour
        
        metrics, dist_ratio, action_name = agent.process_frame(
            pitch, yaw, roll, ear, mar, brow, person_detected, phone_count,
            25, current_hour, "Student", gaze_ratio
        )
        
        if action_name == "Play Focus Music":
            action_name = "Do Nothing"
            
        if action_name != last_llm_action and action_name != "Do Nothing":
            last_llm_action = action_name
            llm.update_context(action_name, metrics)
            
        if action_name == "Do Nothing":
            last_llm_action = None
            llm.reset_message()
            
        current_msg = llm.get_current_message()
        
        # Dispatch actions
        if action_name in ["Sound Alert", "Suggest Short Break"] and dist_ratio > 0.5:
            if not nudge_active:
                threading.Thread(target=show_hard_nudge, args=(current_msg,), daemon=True).start()
                
        elif action_name in ["Gentle Visual Reminder", "Dim Screen"]:
            if current_msg != last_notified_msg and current_msg != "You're doing great! Keep it up.":
                if time.time() - last_toast_time > 15:
                    threading.Thread(target=notification.notify, 
                                     kwargs={
                                         'title': 'Focus AI Assistant', 
                                         'message': current_msg,
                                         'app_name': 'Focus Tracker',
                                         'timeout': 5
                                     }, 
                                     daemon=True).start()
                    last_notified_msg = current_msg
                    last_toast_time = time.time()
                    
        time.sleep(1/10.0)
        
    cap.release()
    llm.stop()
    pygame.mixer.quit()
    print("Exiting...")

if __name__ == "__main__":
    processing_thread = threading.Thread(target=main, daemon=True)
    processing_thread.start()
    create_tray()
