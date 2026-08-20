import os
import sys
import cv2
import time
import threading
import datetime
import pystray
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from plyer import notification
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pygame

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from focus_env import FocusAgentInference
from demo_webcam import estimate_head_pose, calculate_ear, calculate_mar, calculate_brow_dist, calculate_gaze_ratio
from llm_agent import FocusLLMAgent

import win32event
import win32api
import winerror

# 1. SINGLE INSTANCE LOCK
mutex = win32event.CreateMutex(None, 1, 'FocusTrackerMutex')
if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
    notification.notify(title="Focus Tracker", message="App is already running!", timeout=3)
    sys.exit(0)

# Globals
app_running = True
music_playing = False
MUSIC_FILE = "focus_music.mp3"
latest_frame_pil = None
latest_stats = None
nudge_active = False
last_notified_msg = ""
last_toast_time = 0
camera_error = False

pygame.mixer.init()

def toggle_music():
    global music_playing
    if not music_playing:
        if os.path.exists(MUSIC_FILE):
            try:
                pygame.mixer.music.load(MUSIC_FILE)
                pygame.mixer.music.play(-1)
                music_playing = True
            except Exception as e:
                notification.notify(title="Music Error", message=str(e), timeout=3)
        else:
            notification.notify(title="Missing Music File", message="Please place 'focus_music.mp3' in the app folder.", timeout=5)
    else:
        pygame.mixer.music.stop()
        music_playing = False
    return music_playing

def processing_loop():
    global latest_frame_pil, latest_stats, app_running, camera_error
    
    print('Loading agent...')
    agent = FocusAgentInference(model_path="dqn_focus_agent_v2.zip", scaler_path="scaler_v2.pkl")
    print('Loading LLM...')
    llm = FocusLLMAgent()
    
    print('Loading MediaPipe...')
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    print('Loading YOLO...')
    yolo_model = YOLO('yolov8n.pt')
    
    print('Opening Camera...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        camera_error = True
        
    last_llm_action = None
    
    print('Started Loop...')
    while app_running:
        if nudge_active:
            time.sleep(1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            camera_error = True
            # Create a blank frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA UNAVAILABLE OR IN USE", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            latest_frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(1)
            continue
            
        camera_error = False
            
        frame = cv2.flip(frame, 1) # Mirror for natural feel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        person_detected = 0
        phone_count = 0
        yolo_person = 0
        
        yolo_results = yolo_model(frame, stream=True, verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    yolo_person = 1
                    # Draw a light box around person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif cls_id == 67:
                    phone_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "PHONE DETECTED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
        pitch, yaw, roll, ear, mar, brow, gaze_ratio = 0, 0, 0, 0, 0, 0, 0
        
        if detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            pitch, yaw, roll = estimate_head_pose(face_landmarks, frame.shape)
            ear = calculate_ear(face_landmarks)
            mar = calculate_mar(face_landmarks)
            # In gui_app, demo_webcam's calculate_brow_dist requires frame.shape? Let's assume it doesn't since we imported it from demo_webcam. Wait! demo_webcam's calculate_brow_dist might need it. We'll just pass face_landmarks.
            try:
                brow = calculate_brow_dist(face_landmarks, frame.shape)
            except:
                brow = calculate_brow_dist(face_landmarks)
            try:
                gaze_ratio = calculate_gaze_ratio(face_landmarks)
            except:
                gaze_ratio = calculate_gaze_ratio(face_landmarks, frame)
            person_detected = 1
            
        if person_detected == 0 and yolo_person > 0:
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
        
        latest_stats = {
            'action': action_name,
            'msg': current_msg,
            'dist_ratio': dist_ratio,
            'cf': agent.consecutive_frames,
            'person': person_detected
        }
        
        # Save image for UI
        latest_frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(1/15.0)
        
    cap.release()
    llm.stop()

# --- TKINTER GUI ---
root = tk.Tk()
root.title("Focus Tracker AI")
root.geometry("1100x650")
root.configure(bg="#1E1E1E")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.rowconfigure(0, weight=1)

left_frame = tk.Frame(root, bg="#2D2D2D", padx=20, pady=20)
left_frame.grid(row=0, column=0, sticky="nsew")

right_frame = tk.Frame(root, bg="#1E1E1E", padx=10, pady=10)
right_frame.grid(row=0, column=1, sticky="nsew")

tk.Label(left_frame, text="FOCUS TRACKER", font=("Arial", 22, "bold"), bg="#2D2D2D", fg="white").pack(anchor="w", pady=(0, 20))
time_lbl = tk.Label(left_frame, text="", font=("Arial", 16), bg="#2D2D2D", fg="#888888")
time_lbl.pack(anchor="w", pady=(0, 30))

tk.Label(left_frame, text="CURRENT STATUS", font=("Arial", 10), bg="#2D2D2D", fg="#888888").pack(anchor="w")
status_val = tk.Label(left_frame, text="INITIALIZING...", font=("Arial", 22, "bold"), bg="#2D2D2D", fg="#AAAAAA")
status_val.pack(anchor="w", pady=(0, 20))

tk.Label(left_frame, text="AI RECOMMENDATION", font=("Arial", 10), bg="#2D2D2D", fg="#888888").pack(anchor="w")
rec_val = tk.Label(left_frame, text="-", font=("Arial", 16), bg="#2D2D2D", fg="#FFA500")
rec_val.pack(anchor="w", pady=(0, 20))

tk.Label(left_frame, text="AI ASSISTANT (FOCUSMATE)", font=("Arial", 10), bg="#2D2D2D", fg="#888888").pack(anchor="w")
llm_val = tk.Label(left_frame, text="Loading...", font=("Arial", 13, "italic"), bg="#2D2D2D", fg="#4CAF50", wraplength=280, justify="left")
llm_val.pack(anchor="w", pady=(0, 30))

def toggle_music_gui():
    is_playing = toggle_music()
    if is_playing:
        music_btn.config(text="Stop Local Music", bg="#F44336")
    else:
        music_btn.config(text="Play Local Music", bg="#4CAF50")

music_btn = tk.Button(left_frame, text="Play Local Music", command=toggle_music_gui, bg="#4CAF50", fg="white", font=("Arial", 14, "bold"), pady=10)
music_btn.pack(anchor="w", fill="x", pady=(0, 20))

video_lbl = tk.Label(right_frame, bg="#1E1E1E")
video_lbl.pack(expand=True)

latest_frame_tk = None

def show_hard_nudge_window(msg):
    global nudge_active
    if nudge_active: return
    nudge_active = True
    
    nudge_win = tk.Toplevel(root)
    nudge_win.attributes('-topmost', True)
    nudge_win.attributes('-fullscreen', True)
    nudge_win.configure(bg='black')
    
    def on_click():
        global nudge_active
        nudge_active = False
        nudge_win.destroy()
        
    tk.Label(nudge_win, text="HARD NUDGE", fg="red", bg="black", font=("Arial", 50, "bold")).pack(pady=(150, 20))
    tk.Label(nudge_win, text=msg + "\n\nClick below to acknowledge and continue.", fg="white", bg="black", font=("Arial", 25)).pack(expand=True)
    tk.Button(nudge_win, text="I will focus now", font=("Arial", 20), bg="red", fg="white", command=on_click).pack(pady=100)

def update_gui():
    global latest_frame_tk, last_notified_msg, last_toast_time
    if not app_running:
        return
        
    time_lbl.config(text=datetime.datetime.now().strftime("%A, %d %B\n%I:%M:%S %p"))
    
    if latest_frame_pil:
        # Resize image to fit keeping aspect ratio
        img = latest_frame_pil.copy()
        img.thumbnail((800, 600), Image.Resampling.LANCZOS)
        latest_frame_tk = ImageTk.PhotoImage(image=img)
        video_lbl.config(image=latest_frame_tk)
        
    if latest_stats:
        dr = latest_stats['dist_ratio']
        cf = latest_stats['cf']
        act = latest_stats['action']
        msg = latest_stats['msg']
        
        if latest_stats['person'] == 0:
            status_text, status_color = "USER MISSING", "#FF0000"
        elif cf < 50:
            status_text, status_color = "FOCUSED", "#00FF00"
        elif cf < 100:
            status_text, status_color = "EARLY WARNING", "#FFA500"
        else:
            status_text, status_color = "DISTRACTED", "#FF0000"
            
        status_val.config(text=status_text, fg=status_color)
        rec_val.config(text=act)
        llm_val.config(text=msg)
        
        if act in ["Sound Alert", "Suggest Short Break"] and dr > 0.5:
            show_hard_nudge_window(msg)
            
        elif act in ["Gentle Visual Reminder", "Dim Screen"]:
            if msg != last_notified_msg and msg != "You're doing great! Keep it up.":
                if time.time() - last_toast_time > 15:
                    threading.Thread(target=notification.notify, 
                                     kwargs={'title': 'Focus AI Assistant', 'message': msg, 'app_name': 'Focus Tracker', 'timeout': 5}, 
                                     daemon=True).start()
                    last_notified_msg = msg
                    last_toast_time = time.time()
                    
    root.after(30, update_gui)

def start_tray():
    img = Image.new('RGB', (64, 64), color = (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([16, 16, 48, 48], fill=(0, 255, 0))
    
    def on_restore(icon, item):
        icon.stop()
        root.after(0, root.deiconify)
        
    def on_quit(icon, item):
        global app_running
        app_running = False
        icon.stop()
        root.after(0, root.destroy)
        
    def tray_toggle_music(icon, item):
        is_playing = toggle_music()
        # Sync GUI button
        if is_playing:
            root.after(0, lambda: music_btn.config(text="Stop Local Music", bg="#F44336"))
        else:
            root.after(0, lambda: music_btn.config(text="Play Local Music", bg="#4CAF50"))
            
    menu = pystray.Menu(
        pystray.MenuItem("Restore Application", on_restore, default=True),
        pystray.MenuItem(lambda text: "Stop Local Music" if music_playing else "Play Local Music", tray_toggle_music),
        pystray.MenuItem("Quit", on_quit)
    )
    tray_icon = pystray.Icon("FocusTracker", img, "Focus Tracker AI", menu)
    threading.Thread(target=tray_icon.run, daemon=True).start()

def on_closing():
    dialog = tk.Toplevel(root)
    dialog.title("Exit Options")
    dialog.geometry("450x150")
    dialog.configure(bg="#2D2D2D")
    dialog.transient(root)
    dialog.grab_set()
    
    dialog.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - 450) // 2
    y = root.winfo_y() + (root.winfo_height() - 150) // 2
    dialog.geometry(f"+{x}+{y}")
    
    tk.Label(dialog, text="Do you want to minimize to tray or quit completely?", bg="#2D2D2D", fg="white", font=("Arial", 12)).pack(pady=20)
    
    btn_frame = tk.Frame(dialog, bg="#2D2D2D")
    btn_frame.pack()
    
    def do_minimize():
        dialog.destroy()
        root.withdraw()
        start_tray()
        notification.notify(title="Focus Tracker", message="App is running in the background.", timeout=3)
        
    def do_quit():
        global app_running
        app_running = False
        dialog.destroy()
        root.destroy()
        
    tk.Button(btn_frame, text="Minimize to Tray", command=do_minimize, bg="#2196F3", fg="white", width=15).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Quit Completely", command=do_quit, bg="#F44336", fg="white", width=15).pack(side=tk.LEFT, padx=10)

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start Processing
threading.Thread(target=processing_loop, daemon=True).start()

# Start GUI
root.after(30, update_gui)
root.mainloop()
pygame.mixer.quit()

