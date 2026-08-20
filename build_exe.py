import os
import subprocess
import sys
import shutil

def build():
    print('Building FocusTracker GUI... (Directory mode)')
    subprocess.check_call([sys.executable, '-m', 'PyInstaller', '-y', '--onedir', '--noconsole', '--name', 'FocusTracker', '--hidden-import', 'sklearn', '--collect-data', 'stable_baselines3', '--collect-data', 'mediapipe', 'gui_app.py'])
    
    print('Copying required model files and music to dist/FocusTracker/...')
    files_to_copy = [
        ('models/dqn_focus_agent_v2.1.zip', 'dqn_focus_agent_v2.1.zip'),
        ('models/scaler_v2.1.pkl', 'scaler_v2.1.pkl'),
        ('models/yolov8n.pt', 'yolov8n.pt'),
        ('assets/focus_music.mp3', 'focus_music.mp3'),
        ('models/face_landmarker.task', 'face_landmarker.task')
    ]
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy(src, os.path.join('dist', 'FocusTracker', dst))
            print(f'Copied {src} to {dst}')
        else:
            print(f'WARNING: Could not find {src}')
            
    print('Build complete! Check the dist/FocusTracker/ folder for FocusTracker.exe')

if __name__ == '__main__':
    build()
