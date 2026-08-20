import sys
import traceback
from gui_app import processing_loop
try:
    print('Starting processing_loop test...')
    # We don't want it to run forever, so we can mock app_running?
    # Or just run it and see if it immediately crashes.
    import threading
    t = threading.Thread(target=processing_loop)
    t.daemon = True
    t.start()
    import time
    time.sleep(3)
    if not t.is_alive():
        print('Thread died early!')
    else:
        print('Thread is still alive after 3 seconds, meaning no immediate crash!')
except Exception as e:
    traceback.print_exc()
