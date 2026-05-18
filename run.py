"""
The entry point for the GUI tray application
Ensures only one instance runts at a time via a lock file then starts the TrayApp
"""

from tray.app import TrayApp
from core.lock import acquire_lock, release_lock

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__": 
    acquire_lock()
    try:
        app = TrayApp()
        app.run()
    finally:
        release_lock()