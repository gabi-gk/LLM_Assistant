"""
Locking mechanism to prevent multiple instances of the assistant from running simultaneously
Creates a lock file with the current process ID when the assistant starts, and checks for this file
If the file exists and the process ID is active, it exits with an error
"""

import os
import sys
from config import LOCK_FILE

def acquire_lock():
    """ 
    Check for existing lock file and create one if not present
    LOCK_FILE defined in config.py, default is "./data/assistant.lock"
    """
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                pid = int(f.read().strip()) # Check if the process with this process ID is still running
            os.kill(pid, 0)
            print(f"[ERROR] Assistant already running (PID {pid}). Close it first.")
            sys.exit(1) # exit the application if an instance is already running
        except (OSError, ValueError, PermissionError): # no file present or is bugged, continue to creating a new lock file
            pass  # OSError means the process is not running, ValueError means the file was corrupted, PermissionError if the process is owned by another user
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid())) # create a new lock file

def release_lock():
    """ 
    Remove the lock file on exitting the application
    """
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
