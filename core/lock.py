"""
Locking mechanism to prevent multiple instances of the assistant from running simultaneously
Creates a lock file with the current process ID when the assistant starts, and checks for this file
If the file exists and the process ID is active, it exits with an error
"""

import os
import sys
from config import LOCK_FILE
import ctypes
from ctypes import wintypes

def pid_exists(pid):
    """
    Check whether a Marvin instance is already running
    Test for a PID exsisting on the system, branch for Windows to aboid os.kill() which is not supported on Windows
    """
    if os.name == "nt":
        # For Windows, use the Windows API to ask Windows directly if the process is still running
        STILL_ACTIVE = 259
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) # library that exposes the process APIs
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False # no PID
        try:
            exit_code = wintypes.DWORD()
            # Check whether the process is still active through the exit code
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    # Works for non Windows platforms
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError: # process exists but is owned by another user
        return True
    return True

def acquire_lock():
    """
    Check for existing lock file and create one if not present
    LOCK_FILE defined in config.py, default is "./data/assistant.lock"
    """
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                pid = int(f.read().strip()) # Check if the process with this process ID is still running
        except ValueError:
            pid = None # lock file was corrupted, continue to creating a new lock file
        if pid is not None and pid_exists(pid):
            print(f"[ERROR] Assistant already running (PID {pid}). Close it first.")
            sys.exit(1) # exit the application if an instance is already running
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid())) # create a new lock file

def release_lock():
    """ 
    Remove the lock file on exitting the application
    """
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
