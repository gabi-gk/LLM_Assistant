'''
Provides window management tools for the assistant
- List all open windows
- Switch focus to a specific window
- Minimize, maximize windows
- Get the currently active window
OS-aware: uses pygetwindow on Windows, wmctrl on Linux
'''
import subprocess
from config import OS
import pygetwindow as gw
from core.utils import confirm

def list_open_windows():
    """
    List all currently open windows

    returns: formatted string of open window titles
    """
    if OS == "Windows":
        windows = [w.title for w in gw.getAllWindows() if w.title.strip()] # filter out windows with empty titles (like background processes)
        if not windows:
            return "[INFO] No open windows found"
        return "[OPEN WINDOWS]\n" + "\n".join(f"  - {w}" for w in windows) # return formatted list of window titles

    elif OS == "Linux":
        result = subprocess.run(
            "wmctrl -l", shell=True, capture_output=True, text=True # list all windows with wmctrl
        )
        if result.returncode != 0:
            return "[ERROR] wmctrl not found — run: sudo pacman -S wmctrl"
        lines = result.stdout.strip().splitlines() # output format: 0x01200007  0 myhostname Terminal - bash
        titles = [" ".join(line.split()[3:]) for line in lines if line]
        return "[OPEN WINDOWS]\n" + "\n".join(f"  - {t}" for t in titles)

    return f"[ERROR] Unsupported OS: {OS}"


def switch_to_window(title):
    """
    Bring a window to focus by its title 
    
    title: full or partial window title to search for
    returns: success or error message
    """
    if OS == "Windows":
        # partial match — find first window containing the title string
        matches = [w for w in gw.getAllWindows() if title.lower() in w.title.lower() and w.title.strip()]
        if not matches:
            return f"[ERROR] No window found matching '{title}'"
        
        try:
            matches[0].activate() # activate the first matching window
            return f"[SUCCESS] Switched to '{matches[0].title}'"
        except Exception as e:
            return f"[ERROR] Could not switch to window: {e}"

    elif OS == "Linux":
        result = subprocess.run(
            f'wmctrl -a "{title}"', shell=True, capture_output=True, text=True # switch to the window with wmctrl (partial title match)
        )
        if result.returncode != 0:
            return f"[ERROR] Could not switch to '{title}'"
        return f"[SUCCESS] Switched to '{title}'"

    return f"[ERROR] Unsupported OS: {OS}"


def minimize_window(title):
    """
    Minimize a window by its title

    title: full or partial window title
    returns: success or error message
    """
    if OS == "Windows":
        matches = [w for w in gw.getAllWindows() if title.lower() in w.title.lower() and w.title.strip()]
        if not matches:
            return f"[ERROR] No window found matching '{title}'"
        
        try:
            matches[0].minimize()
            return f"[SUCCESS] Minimized '{matches[0].title}'"
        except Exception as e:
            return f"[ERROR] Could not minimize window: {e}"

    elif OS == "Linux":
        result = subprocess.run(
            f'wmctrl -r "{title}" -b add,hidden',
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            return f"[ERROR] Could not minimize '{title}'"
        return f"[SUCCESS] Minimized '{title}'"

    return f"[ERROR] Unsupported OS: {OS}"


def maximize_window(title):
    """
    Maximize a window by its title

    title: full or partial window title
    returns: success or error message
    """
    if OS == "Windows":
        
        matches = [w for w in gw.getAllWindows() if title.lower() in w.title.lower() and w.title.strip()]
        if not matches:
            return f"[ERROR] No window found matching '{title}'"
        
        try:
            matches[0].maximize()
            return f"[SUCCESS] Maximized '{matches[0].title}'"
        except Exception as e:
            return f"[ERROR] Could not maximize window: {e}"

    elif OS == "Linux":
        result = subprocess.run(
            f'wmctrl -r "{title}" -b add,maximized_vert,maximized_horz',
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            return f"[ERROR] Could not maximize '{title}'"
        return f"[SUCCESS] Maximized '{title}'"

    return f"[ERROR] Unsupported OS: {OS}"


def list_active_window():
    """
    Get the title of the currently focused window
    returns: title string of the active window
    """
    if OS == "Windows":
        active = gw.getActiveWindow()
        if active:
            return f"[ACTIVE WINDOW] {active.title}"
        return "[INFO] No active window found"

    elif OS == "Linux":
        result = subprocess.run(
            "xdotool getactivewindow getwindowname",
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            return "[ERROR] xdotool not found — run: sudo pacman -S xdotool"
        return f"[ACTIVE WINDOW] {result.stdout.strip()}"

    return f"[ERROR] Unsupported OS: {OS}"

def close_window(title):
    """
    Close a window by its title
    Requires confirmation before closing to prevent accidental data loss

    title: full or partial window title
    returns: success or error message
    """
    if OS == "Windows":
        matches = [w for w in gw.getAllWindows() if title.lower() in w.title.lower() and w.title.strip()]
        if not matches:
            return f"[ERROR] No window found matching '{title}'"
        
        if not confirm(f"Close '{matches[0].title}'?"): # get confirmation from the utility function
            return "[CANCELLED] Close cancelled"
        
        try:
            matches[0].close()
            return f"[SUCCESS] Closed '{matches[0].title}'"
        except Exception as e:
            return f"[ERROR] Could not close window: {e}"
    elif OS == "Linux":
        result = subprocess.run(
            f'wmctrl -c "{title}"',
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            return f"[ERROR] Could not close '{title}'"
        return f"[SUCCESS] Closed '{title}'"
    return f"[ERROR] Unsupported OS: {OS}"