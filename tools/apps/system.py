import subprocess
from config import OS
from pathlib import Path

def open_application(name):
    """
    Open an application by name or common shortcut
    Searches Start Menu, common paths and Windows built-in shortcuts

    name: application name or keyword e.g. "settings"
    returns: success or error message
    """
    if OS == "Windows":
        # Windows built-in shortcuts and ms-settings URIs
        builtins = {
            "settings":        "ms-settings:",
            "wifi":            "ms-settings:network-wifi",
            "bluetooth":       "ms-settings:bluetooth",
            "display":         "ms-settings:display",
            "sound":           "ms-settings:sound",
            "apps":            "ms-settings:appsfeatures",
            "updates":         "ms-settings:windowsupdate",
            "defender":        "windowsdefender:",
            "calculator":      "calc",
            "notepad":         "notepad",
            "explorer":        "explorer",
            "task manager":    "taskmgr",
            "device manager":  "devmgmt.msc",
            "disk cleanup":    "cleanmgr",
            "control panel":   "control",
            "cmd":             "cmd",
            "powershell":      "powershell",
        }

        name_lower = name.lower().strip()

        # check built-in shortcuts first
        if name_lower in builtins:
            try:
                subprocess.Popen(
                    f'start {builtins[name_lower]}',
                    shell=True
                )
                return f"[SUCCESS] Opened '{name}'"
            except Exception as e:
                return f"[ERROR] Could not open '{name}': {e}"

        # check the app exists in PATH before launching — Popen with shell=True never raises for missing commands
        check = subprocess.run(f'where "{name}"', shell=True, capture_output=True)
        if check.returncode != 0:
            return f"[ERROR] '{name}' not found. Try find_application to search for it."
        subprocess.Popen(name, shell=True)
        return f"[SUCCESS] Launched '{name}'"

    elif OS == "Linux":
        check = subprocess.run(f'which "{name}"', shell=True, capture_output=True)
        if check.returncode != 0:
            return f"[ERROR] '{name}' not found. Try find_application to search for it."
        subprocess.Popen(name, shell=True)
        return f"[SUCCESS] Launched '{name}'"

    return f"[ERROR] Unsupported OS: {OS}"


def find_application(name):
    """
    Search for an installed application by name
    Useful when not sure of the exact app name or path

    name: partial application name to search for
    returns: list of matching installed applications
    """
    if OS == "Windows":
        # search registry for installed apps
        result = subprocess.run(
            f'powershell "Get-StartApps | Where-Object {{$_.Name -like \'*{name}*\'}} | Select-Object Name, AppID"',
            shell=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            return f"[INSTALLED APPS matching '{name}']\n{result.stdout.strip()}"
        return f"[INFO] No installed apps found matching '{name}'"

    elif OS == "Linux":
        result = subprocess.run(
            f"find /usr/share/applications -name '*{name}*'",
            shell=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            apps = [Path(p).stem for p in result.stdout.strip().splitlines()]
            return f"[INSTALLED APPS matching '{name}']\n" + "\n".join(apps)
        return f"[INFO] No apps found matching '{name}'"

    return f"[ERROR] Unsupported OS: {OS}"