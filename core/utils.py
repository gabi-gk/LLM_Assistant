'''
Shared utility functions used across the project
- System prompt formatting with current time and timezone
- Confirmation dialogs for file and shell operations
'''

from datetime import datetime
import tkinter.messagebox as messagebox
import tzlocal

def get_system_prompt(base_prompt):
    """
    Inject local time and timezone to the model's prompt on each turn
    
    base_prompt: system prompt from config
    returns: system prompt formatted with time
    """
    try:
        timezone = tzlocal.get_localzone_name() # read system timezone automatically
    except Exception:
        timezone = "Europe/London"
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    return base_prompt.format( # inject the time to the prompt on each request
        current_time=current_time,
        current_timezone=timezone
    )

def confirm(prompt, use_gui=True):
    """
    Ask for confirmation and return true if confirmed

    prompt: the confirmation message to show
    use_gui: true for tray, false for terminal
    returns True if the user confirms, False otherwise
    """
    if use_gui: # show in a message box
        return messagebox.askyesno("Marvin asking for confirmation", prompt)
    else:
        response = input(f"\n[CONFIRM] {prompt} (y/n): ").strip().lower() # normal terminal input
        return response in ("y", "yes")