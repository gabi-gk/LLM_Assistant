'''
Allows the assistant to send Windows desktop notifiactions
- One-time scheduled reminders 
- Persistent reminders that repeat at intervals until cancelled
- List, edit and cancel active reminders
'''

import threading
import time
import os
import json
from pathlib import Path
from plyer import notification
from config import REMINDERS_FILE


# active persistent reminders — key: reminder_id, value: stop Event
active_reminders = {}

def save_reminders(reminders_data):
    """
    Persist reminder metadata to disk so they survive restarts

    reminders_data: a dict of reminder_id to reminder details (title, message, interval)
    """
    os.makedirs("./data", exist_ok=True)
    with open(REMINDERS_FILE, "w") as f:
        json.dump(reminders_data, f, indent=2)


def load_reminders():
    """
    Load persisted reminders from disk
    """
    if not Path(REMINDERS_FILE).exists():
        return {}
    try:
        with open(REMINDERS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def restore_reminders():
    """
    Called on startup to restore any persistent reminders that were running when the assistant was last closed
    """
    saved = load_reminders()
    if not saved:
        return

    print(f"[REMINDERS] Restoring {len(saved)} active reminder(s)...")
    for reminder_id, data in saved.items():
        persistent_reminder(
            data["title"],
            data["message"],
            data["interval_minutes"],
            reminder_id=reminder_id   # restore with original ID
        )

def send_notification(title, message):
    """
    Send an immediate Windows desktop notification

    title: the title of the notification
    message: the body text of the notification
    """
    try:
        notification.notify( # uses plyer to send a notification
            title=title,
            message=message,
            app_name="Marvin",
            timeout=10
        )
        return f"[SUCCESS] Notification sent: {title}"
    except Exception as e:
        return f"[ERROR] Could not send notification: {e}"


def schedule_reminder(title, message, delay_minutes):
    """
    Send a single notification after a delay

    title: the title of the notification
    message: the body text of the notification
    delay_minutes: how many minutes to wait before sending the notification
    """
    def remind():
        time.sleep(delay_minutes * 60)
        send_notification(title, message)

    thread = threading.Thread(target=remind, daemon=True) # run in background so it doesn't block the main app
    thread.start() 
    return f"[SUCCESS] Reminder set for {delay_minutes} minutes: {title}"


def persistent_reminder(title, message, interval_minutes, reminder_id=None):
    """
    Send a notification repeatedly at set intervals until cancelled
    
    title: the title of the notification
    message: the body text of the notification
    interval_minutes: how many minutes to wait between notifications
    reminder_id: (optional) a unique ID for this reminder, if not provided one will be generated
    returns a unique reminder ID that can be used to edit or cancel the reminder
    """
    if reminder_id is None: # generate a unique ID if one isn't provided
        # replace spaces so ID is clean and unambiguous for the model
        safe_title = title.replace(" ", "_")
        reminder_id = f"{safe_title}_{int(time.time())}"

    stop_event = threading.Event() # used to signal the reminder thread to stop when the reminder is cancelled
    active_reminders[reminder_id] = stop_event

    # save to disk so reminder survives restarts
    saved = load_reminders()
    saved[reminder_id] = {
        "title": title,
        "message": message,
        "interval_minutes": interval_minutes
    }
    save_reminders(saved)

    def remind_loop():
        '''
        the background function that notfies at the set inverval until cancelled
        '''
        while not stop_event.is_set():
            send_notification(title, message)
            # check stop_event every second so cancellation is responsive
            for _ in range(int(interval_minutes * 60)):
                if stop_event.is_set():
                    break
                time.sleep(1)

        # clean up from disk when stopped
        saved = load_reminders()
        if reminder_id in saved:
            del saved[reminder_id]
            save_reminders(saved)

    thread = threading.Thread(target=remind_loop, daemon=True) # run in background so it doesn't block the main app
    thread.start()
    return f"[SUCCESS] Persistent reminder started (id: {reminder_id}): {title}"


def cancel_reminder(reminder_id):
    """
    Cancel a persistent reminder by its id

    reminder_id: the unique ID of the reminder to cancel
    """
    if reminder_id in active_reminders:
        active_reminders[reminder_id].set() # signals thread to stop
        del active_reminders[reminder_id]
        return f"[SUCCESS] Reminder cancelled: {reminder_id}"
    return f"[ERROR] No active reminder with id: {reminder_id}"

def edit_reminder(reminder_id, title=None, message=None, interval_minutes=None):
    """
    Edit an active persistent reminder
    Cancels the existing one and recreates with updated values
    Unchanged fields keep their original values

    reminder_id: the unique ID of the reminder to edit
    title: (optional) new title for the reminder
    message: (optional) new message for the reminder
    interval_minutes: (optional) new interval for the reminder
    """
    saved = load_reminders()

    if reminder_id not in saved:
        return f"[ERROR] No active reminder with id: {reminder_id}"

    # keep existing values for any fields not being updated
    existing = saved[reminder_id]
    new_title = title or existing["title"]
    new_message = message or existing["message"]
    new_interval = interval_minutes or existing["interval_minutes"]

    # cancel existing then recreate with updated values
    cancel_reminder(reminder_id)
    return persistent_reminder(new_title, new_message, new_interval)

def list_reminders():
    """
    List all currently active persistent reminders

    returns a formatted string listing all active reminders with their details
    """
    saved = load_reminders()
    if not saved:
        return "[INFO] No active reminders"
    lines = ["[ACTIVE REMINDERS]"]
    for rid, data in saved.items():
        lines.append(f" ID: {rid}")
        lines.append(f" Title: {data['title']}")
        lines.append(f" Interval: every {data['interval_minutes']} minutes")
        lines.append("")
    return "\n".join(lines)