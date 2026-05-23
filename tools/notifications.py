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
from datetime import datetime, timedelta


# active persistent reminders- key: reminder_id, value: stop Event
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
    '''
    Restore existing reminders at app or pc reset
    '''    
    saved = load_reminders()
    if not saved:
        return

    print(f"[REMINDERS] Restoring {len(saved)} active reminder(s)...")
    for reminder_id, data in saved.items():
        # calculate remaining time from saved next_trigger_time
        next_trigger_str = data.get("next_trigger_time")
        
        if next_trigger_str:
            next_trigger = datetime.fromisoformat(next_trigger_str)
            remaining_seconds = (next_trigger - datetime.now()).total_seconds()
            # if already past due, fire immediately
            remaining_seconds = max(0, remaining_seconds)
        else:
            remaining_seconds = 0

        if data.get("type") == "scheduled":
            # restore as one-time reminder with remaining time
            remaining_minutes = remaining_seconds / 60
            schedule_reminder(
                data["title"],
                data["message"],
                remaining_minutes if remaining_minutes > 0 else 0.1
            )
        else:
            # restore persistent reminder - first fire after remaining time
            persistent_reminder(
                data["title"],
                data["message"],
                data["interval_minutes"],
                reminder_id=reminder_id,
                initial_delay_seconds=remaining_seconds # preserve schedule
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
    safe_title = title.replace(" ", "_")
    reminder_id = f"{safe_title}_{int(time.time())}"
    
    stop_event = threading.Event()
    active_reminders[reminder_id] = stop_event

    next_trigger = datetime.now() + timedelta(minutes=delay_minutes)
    save_reminder_to_disk( # make sure to save the notification to the disc
        reminder_id, title, message,
        delay_minutes=delay_minutes,
        next_trigger_time=next_trigger.isoformat(),
        reminder_type="scheduled"
    )

    def remind():
        '''
        the background function that keeps track of the notification until cancelled
        '''
        # cound down for easy interrupt for cancelation
        for _ in range(int(delay_minutes * 60)):
            if stop_event.is_set():
                remove_reminder_from_disk(reminder_id)
                return
            time.sleep(1)
        
        send_notification(title, message) # fire after delay
        
        # remove from the disc and active reminders list
        if reminder_id in active_reminders:
            del active_reminders[reminder_id]
        remove_reminder_from_disk(reminder_id)

    threading.Thread(target=remind, daemon=True).start() # start the reminder
    return f"[SUCCESS] Reminder set for {delay_minutes} minutes: {title} (id: {reminder_id})"


def persistent_reminder(title, message, interval_minutes, reminder_id=None, initial_delay_seconds=0):
    """
    Send a notification repeatedly at set intervals until cancelled
    
    title: the title of the notification
    message: the text of the notification
    interval_minutes: how many minutes to wait between notifications
    reminder_id: unique ID generated if not provided, passed when restoring
    initial_delay_seconds: seconds to wait before first fire - used when restoring a reminder
    returns a unique reminder ID that can be used to edit or cancel the reminder
    """
    if reminder_id is None:
        safe_title = title.replace(" ", "_") # strip name 
        reminder_id = f"{safe_title}_{int(time.time())}"

    stop_event = threading.Event()
    active_reminders[reminder_id] = stop_event

    next_trigger = datetime.now() + timedelta(minutes=interval_minutes) # next trigger time
    save_reminder_to_disk( # save to disk for restore
        reminder_id, title, message,
        interval_minutes=interval_minutes,
        next_trigger_time=next_trigger.isoformat(),
        reminder_type="persistent"
    )

    def remind_loop():
        '''
        background functionthat fires at intervals until cancelled, delay 0 for new ones, calculated delay after restart
        '''
        if initial_delay_seconds > 0:
             # cound down for easy interrupt for cancelation
            for _ in range(int(initial_delay_seconds)):
                if stop_event.is_set():
                    remove_reminder_from_disk(reminder_id)
                    return
                time.sleep(1)
        
        # fire then wait at set interval until cancelled
        while not stop_event.is_set():
            send_notification(title, message)
            update_next_trigger_time(reminder_id, interval_minutes)
            for _ in range(int(interval_minutes * 60)):
                if stop_event.is_set():
                    break
                time.sleep(1)
        
        remove_reminder_from_disk(reminder_id)

    threading.Thread(target=remind_loop, daemon=True).start() # start the reminder
    return f"[SUCCESS] Persistent reminder started (id: {reminder_id}): {title}"

def save_reminder_to_disk(reminder_id, title, message, interval_minutes=None, delay_minutes=None, next_trigger_time=None, reminder_type="persistent"):
    """
    Save a reminder to disk with full state including next trigger time.
    Called by both schedule_reminder and persistent_reminder.
    """
    saved = load_reminders()
    saved[reminder_id] = {
        "title": title,
        "message": message,
        "type": reminder_type,
        "interval_minutes": interval_minutes,
        "delay_minutes": delay_minutes,
        "next_trigger_time": next_trigger_time 
    }
    save_reminders(saved)


def remove_reminder_from_disk(reminder_id):
    """
    Remove a reminder from disk when cancelled or fired.
    Called by both schedule_reminder and persistent_reminder.
    """
    saved = load_reminders()
    if reminder_id in saved:
        del saved[reminder_id]
        save_reminders(saved)


def update_next_trigger_time(reminder_id, interval_minutes):
    """
    Update the next trigger time on disk after each firing.
    Allows correct restore after restart.
    """
    from datetime import datetime, timedelta
    saved = load_reminders()
    if reminder_id in saved:
        next_trigger = datetime.now() + timedelta(minutes=interval_minutes)
        saved[reminder_id]["next_trigger_time"] = next_trigger.isoformat()
        save_reminders(saved)

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