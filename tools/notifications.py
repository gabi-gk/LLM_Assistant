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
import tkinter as tk
from pathlib import Path
from plyer import notification
from config import REMINDERS_FILE
from datetime import datetime, timedelta


# active persistent reminders- key: reminder_id, value: stop Event
active_reminders = {}

# registers when the notification is not acknowledged
escalation = None

# main tkinter root - must be set via set_tk_root() before popups are shown
tk_root = None


def set_escalation(callback):
    """
    Register a callback for reminder escalation that connects the notifcations to the tray app
    callback: function(title, message) that shows chat and injects escalation message
    """
    global escalation
    escalation = callback


def set_tk_root(root):
    """
    Register the main tkinter Tk instance so popups can be scheduled on the main thread
    """
    global tk_root
    tk_root = root


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
        next_trigger_str = data.get("next_trigger_time")

        # calculate the next trigger time taking into account system shutdown
        if next_trigger_str:
            next_trigger = datetime.fromisoformat(next_trigger_str)
            remaining_seconds = (next_trigger - datetime.now()).total_seconds()
            remaining_seconds = max(0, remaining_seconds)
        else:
            remaining_seconds = 0

        # Determine notification type
        if data.get("type") == "scheduled":
            remaining_minutes = remaining_seconds / 60
            schedule_reminder(
                data["title"],
                data["message"],
                remaining_minutes if remaining_minutes > 0 else 0.1,
                require_confirmation=data.get("require_confirmation", False),
                snooze_minutes=data.get("snooze_minutes", 5),
                escalation_minutes=data.get("escalation_minutes", 5)
            )
        else:
            persistent_reminder(
                data["title"],
                data["message"],
                data["interval_minutes"],
                reminder_id=reminder_id,
                initial_delay_seconds=remaining_seconds,
                require_confirmation=data.get("require_confirmation", False),
                snooze_minutes=data.get("snooze_minutes", 5),
                escalation_minutes=data.get("escalation_minutes", 5)
            )


def send_notification(title, message):
    """
    Send an immediate Windows desktop notification

    title: the title of the notification
    message: the body text of the notification
    """
    try:
        notification.notify(
            title=title,
            message=message,
            app_name="Marvin",
            timeout=10
        )
        return f"[SUCCESS] Notification sent: {title}"
    except Exception as e:
        return f"[ERROR] Could not send notification: {e}"


def show_confirmation_popup(title, message, snooze_minutes=5, escalation_minutes=5):
    """
    Show an always-on-top confirmation popup on the main tkinter thread

    title: reminder title
    message: reminder message
    snooze_minutes: communicated to caller so it knows how long to wait on "snoozed"
    escalation_minutes: how long before escalation fires if popup is ignored
    returns "done", "snoozed", or "escalated".
    """
    result = [None]
    dismissed = threading.Event()

    def show():
        ''' 
        Generate a tkinker popup with the confirmation notification
        '''
        popup = tk.Toplevel(tk_root)
        popup.title("Marvin Reminder")
        popup.attributes("-topmost", True)
        popup.attributes("-alpha", 0.95)
        popup.geometry("350x160")
        popup.configure(bg="#1e1e1e")
        popup.resizable(False, False)

        def keep_on_top():
            '''
            force the popup to stay on top
            '''
            if popup.winfo_exists():
                popup.attributes("-topmost", True)
                popup.lift()
                popup.after(1000, keep_on_top)

        tk.Label(
            popup, text=f"⏰ {title}",
            font=("Consolas", 13, "bold"),
            bg="#1e1e1e", fg="#10f51b"
        ).pack(pady=(15, 5))

        tk.Label(
            popup, text=message,
            font=("Consolas", 11),
            bg="#1e1e1e", fg="#d4d4d4",
            wraplength=300
        ).pack(pady=5)

        btn_frame = tk.Frame(popup, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        def on_confirm():
            '''
            Confirm removes the popup
            '''
            result[0] = "done"
            dismissed.set()
            popup.destroy()

        def on_snooze():
            '''
            snooze moves the next notification trigger 
            '''
            result[0] = "snoozed"
            dismissed.set()
            popup.destroy()

        # confirm and snooze button generation
        tk.Button(
            btn_frame, text="Done",
            bg="#10f51b", fg="#1e1e1e",
            font=("Consolas", 11), borderwidth=0, padx=15,
            command=on_confirm
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text=f"Snooze {snooze_minutes}min",
            bg="#2d2d2d", fg="#d4d4d4",
            font=("Consolas", 11), borderwidth=0, padx=15,
            command=on_snooze
        ).pack(side=tk.LEFT, padx=5)

        keep_on_top()

    if tk_root:
        tk_root.after(0, show)
    else:
        print("[WARN] show_confirmation_popup called before set_tk_root, falling back to notification")
        send_notification(title, message)
        return "done"

    dismissed.wait(timeout=escalation_minutes * 60)

    if not dismissed.is_set():
        if escalation:
            escalation(title, message)
        return "escalated"

    return result[0]


def show_escalation_popup(title, message):
    """
    Show an urgent escalation popup on the main tkinter thread when a confirmation popup was ignored past the escalation threshold

    title: reminder title that was ignored
    message: reminder message
    """
    def show():
        ''' 
        Generate a tkinker popup with the confirmation notification
        '''
        popup = tk.Toplevel(tk_root)
        popup.title("Marvin — Ignored Reminder")
        popup.attributes("-topmost", True)
        popup.attributes("-alpha", 0.97)
        popup.geometry("380x180")
        popup.configure(bg="#2a0000")
        popup.resizable(False, False)

        def keep_on_top():
            '''
            force the popup to stay on top
            '''
            if popup.winfo_exists():
                popup.attributes("-topmost", True)
                popup.lift()
                popup.after(800, keep_on_top)

        tk.Label(
            popup, text="! REMINDER IGNORED",
            font=("Consolas", 13, "bold"),
            bg="#2a0000", fg="#ff4444"
        ).pack(pady=(15, 5))

        tk.Label(
            popup, text=f"{title}\n{message}",
            font=("Consolas", 11),
            bg="#2a0000", fg="#ffaaaa",
            wraplength=320
        ).pack(pady=5)

        def on_dismiss():
            '''
            dismiss removes the popup
            '''
            popup.destroy()

        tk.Button(
            popup, text="Got it",
            bg="#ff4444", fg="#ffffff",
            font=("Consolas", 11, "bold"), borderwidth=0, padx=20,
            command=on_dismiss
        ).pack(pady=10)

        keep_on_top()

    if tk_root:
        tk_root.after(0, show)


def schedule_reminder(title, message, delay_minutes, require_confirmation=True, snooze_minutes=5, escalation_minutes=5):
    """
    Send a single notification after a delay

    title: the title of the notification
    message: the body text of the notification
    delay_minutes: how many minutes to wait before sending the notification
    require_confirmation: if True shows a popup requiring acknowledgement instead of a plain notification
    snooze_minutes: snooze duration if popup is snoozed (only used when require_confirmation=True)
    escalation_minutes: time before escalating if popup not dismissed (only used when require_confirmation=True)
    """
    
    # remove spaces from the titles
    safe_title = title.replace(" ", "_")
    reminder_id = f"{safe_title}_{int(time.time())}"

    stop_event = threading.Event()
    active_reminders[reminder_id] = stop_event

    # calculate the next trigger time and save everything to the disk
    next_trigger = datetime.now() + timedelta(minutes=delay_minutes)
    save_reminder_to_disk(
        reminder_id, title, message,
        delay_minutes=delay_minutes,
        next_trigger_time=next_trigger.isoformat(),
        reminder_type="scheduled",
        require_confirmation=require_confirmation,
        snooze_minutes=snooze_minutes,
        escalation_minutes=escalation_minutes
    )

    def remind():
        '''
        works in the background to allow for stop interrupt
        '''
        for _ in range(int(delay_minutes * 60)):
            if stop_event.is_set():
                remove_reminder_from_disk(reminder_id)
                return
            time.sleep(1)

        if require_confirmation:
            outcome = show_confirmation_popup(
                title, message,
                snooze_minutes=snooze_minutes,
                escalation_minutes=escalation_minutes
            )
            if outcome == "snoozed":
                # snooze creates a fresh scheduled reminder
                schedule_reminder(title, message, snooze_minutes,
                                  require_confirmation=True,
                                  snooze_minutes=snooze_minutes,
                                  escalation_minutes=escalation_minutes)
        else:
            send_notification(title, message)

        if reminder_id in active_reminders:
            del active_reminders[reminder_id]
        remove_reminder_from_disk(reminder_id)

    threading.Thread(target=remind, daemon=True).start()
    return f"[SUCCESS] Reminder set for {delay_minutes} minutes: {title} (id: {reminder_id})"


def persistent_reminder(title, message, interval_minutes, reminder_id=None, initial_delay_seconds=None,
                        require_confirmation=True, snooze_minutes=5, escalation_minutes=5):
    """
    Send a notification repeatedly at set intervals until cancelled

    title: the title of the notification
    message: the text of the notification
    interval_minutes: how many minutes to wait between notifications
    reminder_id: unique ID generated if not provided, passed when restoring
    initial_delay_seconds: seconds to wait before first fire 
    require_confirmation: if True shows a popup requiring acknowledgement instead of a plain notification
    snooze_minutes: snooze duration if popup is snoozed (only used when require_confirmation=True)
    escalation_minutes: time before escalating if popup not dismissed (only used when require_confirmation=True)
    """

    # remove spaces from the titles
    if reminder_id is None:
        safe_title = title.replace(" ", "_")
        reminder_id = f"{safe_title}_{int(time.time())}"

    stop_event = threading.Event()
    active_reminders[reminder_id] = stop_event


    # calculate the next trigger time and save everything to the disk
    next_trigger = datetime.now() + timedelta(minutes=interval_minutes)
    save_reminder_to_disk(
        reminder_id, title, message,
        interval_minutes=interval_minutes,
        next_trigger_time=next_trigger.isoformat(),
        reminder_type="persistent",
        require_confirmation=require_confirmation,
        snooze_minutes=snooze_minutes,
        escalation_minutes=escalation_minutes
    )

    def remind_loop():
        '''
        works in the background to allow for stop interrupt
        '''

        # new reminders wait full time, restored calcualte their next fire tiem
        first_wait = int(interval_minutes * 60) if initial_delay_seconds is None else int(initial_delay_seconds)

        for _ in range(first_wait):
            if stop_event.is_set():
                remove_reminder_from_disk(reminder_id)
                return
            time.sleep(1)

        while not stop_event.is_set():
            if require_confirmation:
                outcome = show_confirmation_popup(
                    title, message,
                    snooze_minutes=snooze_minutes,
                    escalation_minutes=escalation_minutes
                )
            else:
                outcome = "done"
                send_notification(title, message)

            if outcome == "snoozed":
                # wait snooze_minutes then loop back to re-show the popup
                for _ in range(int(snooze_minutes * 60)):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                continue

            # resets interval on done or escalated, snooze continues the notification
            update_next_trigger_time(reminder_id, interval_minutes)
            for _ in range(int(interval_minutes * 60)):
                if stop_event.is_set():
                    break
                time.sleep(1)

        remove_reminder_from_disk(reminder_id)

    threading.Thread(target=remind_loop, daemon=True).start()
    return f"[SUCCESS] Persistent reminder started (id: {reminder_id}): {title}"


def save_reminder_to_disk(reminder_id, title, message, interval_minutes=None, delay_minutes=None,
                          next_trigger_time=None, reminder_type="persistent", require_confirmation=False,
                          snooze_minutes=5, escalation_minutes=5):
    """
    Save a reminder to disk with full state including next trigger time
    Called by both schedule_reminder and persistent_reminder
    """
    saved = load_reminders()
    saved[reminder_id] = {
        "title": title,
        "message": message,
        "type": reminder_type,
        "interval_minutes": interval_minutes,
        "delay_minutes": delay_minutes,
        "next_trigger_time": next_trigger_time,
        "require_confirmation": require_confirmation,
        "snooze_minutes": snooze_minutes,
        "escalation_minutes": escalation_minutes
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
        active_reminders[reminder_id].set()  # signals thread to stop
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

    existing = saved[reminder_id]
    # new title or keep remaining if not provided
    new_title = title or existing["title"]
    new_message = message or existing["message"]

    cancel_reminder(reminder_id)

    if existing.get("type") == "scheduled":
        # reschedule with updated values - keep remaining time if delay not provided
        new_delay = interval_minutes or existing.get("delay_minutes", 5)
        return schedule_reminder(
            new_title, new_message, new_delay,
            require_confirmation=existing.get("require_confirmation", False),
            snooze_minutes=existing.get("snooze_minutes", 5),
            escalation_minutes=existing.get("escalation_minutes", 5)
        )
    else:
        new_interval = interval_minutes or existing["interval_minutes"]
        return persistent_reminder(
            new_title, new_message, new_interval,
            require_confirmation=existing.get("require_confirmation", False),
            snooze_minutes=existing.get("snooze_minutes", 5),
            escalation_minutes=existing.get("escalation_minutes", 5)
        )


def list_reminders():
    """
    List all currently active reminders

    returns a formatted string listing all active reminders with their details
    """
    saved = load_reminders()
    if not saved:
        return "[INFO] No active reminders"
    lines = ["[ACTIVE REMINDERS]"]
    for rid, data in saved.items():
        lines.append(f" ID: {rid}")
        lines.append(f" Title: {data['title']}")
        if data.get("type") == "scheduled":
            lines.append(f" Fires in: {data.get('delay_minutes')} minutes (one-time)")
        else:
            lines.append(f" Interval: every {data['interval_minutes']} minutes")
        lines.append("")
    return "\n".join(lines)
