'''
Provides self management functions for Marvin's self model
- Snapshot and restore functions
- Verification checks for runaway observations and user changes
- callable update function for the model

The self file is a markdown file with two regions, user made notes above the obserations marker and model written content below it
'''

from pathlib import Path
import shutil
from datetime import datetime
from config import SELF_MODEL_PATH, SELF_MODEL_START, SELF_MODEL_END, SELF_MODEL_ACK, SELF_BACKUP_DIR, OBSERVATIONS_MARKER, DEBUG

# how many identical/near-identical consecutive observations counts as runaway
REPEAT_THRESHOLD = 3
# soft flag if the observations list grows beyond this - nudge towards a review but not an error
OBSERVATIONS_SOFT_LIMIT = 40
# how many last snapshots of the self model to keep
MAX_BACKUPS = 3

def inject_self_model(history, prepend=False):
    """
    Inject Marvin's self model into conversation history.

    history: the conversation history list to inject into
    prepend: True inserts at the start (startup/restore), False appends (after clear)
    returns: True if injected, False if file not found
    """
    path = Path(SELF_MODEL_PATH)
    if not path.exists():
        return False

    history[:] = [m for m in history if not m.get("content", "").startswith(SELF_MODEL_START)] # strip the message so future calls replace rather than duplicate

    self_model = path.read_text(encoding="utf-8")
    user_msg = {"role": "user", "content": f"{SELF_MODEL_START}\n[Startup context - your self model]\n{self_model}\n{SELF_MODEL_END}"} # mark of start and end
    assistant_msg = {"role": "assistant", "content": SELF_MODEL_ACK}

    if prepend:
        history.insert(0, assistant_msg)
        history.insert(0, user_msg)
    else:
        history.append(user_msg)
        history.append(assistant_msg)

    print("[SELF MODEL] Injected into context")
    return True

def snapshot_files():
    '''
    Return existing snapshot files, newest first (sorted by timestamped filename).
    '''
    backup_dir = Path(SELF_BACKUP_DIR)
    if not backup_dir.exists():
        return []
    # timestamped names sort chronologically, reverse for newest-first
    return sorted(backup_dir.glob("marvin_self_*.md"), reverse=True)

def snapshot_self_model():
    '''
    Save a copy of the current self-model into the backup dir, IF it differs from the most recent snapshot

    returns: True if a snapshot was written, False if unchanged or file missing
    '''
    live = Path(SELF_MODEL_PATH)
    if not live.exists():
        print("[SELF BACKUP] Live self-model not found, nothing to snapshot")
        return False

    backup_dir = Path(SELF_BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)

    live_text = live.read_text(encoding="utf-8")

    # only snapshot if the file changed since the most recent snapshot
    existing = snapshot_files()
    if existing:
        newest_text = existing[0].read_text(encoding="utf-8")
        if newest_text == live_text:
            return False # unchanged, no new snapshot needed

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = backup_dir / f"marvin_self_{timestamp}.md" # timestamp the backups
    shutil.copy2(live, dest) # copy2 to preserve metadata like modified time
    print(f"[SELF BACKUP] Snapshot saved: {dest.name}")

    # (re-list so the new one is included and delete any old ones beyond the max
    all_snaps = snapshot_files() # newest first
    for old in all_snaps[MAX_BACKUPS:]:
        old.unlink()
        print(f"[SELF BACKUP] Removed old snapshot: {old.name}")

    return True

def restore_self_model():
    '''
    Restore the live self-model from the most recent snapshot if the file fails a hard check

    returns: True if restored, False if no snapshot available
    '''
    existing = snapshot_files() # newest first
    if not existing:
        print("[SELF BACKUP] No snapshot to restore from")
        return False

    newest = existing[0]
    shutil.copy2(newest, Path(SELF_MODEL_PATH))
    print(f"[SELF BACKUP] Restored self-model from snapshot: {newest.name}")
    return True

def split_regions(text):
    '''
    Split self-model text into (user-content, observations) at the marker
    Returns (user_str, observations_str)
    '''
    if OBSERVATIONS_MARKER not in text:
        return text, None
    head, _, tail = text.partition(OBSERVATIONS_MARKER)
    return head, tail


def observation_lines(observations_text):
    '''
    Pull the actual observation entries (non-empty lines starting with '-') from the observations part
    '''
    if not observations_text:
        return []
    return [ln.strip() for ln in observations_text.splitlines() if ln.strip().startswith("-")] # format starts with "- [date]"

# Note only similar lines no contextually similar ones
def check_repeats(obs_lines):
    '''
    Detect repetition: the same observation (ignores the date)
    repeated REPEAT_THRESHOLD+ times consecutively
    Returns True if repetition is detected
    '''
    def strip_date(line):
        # remove the - date part and compare the actual observations
        if "]" in line:
            return line.split("]", 1)[1].strip()
        return line

    streak = 1 # count how many times the same observation has been repeated in a row
    for i in range(1, len(obs_lines)):
        if strip_date(obs_lines[i]) == strip_date(obs_lines[i - 1]):
            streak += 1
            if streak >= REPEAT_THRESHOLD:
                return True
        else:
            streak = 1
    return False

def verify_self_model():
    '''
    Run all checks against the live self-model and generate a status report dict
    app.py decides what to do with this
    '''
    status = {
        "code": "ok", # primary result
        "user_changed": False, # what the human edited part was changed
        "user_diff": None, # if changed, the (snapshot_user, live_user) for review
        "observations_repeat": False, # whether the model wrote the same observation repeatedly
        "observations_long": False, # whether the observations section is getting long 
        "ok": True, # overall status flag, set to False if any critical issues are detected
    }

    live_path = Path(SELF_MODEL_PATH)
    if not live_path.exists():
        status.update(code="no_file", ok=False) # no file detected
        return status

    live_text = live_path.read_text(encoding="utf-8")
    live_user, live_obs = split_regions(live_text) # divide into the human and model sections

    if live_obs is None:
        status.update(code="no_marker", ok=False) # if no model marker found
        return status

    obs_lines = observation_lines(live_obs) # get the stripped observation lines for checks
    if check_repeats(obs_lines):
        status.update(observations_repeat=True, code="observations_repeat", ok=False) # check for repeats
    if len(obs_lines) > OBSERVATIONS_SOFT_LIMIT:
        status.update(observations_long=True)  # check for length

    snapshots = snapshot_files()  # newest first
    if not snapshots:
        # No error on the first run
        if status["code"] == "ok":
            status["code"] = "no_snapshot"
        return status

    snap_text = snapshots[0].read_text(encoding="utf-8")
    snap_user, _ = split_regions(snap_text) # get the human part of the snapshot for comparison

    if snap_user.strip() != live_user.strip():
        status.update(user_changed=True, user_diff=(snap_user, live_user), ok=False)
        # user change is the headline if nothing worse is going on
        if status["code"] in ("ok", "no_snapshot"):
            status["code"] = "user_changed"

    return status

# Called by the model via his tool registry
def update_self_model(observation):
    """
    Append a new observation to Marvin's self model, below the OBSERVATIONS_MARKER.
    Called by the model without user input.

    observation: the fact or observation to save
    returns: success or error message
    """
    path = Path(SELF_MODEL_PATH)

    if not path.exists():
        return f"[ERROR] Self model not found at {SELF_MODEL_PATH}"

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[ERROR] Could not read self model: {e}"

    # the marker defines where observations begin, lets the user curate the self model above that line
    if OBSERVATIONS_MARKER not in content:
        return (f"[ERROR] Self model is missing the '{OBSERVATIONS_MARKER}' section marker. "
                f"Cannot safely add an observation.")

    timestamp = datetime.now().strftime("%Y-%m-%d")
    entry = f"- [{timestamp}] {observation}" # Timestamp his observations

    # split into curated region (before marker, untouched) and observations (after)
    head, _, tail = content.partition(OBSERVATIONS_MARKER)
    # tail is everything after the marker, including the existing observation lines
    new_tail = tail.rstrip() + "\n" + entry + "\n"
    new_content = head + OBSERVATIONS_MARKER + new_tail

    try:
        path.write_text(new_content, encoding="utf-8")
        if DEBUG:
            print(f"[SELF MODEL] Updated: {observation}")
        return f"[SUCCESS] Noted: {observation}"
    except Exception as e:
        return f"[ERROR] Could not update self model: {e}"