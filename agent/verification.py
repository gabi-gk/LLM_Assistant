'''
Checks whether a tool actually did what its return string claims
- outcomes.py classifies what the tool said - this checks whether it's true
- capture() records the relevant state before the call, verify() compares it against the state after
- only tools with a verifier can come back VERIFIED or FAILED, everything else is UNCHECKED
- a verifier that can't run returns UNCHECKED, never VERIFIED
- verify_delete_from_kb encodes the intended behaviour - highlighting a bug in the code - revisit this later

'''

import os
from pathlib import Path
from config import DEBUG, SELF_MODEL_PATH, SNOOZE_MINUTES
from agent.outcomes import VERIFIED, FAILED, UNCHECKED
from tools.files import get_default_path
from tools.notifications import load_reminders, active_reminders
from tools.knowledge import get_rag
from core.self_management import verify_self_model, split_regions

# TEXT FILES

def target_path(args):
    """
    Resolve the path the same way the file tools do so bare filenames are checked in the default save dir
    """
    return get_default_path(Path(args.get("path", "")))

def written_bytes(content):
    """
    Produces the exact bytes that the tools produces
    Changes line endings to the system's default, then encodes to bytes

    content: the string the tool was given
    returns: the bytes that should be in the file after the tool runs
    """
    return content.replace("\n", os.linesep).encode("utf-8") # replace \n with the system's line endings, then encode to bytes

def capture_file(args):
    '''
    returns the size of the file before modification to verify that append_file grew it by the right amount
    '''
    path = target_path(args)
    return {"size": path.stat().st_size if path.is_file() else 0} # size or 0 if the file didn't exist

def verify_write(args, tool_result, before):
    """
    The file must hold exactly the content that was sent
    """
    path = target_path(args)
    if str(path) not in tool_result: # the tool reported a different file than the one that was written
        return FAILED, f"the tool reported a different file than {path}"
    if not path.is_file(): # if the tool reported success but the file doesn't exist
        return FAILED, f"{path} does not exist"
    if path.read_bytes() != written_bytes(args.get("content", "")): # if the file exists but doesn't contain the content that was written
        return FAILED, f"{path} exists but does not contain the written content"
    return VERIFIED, f"{path} contains the written content" # if all checks passed, the file contains exactly what was written

def verify_append(args, tool_result, before):
    """
    The file must have grown by exactly the appended content, and end with it
    """
    path = target_path(args)
    if str(path) not in tool_result: # if the tool reported a different path than the one that was appended to
        return FAILED, f"the tool reported a different file than {path}"
    if not path.is_file(): # if the tool reported success but the file doesn't exist
        return FAILED, f"{path} does not exist"
    data = path.read_bytes() 
    added = written_bytes(args.get("content", ""))
    if len(data) != before["size"] + len(added) or not data.endswith(added): # if the file exists but doesn't have the expected size or content
        return FAILED, f"{path} did not grow by the appended content"
    return VERIFIED, f"{path} ends with the appended content" # if all checks passed, the file grew by exactly the appended content


# REMIDNERS

def capture_reminders(args):
    '''
    Read the state of active_reminders from the disk so it's a copy that can't change while the tool runs
    '''
    return load_reminders()

def reported_id(tool_result, saved):
    """
    Find which saved reminder the result names
    Titles can contain brackets and colons, so match each known id literally instead of parsing it out
    Relies on the tool's message containing "(id: <id>)" - take care when changing the tool's output format
    """
    matches = [rid for rid in saved if f"(id: {rid})" in tool_result] # find the ids of the reminders that are mentioned in the tool's result
    return max(matches, key=len) if matches else None # rare case where a title contains another title, the longer id is returned

def check_scheduled(tool_result, before, expected, minutes, replaces=None):
    """
    Check that the reported reminder is saved, running, and saved with the expected values

    expected: dictionary of expected fields and values
    minutes: the time until the remnder is meant to fire, or the repeating interval
    replaces: the id of the reminder that was changed so the verifier can check that it was removed
    """
    saved = load_reminders() # load the reminders from disk
    rid = reported_id(tool_result, saved) # find the id of the reminder the tool reported
    if rid is None: # if the reminder doesn't exist, it is not running
        if int(float(minutes) * 60) == 0: # case where the reminder might have fired as the tool was running
            return UNCHECKED, "the reminder had no delay and may have already fired"
        return FAILED, "the reported reminder id is not in the active reminders"
    if rid in before and rid != replaces: # if the reminder existed before the tool ran and it wasn't the one that was replaced, it shouldn't have been overwritten
        return FAILED, f"id {rid} already belonged to another reminder, which was overwritten"
    if rid not in active_reminders: # the reminder is saved but not running, so it won't fire
        return FAILED, f"{rid} is saved but not running"
    wrong = [field for field, value in expected.items() if saved[rid].get(field) != value] # fields whose saved value differs from what was asked for
    if wrong: # if any fields do not match the expected values, it was not saved correctly
        return FAILED, f"{rid} was saved with a different {', '.join(wrong)}"
    return VERIFIED, f"reminder {rid} is saved and running" # all checks passed

def verify_create_reminder(args, tool_result, before):
    '''
    verify that the reminder was saved and is running with the expected values
    '''
    repeat = bool(args.get("repeat", False)) # read whether it repeats
    minutes = args.get("minutes") # the time until the reminder fires, or the interval if repeating
    expected = { # the expected values the saved reminder must hold, mirror create_reminder's defaults
        "title": args.get("title"),
        "message": args.get("message"),
        "type": "persistent" if repeat else "scheduled",
        "interval_minutes" if repeat else "delay_minutes": minutes, # key name depends on type: repeating reminders store an interval, one-offs a delay
    }
    return check_scheduled(tool_result, before, expected, minutes)

def verify_edit_reminder(args, tool_result, before):
    '''
    verify that the reminder was edited and is running with the expected values
    '''
    old_id = args.get("reminder_id") # the id of the reminder that was changed
    old = before[old_id] # edit_reminder errors out if the id was missing, so it exists here
    scheduled = old.get("type") == "scheduled" # reminder type

    # mirror edit_reminder's fallbacks for fields that weren't provided
    minutes = args.get("minutes") or (old.get("delay_minutes", SNOOZE_MINUTES) if scheduled else old["interval_minutes"])
    expected = { # the expected values the saved reminder must hold, mirror edit_reminder's defaults
        "title": args.get("title") or old["title"],
        "message": args.get("message") or old["message"],
        "type": "scheduled" if scheduled else "persistent",
        "delay_minutes" if scheduled else "interval_minutes": minutes,
    }

    verdict, detail = check_scheduled(tool_result, before, expected, minutes, replaces=old_id) # check that the new reminder is saved and running with the expected values
    if verdict != VERIFIED: # if it failed, the old reminder is still there so report that instead of the new one
        return verdict, detail
    # same title in the same second gives the same id back, otherwise the old one has to be gone
    if f"(id: {old_id})" not in tool_result and (old_id in load_reminders() or old_id in active_reminders):
        return FAILED, f"the original reminder {old_id} is still active"
    return verdict, detail # all checks passed, the reminder was edited and is running with the expected values

def verify_cancel_reminder(args, tool_result, before):
    '''
    verify that the reminder was removed and is no longer running
    '''
    rid = args.get("reminder_id") # find the id of the reminder Marvin asked to cancel
    if rid in load_reminders(): # the remider is still saved, so it wasn't cancelled
        return FAILED, f"{rid} is still in the active reminders"
    if rid in active_reminders: # the reminder is still running, so it wasn't cancelled
        return FAILED, f"{rid} is still running"
    return VERIFIED, f"{rid} is no longer saved or running" # all checks passed, the reminder was cancelled


# KNOWLEDGE BASE
def indexed_sources():
    """
    Map every chunk id in the index to the file it came from to see if it's still there
    """
    data = get_rag().collection.get(include=["metadatas"]) # get the metadata for every chunk in the index
    # pair up the ids and metadatas, return the metadata's source field or "" if the source is missing
    return {cid: (meta or {}).get("source", "") for cid, meta in zip(data["ids"], data["metadatas"])} 

def capture_index(args):
    '''
    Record the state of the knowledge base index if rag is available
    '''
    return indexed_sources() if get_rag() else None # return unchecked if the base is missing

def verify_delete_from_kb(args, tool_result, before):
    """
    Nothing from the target remains, and nothing else wasremoved
    """
    target = Path(args.get("filename", ""))

    def belongs(source):
        '''
        Decides whether a chunk's source is the target file - case-insensitive on Windows, case-sensitive on Linux
        Compares the filename only, not the folder
        files in different folders all count as the target
        '''
        # exact file when an extension is given, otherwise any file with that name, case per the OS
        source = Path(source)
        if target.suffix: # if the target has an extension, match the full filename
            return os.path.normcase(source.name) == os.path.normcase(target.name)
        return os.path.normcase(source.stem) == os.path.normcase(target.name)

    after = indexed_sources() # get the state of the index after the tool ran
    left = [cid for cid, source in after.items() if belongs(source)] # find any chunks that still belong to the target file
    if left: # if any chunks were not removed, the tool failed to delete them all
        return FAILED, f"{len(left)} chunks for {target.name} are still in the index"

    removed = [cid for cid in before if cid not in after] # find the chunks that were removed from the index
    others = sorted({Path(before[cid]).name for cid in removed if not belongs(before[cid])}) # find any chunks that were removed but don't belong to the target file, return their names sorted
    if others: # if the tool removed chunks that were not meant to be removed the verification fails
        return FAILED, f"also removed chunks belonging to other files: {', '.join(others)}"
    if tool_result.upper().startswith("[SUCCESS]") and f"Removed {len(removed)} chunks" not in tool_result: # if the tool reported a different number of removed chunks than were actually removed, the verification fails
        return FAILED, f"the tool reported a different chunk count than the {len(removed)} actually removed"
    return VERIFIED, f"no chunks for {target.name} remain in the index" # all checks passed, the target file's chunks were removed


# SELF MODEL

def capture_self_model(args):
    '''
    Check whether the self model exists and record its content, so the verifier can compare if the new observation was added
    '''
    path = Path(SELF_MODEL_PATH)
    return {"text": path.read_text(encoding="utf-8") if path.exists() else None}

def verify_update_self_model(args, tool_result, before):
    """
    verification comes from self_files built in method, this only adds the check that the entry landed
    user_changed is ignored here - that's the user's edit, handled at startup by the tray
    """
    health = verify_self_model() # get the status from self_files check
    if health["code"] in ("no_file", "no_marker"): # if the self model is missing or has no marker, the update failed
        return FAILED, f"self model check failed after the update ({health['code']})"

    text = Path(SELF_MODEL_PATH).read_text(encoding="utf-8")
    _, observations = split_regions(text) # split the self model into the regions, get the observations region
    entry = f"] {args.get('observation', '')}".rstrip() # entries are "- [date] observation", date-agnostic match so skip the timesamps for matching
    if text == before["text"] or not observations.rstrip().endswith(entry): # file unchanged, or the observation isn't the last line so it wasn't saved
        return FAILED, "the observation is not the newest entry in the self model"
    if health["observations_repeat"]: # if the self model has repeated observations, the model shouldn't have saved it again
        return VERIFIED, "observation saved, but this observation is already present - do not save it again"
    return VERIFIED, "observation is the newest entry in the self model" # all checks passed, the observation was saved


# tools with a real check, anything not listed comes back UNCHECKED
VERIFIERS = {
    "write_file": verify_write,
    "create_file": verify_write,
    "append_file": verify_append,
    "create_reminder": verify_create_reminder,
    "edit_reminder": verify_edit_reminder,
    "cancel_reminder": verify_cancel_reminder,
    "delete_from_knowledge_base": verify_delete_from_kb,
    "update_self_model": verify_update_self_model,
}

# tools that only return data - the model sees the data itself, so UNCHECKED on these is logged but not shown to it
READ_ONLY_TOOLS = {
    "read_file", "find_file", "list_directory",
    "list_reminders",
    "search_knowledge_base", "search_conversation_logs",
    "list_open_windows", "list_active_window",
    "find_application",
    "search_web", "fetch_page",
    "tool_help",
}

# verifiers that need the state from before the call
CAPTURES = {
    "append_file": capture_file,
    "create_reminder": capture_reminders,
    "edit_reminder": capture_reminders,
    "delete_from_knowledge_base": capture_index,
    "update_self_model": capture_self_model,
}

def capture(tool_name, args):
    """
    Records the current state for the verifier to compare against after the tool runs

    tool_name: the tool that will run
    args: the arguments it will be run with
    returns: the snapshot, or None if the tool needs none or it couldn't be taken
    """
    if tool_name not in CAPTURES: # skip for tools that are verified from the end state alone
        return None
    try:
        return CAPTURES[tool_name](args) # run the functions that captures the state for the tool
    except Exception as e: # if a snapshot failed, the verifier will return UNCHECKED without stopping the tool
        if DEBUG:
            print(f"[DEBUG] capture for {tool_name} failed: {e}")
        return None

def verify(tool_name, args, tool_result, before=None):
    """
    Check the tool's claim against the real state, call only after it reported success

    tool_name: the tool that ran
    args: the arguments it ran with
    tool_result: the string it returned
    before: the snapshot from capture()
    returns: (VERIFIED | FAILED | UNCHECKED, detail) - never raises, a broken verifier is UNCHECKED not FAILED
    """
    try:
        if tool_name not in VERIFIERS: # if the tool has no verifier, it can't be checked
            return UNCHECKED, "this tool's outcome cannot be verified"
        if tool_name in CAPTURES and before is None: # if the tool needs a snapshot but it wasn't taken, the verifier can't run
            return UNCHECKED, "the state before the call was not recorded, outcome was not verified"
        verdict, detail = VERIFIERS[tool_name](args, tool_result, before) # run the verifier for the tool
        if verdict not in (VERIFIED, FAILED, UNCHECKED): # if the verifier returned something unexpected, treat it as UNCHECKED, otherwise it would bypass as a success
            return UNCHECKED, f"verifier returned an unknown verdict ({verdict}), outcome was not verified"
        return verdict, detail
    except Exception as e: # broad - a broken verifier must never fail a tool call
        return UNCHECKED, f"verification could not run ({e}), outcome was not verified"
