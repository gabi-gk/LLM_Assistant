'''
Every tool call gets three questions:
    - Classification - What did the tool say happened? (OK, ERROR, BLOCKED, CANCELLED)
    - Verification - Is it actually true? (VERIFIED, FAILED, UNCHECKED)
    - Enforcement - errors for the model and a system note for the user
'''

# Classification - what a tool's return string says happened
OK = "OK"
ERROR = "ERROR"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"

# Loop-side failures - no return string exists to classify
UNKNOWN_TOOL = "UNKNOWN_TOOL" # name not in TOOLS
MISSING_ARGS = "MISSING_ARGS" # <args> block absent
JSON_ERROR = "JSON_ERROR" # <args> present but unparseable
EXECUTION_FAILED = "EXECUTION_FAILED" # tool raised while running

# Verification - whether what the tool claimed actually happened (see agent/verification.py)
VERIFIED = "VERIFIED" # the claimed outcome was checked against the real state and matches
FAILED = "FAILED" # the claimed outcome was checked and does NOT match
UNCHECKED = "UNCHECKED" # no way to check the claim, never treat as verified

def check_output(tool_result):
    """
    Classify what a tool actually returned

    tool_result: whatever was returned from the main loop after a tool call
    returns: a status string
    """
    tool_result = str(tool_result).lstrip() # string for the error check

    if tool_result.startswith("[ERROR]"):
        status = ERROR
    elif tool_result.startswith("[BLOCKED]"):
        status = BLOCKED
    elif tool_result.startswith("[CANCELLED]"):
        status = CANCELLED
    else:
        status = OK
    return status

def annotate(tool_result, verdict, detail):
    """
    Build the result string the model sees, so he cannot report a tool as sucess if it failed verification

    tool_result: the raw string the tool returned
    verdict: VERIFIED, FAILED or UNCHECKED
    detail: short reason from the verifier
    returns: the string to feed back to the model
    """
    if verdict == FAILED:
        # lead with the error so the original [SUCCESS] can't be repeated as fact, but keep it visible
        return f"[ERROR] Verification failed: {detail}\nThe tool reported: {tool_result}"
    return f"{tool_result}\n[{verdict}] {detail}"

def format_issues(issues):
    """
    Turn the issues run_agent collected into a short system note for the user
    Shown after Marvin's reply and never added to history, so he can't learn to copy it

    issues: list of strings filled by run_agent
    returns: the note, or None if nothing went wrong
    """
    if not issues:
        return None
    return "Tool issues this request:\n" + "\n".join(f"- {issue}" for issue in issues)
