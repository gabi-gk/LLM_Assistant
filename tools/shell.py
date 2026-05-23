'''
Provides a secure interface for running shell commands with risk-based gating
    - BLOCKED: destructive commands, never runs
    - CONFIRM: modifying commands, shows command and asks approval
    - SAFE: read-only commands, runs directly
- Classifies commands based on their first word
- Uses subprocess to execute commands and capture output
- Called by the assistant to run cmd commands
'''

import subprocess
from pathlib import Path
from core.utils import confirm

# fully blocked destructive commands
BLOCKED_COMMANDS = {
    'rm', 'rmdir', 'format', 'del', 'mkfs', 'fdisk',
    'shutdown', 'reboot', 'rd', ':(){:|:&};:'
}

# Commands that require confirmation before proceeding
CONFIRM_COMMANDS = {
    'pip', 'pip3', 'mkdir', 'move', 'copy', 'xcopy', 'robocopy',
    'git', 'npm', 'cargo', 'apt', 'pacman', 'mv', 'cp',
    'python', 'py', 'powershell', 'bash', 'sh', 'reg', 'schtasks',
    # network commands that send data or open connections
    'curl', 'wget', 'ssh', 'ftp', 'sftp', 'nc', 'netcat', 'net'
}


def run_command(cmd):
    """
    Run a shell command with risk-based gating

    cmd: string of the command to run
    returns: string of the command output
    """
    risk = classify(cmd) # determine the risk level of the command based on its first word

    if risk == "blocked":
        return f"[BLOCKED] '{cmd}' is disabled"

    if risk == "confirm":
        print(f"\n[SHELL] Command to run:\n  {cmd}")
        if not confirm("Run this command?"):
            return "[CANCELLED] Command cancelled"

    try:
        result = subprocess.run( # execute the command and capture output
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30 # prevent hanging commands
        )
        output = result.stdout # present the normal output 
        if result.stderr: # if there is any error output, include it in the response
            output += f"\n[STDERR]\n{result.stderr}"
        return output if output.strip() else "[Command completed with no output]"
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out after 30 seconds"
    except Exception as e:
        return f"[ERROR] Command failed: {e}"


def classify(cmd):
    """
    Classify a command as blocked, confirm or safe
    Checks the first word - the actual executable being called

    cmd: string of the command to classify
    returns: "blocked", "confirm" or "safe"
    """
    first_word = cmd.strip().split()[0].lower()
    # strip path so C:\\windows\\system32\\cmd.exe → cmd
    first_word = Path(first_word).stem.lower()

    if first_word in BLOCKED_COMMANDS:
        return "blocked"
    if first_word in CONFIRM_COMMANDS:
        return "confirm"
    return "safe"