'''
Containts the list of tools and their descriptions for the model
- the TOOLS dict maps tool name strings to actual functions
- the TOOL_DESCRIPTIONS string is injected into the prompt to explain how to use the tools and their syntax
- the tool_help function returns detailed descriptions and examples for each tool group when the model calls it
'''

from tools import (
    read_file, write_file, append_file, create_file, list_directory, find_file,
    run_command,
    send_notification, schedule_reminder, persistent_reminder,
    cancel_reminder, list_reminders, edit_reminder
)

def tool_help(group):
    """
    Return full descriptions and examples for a tool group

    group: string name of the tool group, e.g. "file_tools"
    returns: string description of the tools in that group, or error if group not found
    """
    groups = {
        "file_tools": FILE_TOOL_DESCRIPTIONS,
        "shell_tools": SHELL_TOOL_DESCRIPTIONS,
        "notification_tools": NOTIFICATION_TOOL_DESCRIPTIONS,
    }
    if group in groups:
        return groups[group]
    available = ", ".join(groups.keys()) # list available groups in error message
    return f"[ERROR] Unknown group: {group}. Available: {available}"

# maps tool name strings to actual functions
# agent loop uses this to execute tool calls by name
TOOLS = {
    # file
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "create_file": create_file,
    "find_file": find_file,
    "list_directory": list_directory,
    # Shell
    "run_command": run_command,
    # Notification
    "send_notification": send_notification,
    "schedule_reminder": schedule_reminder,
    "persistent_reminder": persistent_reminder,
    "cancel_reminder": cancel_reminder,
    "list_reminders": list_reminders,
    "edit_reminder": edit_reminder,
    # Tool help
    "tool_help": tool_help
}

# Detailed descritions, injected into the prompt when requested
FILE_TOOL_DESCRIPTIONS = """
File tools — use these to read, write and navigate the filesystem:

- read_file — read contents of any file, accepts a full path or just a filename - when user wants to SEE the content
  Example: <tool>read_file</tool>
  <args>{"path": "notes.txt"}</args>

- write_file — write to file, shows preview and confirms first
  Example: <tool>write_file</tool>
  <args>{"path": "data/output.txt", "content": "text to write"}</args>

- append_file — append to existing file, shows preview and confirms first
  Example: <tool>append_file</tool>
  <args>{"path": "data/output.txt", "content": "text to append"}</args>

- create_file — create new file, shows preview and confirms first
  Example: <tool>create_file</tool>
  <args>{"path": "data/newfile.txt", "content": "initial content"}</args>

- list_directory — list files and folders in a directory
  Example: <tool>list_directory</tool>
  <args>{"path": "data"}</args>

- find_file - finds a file by name, return its full path - when user wants to know WHERE the file is
  Example: <tool>find_file</tool>
  <args>{"filename": "filename"}</args>
"""

# Detailed descritions, injected into the prompt when requested
SHELL_TOOL_DESCRIPTIONS = """

Shell tools - use these to run commands on the system:
- run_command — run a shell command
  Destructive commands (rm, del, format) are fully blocked
  Modifying commands (pip, git, mkdir) require confirmation and show the command
  Read-only commands (dir, ls, type) run directly
  Example: <tool>run_command</tool>
  <args>{"cmd": "dir data"}</args>
"""

# Detailed descritions, injected into the prompt when requested
NOTIFICATION_TOOL_DESCRIPTIONS = """
Notification tools - use these for reminders and desktop notifications:

IMPORTANT RULES: 
- Reminder ID's are generated automatically each time you create a reminder, they look like "title_1234567890"
- They are NOT the same as the remainder title
- Only call list_reminders if you need an ID for cancel or edit operations
- Never call list_reminders before creating a new reminder
- If a tool call succeeds, report the result and stop

- send_notification — immediate singular desktop notification
  Example: <tool>send_notification</tool>
  <args>{"title": "Reminder", "message": "Your message here"}</args>

- schedule_reminder — one-time desktop notification after a delay
  Use when user wants a single reminder at a specific time.
  Example: <tool>schedule_reminder</tool>
  <args>{"title": "Meeting", "message": "Time to meet", "delay_minutes": 30}</args>

- persistent_reminder — repeating notification at set intervals until cancelled
  Use for countdowns or repeatable information
  Example: <tool>persistent_reminder</tool>
  <args>{"title": "Break", "message": "Take a break", "interval_minutes": 60}</args>

- list_reminders — show all active reminders with their IDs
  Always call this before cancelling or editing to get the reminder ID
  Example: <tool>list_reminders</tool>
  <args>{}</args>

- cancel_reminder — stop a persistent reminder usings its ID
  Get the ID from list_reminders first
  Example: <tool>cancel_reminder</tool>
  <args>{"reminder_id": "Break_1234567890"}</args> 
  
- edit_reminder — update an active reminder
  Get the ID from list_reminders first
  Only provide the fields you want to change, others stay the same
  Example: <tool>edit_reminder</tool>
  <args>{"reminder_id": "Countdown_1234567890", "interval_minutes": 2}</args>
"""

# Main description, always injected into the prompt
TOOL_DESCRIPTIONS = """
You have access to external tools. Call them using this exact XML-like syntax, replacing the name and arguments as needed:
<tool>tool_name</tool>
<args>{"argument": "value"}</args>

ALWAYS use the XML format above to actually execute tools.
If you want to use a tool, output the XML immediately - do not explain first.

Avaliable tool groups - call tool_help first to see full syntax details and example:
- file_tools: read_file, write_file, append_file, create_file, list_directory, find_file
- shell_tools: run_command
- notification_tools: send_notification, schedule_reminder, persistent_reminder, cancel_reminder, edit_reminder, list_reminders

To get help on a group:
<tool>tool_help</tool>
<args>{"group": "notification_tools"}</args>
"""