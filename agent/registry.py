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
    cancel_reminder, list_reminders, edit_reminder,
    search_knowledge_base, search_conversation_logs, update_self_model,
    list_open_windows, switch_to_window, minimize_window, maximize_window, list_active_window, close_window, 
    open_application, find_application,
    search_web, fetch_page, open_url, search_and_open,
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
        "knowledge_tools": KNOWLEDGE_TOOL_DESCRIPTIONS,
        "window_tools": WINDOW_TOOL_DESCRIPTIONS,
        "app_tools": APP_TOOL_DESCRIPTIONS,
        "browser_tools": BROWSER_TOOL_DESCRIPTIONS,
        "self_model_tools": SELF_MODEL_DESCRIPTIONS,
    }
    if group in groups:
        return groups[group]
    available = ", ".join(groups.keys()) # list available groups in error message
    return f"[ERROR] Unknown group: {group}. Available: {available}"

def get_discord_tools(search_kb_func, deep_history_func, mention_user_func):
    """
    Returns a restricted tool registry for Discord use.
    No file/shell/window tools - knowledge and history only.
    
    Called by discord_bot.py on startup with its own function references.
    """
    return {
        "search_knowledge_base": search_kb_func,
        "get_deep_history": deep_history_func,
        "mention_user": mention_user_func,
    }

DISCORD_TOOL_DESCRIPTIONS = """
You have access to these tools on Discord. Use XML format to call them:
<tool>tool_name</tool>
<args>{"argument": "value"}</args>

Available tools:
- get_deep_history - fetch deeper channel history, optionally filtered by user or time
  Use when asked about past discussions, a specific user's messages, or what happened recently
  Example (full history): <tool>get_deep_history</tool>
  <args>{"limit": 100}</args>
  Example (specific user): <tool>get_deep_history</tool>
  <args>{"limit": 100, "username_filter": "wolfiewoof"}</args>
  Example (last 2 hours): <tool>get_deep_history</tool>
  <args>{"limit": 200, "hours": 2}</args>
  Example (user in last hour): <tool>get_deep_history</tool>
  <args>{"limit": 100, "username_filter": "wolfiewoof", "hours": 1}</args>

- search_knowledge_base - search personal documents and notes
  Example: <tool>search_knowledge_base</tool>
  <args>{"query": "roosevelt quote"}</args>

- mention_user - get the Discord mention string for a server member
  Use this when asked to ping or mention someone
  The returned string like <@123456789> will automatically ping them when sent
  Example: <tool>mention_user</tool>
  <args>{"username": "wolfiewoof"}</args>

"""

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
    # Context search
    "search_knowledge_base": search_knowledge_base,
    "search_conversation_logs": search_conversation_logs,
    "update_self_model": update_self_model,
    # window management
    "list_open_windows": list_open_windows,
    "switch_to_window": switch_to_window,
    "minimize_window": minimize_window,
    "maximize_window": maximize_window,
    "list_active_window": list_active_window,
    "close_window": close_window,
    # App management
    "open_application": open_application,
    "find_application": find_application,
    # Browser commands
    "search_web": search_web,
    "fetch_page": fetch_page,
    "open_url": open_url,
    "search_and_open": search_and_open,
    # Tool help
    "tool_help": tool_help
}

# Detailed descritions, injected into the prompt when requested
FILE_TOOL_DESCRIPTIONS = """
File tools - use these to read, write and navigate the filesystem:

IMPORTANT: The path parameter is always called "path", filename is always called "filename"

- read_file - read contents of any file, accepts a full path or just a filename - when user wants to SEE the content
  Example: <tool>read_file</tool>
  <args>{"path": "notes.txt"}</args>

- write_file - write to file, shows preview and confirms first
  Example: <tool>write_file</tool>
  <args>{"path": "data/output.txt", "content": "text to write"}</args>

- append_file - append to existing file, shows preview and confirms first
  Example: <tool>append_file</tool>
  <args>{"path": "data/output.txt", "content": "text to append"}</args>

- create_file - create new file, shows preview and confirms first
  Example: <tool>create_file</tool>
  <args>{"path": "data/newfile.txt", "content": "initial content"}</args>

- list_directory - list files and folders in a directory
  Example: <tool>list_directory</tool>
  <args>{"path": "data"}</args>

- find_file - finds a file by name, return its full path - when user wants to know WHERE the file is
  Example: <tool>find_file</tool>
  <args>{"filename": "filename"}</args>
"""

# Detailed descritions, injected into the prompt when requested
SHELL_TOOL_DESCRIPTIONS = """

Shell tools - use these to run commands on the system:
- run_command - run a shell command
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

- send_notification - immediate singular desktop notification
  Example: <tool>send_notification</tool>
  <args>{"title": "Reminder", "message": "Your message here"}</args>

- schedule_reminder - one-time desktop notification after a delay
  Use when user wants a single reminder at a specific time.
  Optional: require_confirmation=true for reminders needing acknowledgement
  Optional: snooze_minutes (default 5), escalation_minutes (default 5)
  Example: <tool>schedule_reminder</tool>
  <args>{"title": "Meeting", "message": "Time to meet", "delay_minutes": 30}</args>
  Example with confirmation: 
  <tool>schedule_reminder</tool>
  <args>{"title": "Training", "message": "Workout", "delay_minutes": 60, "require_confirmation": true}</args>
  
- persistent_reminder - repeating notification at set intervals until cancelled
  Use for countdowns or repeatable information
  Optional: require_confirmation=true for reminders needing acknowledgement
  Optional: snooze_minutes (default 5), escalation_minutes (default 5)
  Example: <tool>persistent_reminder</tool>
  <args>{"title": "Break", "message": "Take a break", "interval_minutes": 60}</args>
  Example with confirmation: 
  <tool>persistent_reminder</tool>
  <args>{"title": "Meds", "message": "Take your medication", "interval_minutes": 60, "require_confirmation": true}</args>

- list_reminders - show all active reminders with their IDs
  Always call this before cancelling or editing to get the reminder ID
  Example: <tool>list_reminders</tool>
  <args>{}</args>

- cancel_reminder - stop a persistent reminder usings its ID
  Get the ID from list_reminders first
  Example: <tool>cancel_reminder</tool>
  <args>{"reminder_id": "Break_1234567890"}</args> 
  
- edit_reminder - update an active reminder
  Get the ID from list_reminders first
  Only provide the fields you want to change, others stay the same
  Example: <tool>edit_reminder</tool>
  <args>{"reminder_id": "Countdown_1234567890", "interval_minutes": 2}</args>
"""

KNOWLEDGE_TOOL_DESCRIPTIONS = """
Knowledge tools - use these to search for information not in the current conversation if the users asks for something specific or if you can't find relevant information in the current conversation:

IMPORTANT RULES:
- Only use these tools if you genuinely cannot answer from the current conversation history
- Never use these for follow-up questions where the answer is already in the conversation
- If the user asks about something specific, check and answer from the conversation history first
- After receiving search results, ALWAYS provide a direct answer immediately - do not search again unless the results were empty

- search_conversation_logs - search past conversation history, if you don't find anything relevant check the knowledge base next
  Example: <tool>search_conversation_logs</tool>
  <args>{"query": "favourite cat"}</args>

- search_knowledge_base - search personal documents, notes, PDFs and files if asked by the user or if you can't find relevant information in the logs
  Example: <tool>search_knowledge_base</tool>
  <args>{"query": "DPO evaluation metrics"}</args>
"""

WINDOW_TOOL_DESCRIPTIONS = """
Window management tools - use these to control open windows and applications:

- list_open_windows - list all currently open windows
  Example: <tool>list_open_windows</tool>
  <args>{}</args>

- list_active_window - get the currently focused window
  Example: <tool>list_active_window</tool>
  <args>{}</args>

- switch_to_window - bring a window to focus (partial title match supported)
  Example: <tool>switch_to_window</tool>
  <args>{"title": "Discord"}</args>

- minimize_window - minimize a window
  Example: <tool>minimize_window</tool>
  <args>{"title": "Discord"}</args>

- maximize_window - maximize a window
  Example: <tool>maximize_window</tool>
  <args>{"title": "Discord"}</args>

- close_window - close a window
  Example: <tool>close_window</tool>
  <args>{"title": "Discord"}</args>
"""

APP_TOOL_DESCRIPTIONS = """
Application management tools - use these to find and open applications on the system:

- open_application - open an application by name or common shortcut
  Example: <tool>open_application</tool>
  <args>{"name": "settings"}</args>

- find_application - search for an installed application by name, useful when not sure of the exact app name or path
  Example: <tool>find_application</tool>
  <args>{"name": "discord"}</args>
"""

BROWSER_TOOL_DESCRIPTIONS = """
Browser tools - use these to search the web and manage browser tabs:

- search_web - search the web and return results as text
  Use when asked about current events, facts or anything needing web lookup
  Example: <tool>search_web</tool>
  <args>{"query": "latest Python releases"}</args>

- fetch_page - fetch and return the text content of a webpage
  Use when asked to read a specific URL or get details from a page
  Example: <tool>fetch_page</tool>
  <args>{"url": "https://docs.python.org"}</args>

- open_url - open a URL in the user's default browser
  Use when asked to open or navigate to a website
  Example: <tool>open_url</tool>
  <args>{"url": "https://youtube.com"}</args>

- search_and_open - search and open the top result in the browser
  Use when asked to find and open something
  Example: <tool>search_and_open</tool>
  <args>{"query": "Python documentation"}</args>
"""

# Marvin's personal file
SELF_MODEL_DESCRIPTIONS = """
Self model tools - use these to read and update your own knowledge about yourself and Fenn:

- update_self_model - append a new observation to your self model file
  Use when you learn something new about yourself, Fenn, or your relationship worth keeping
  Be specific and factual - write what you actually learned not vague summaries
  Example: <tool>update_self_model</tool>
  <args>{"observation": "Fenn prefers concise answers and dislikes filler phrases"}</args>

  Example: <tool>update_self_model</tool>
  <args>{"observation": "Fenn's favourite cat is a black panther"}</args>
"""

# Main description, always injected into the prompt
TOOL_DESCRIPTIONS = """
You have access to external tools. Call them using this exact XML-like syntax, replacing the name and arguments as needed:
<tool>tool_name</tool>
<args>{"argument": "value"}</args>

ALWAYS use the XML format above to actually execute tools.
If you want to use a tool, output the XML immediately - do not explain first.
Make sure to follow the syntax exactly, or the tool call will fail. If your tool call fails, check the syntax and try again.

Always call tool_help first to see detailed descriptions and examples for each tool group before using them
<tool>tool_help</tool>
<args>{"group": "notification_tools"}</args>

Avaliable tool groups - call tool_help first to see full syntax details and example:
- file_tools: read_file, write_file, append_file, create_file, list_directory, find_file
- shell_tools: run_command
- notification_tools: send_notification, schedule_reminder, persistent_reminder, cancel_reminder, edit_reminder, list_reminders
- knowledge_tools: search_conversation_logs, search_knowledge_base
- window_tools: list_open_windows, switch_to_window, minimize_window, maximize_window, list_active_window, close_window
- app_tools: open_application, find_application
- browser_tools: search_web, fetch_page, open_url, search_and_open
- self_model_tools: update_self_model
"""