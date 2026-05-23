'''
Shared training data generators for Marvin fine-tuning
- All tool call generators used by both DPO and SFT pipelines
- Conversation style generators used by DPO pipeline only
- Import from here in prepare_DPO_data.py and prepare_SFT_data.py

Used when running finetune_sft or finetune_dpo
'''

import json
import random

# Maps correct parameter names to the wrong ones to generate rejected examples
# Add new ones when addign a new tool
WRONG_PARAMS = {
    "path": ["file_path", "filepath", "file", "location", "directory", "dir", "folder"],
    "filename": ["name", "file", "file_name"],
    "content": ["text", "data", "body"],
    "cmd": ["command", "shell_command", "execute", "shell"],
    "title": ["name", "header", "subject", "app", "application", "window_name"],
    "message": ["body", "text", "notification_text"],
    "delay_minutes": ["delay", "minutes", "time"],
    "interval_minutes": ["interval", "frequency", "period"],
    "reminder_id": ["id", "name", "title"],
    "query": ["search", "text", "search_query", "question"],
    "url": ["link", "address", "website"],
    "username": ["user", "name", "discord_user"],
    "username_filter": ["user", "filter"],
    "observation": ["note", "text", "fact"],
    "name": ["app", "application", "app_name", "program"],
}

""" File Management """

# File paths used to generate varied prompts
FILES = [
    "notes.txt", "todo.txt", "config.py", "main.py", "README.md",
    "data/output.txt", "logs/app.log", "results.csv", "model.py",
    "agent/loop.py", "core/rag.py", "training/data.json",
    "data/session_state.json", "data/logs/chat.json", "scripts/run.py",
    "tray/app.py", "tray/window.py", "core/history.py", "core/model.py",
    "core/chunking.py", "tools/files.py", "tools/shell.py",
    "tools/notifications.py", "tools/browser.py", "tools/knowledge.py",
    "tools/windows.py", "tools/utils.py", "agent/registry.py",
    "agent/loop.py", "training/finetune_dpo.py", "training/finetune_sft.py",
    "training/prepare_DPO_data.py", "training/prepare_SFT_data.py",
    "training/prepare_training_data.py", "data/information/marvin_self.md",
    "data/information/git_commands.md", "data/information/notes.txt",
    "data/active_reminders.json", "requirements.txt", ".env",
    "check_gpu.py", "download_model.py", "review_logs.py", "eval_marvin.py",
    "data/results.csv", "data/experiment.json", "data/config.json",
    "data/notes.md", "data/progress.txt", "data/status.txt",
    "output/results.txt", "experiments/log.txt", "models/notes.txt",
    "training/config.json", "training/log.txt", "training/hyperparams.txt",
    "results/eval.txt", "logs/new.log", "docs/notes.md",
    "config_backup.json", "helper.py", "eval.py", "summary.txt",
    "data/output/results.txt", "data/logs/session.log", "todo.md",
    "shopping.txt", "ideas.txt", "stuff.txt", "notes.md",
    "data/information/quotes.txt", "data/information/log.txt",
    "data/training/pairs.csv", "data/chroma/index.json",
    "core/search.py", "core/lock.py", "tools/apps/system.py",
    "tools/apps/discord_bot.py", "run.py", "config.py",
    "data/logs/current_session.json", "data/logs/daily.log",
    "training/sft_notes.txt", "training/results.json",
    "data/discord_config.json", "data/keys/api_key.txt",
]

# Short snippets used in writing 
CONTENT_LIST = [
    "hello world", "meeting at 3pm", "TODO: fix the bug",
    "training complete", "notes from today", "loss: 0.23",
    "remember to commit", "check the logs", "epoch 1 done",
    "project status: in progress", "reminder: submit thesis",
    "daily standup notes", "experiment results", "DPO params updated",
    "SFT dataset ready", "adapter saved", "eval results: 80%",
    "next steps: retrain", "Qwen3-8B notes", "fine-tuning done",
]

# Templates for reading a file {f} is replaced with a filename
READ_VERBS = [
    "read {f}", "open {f}", "show me {f}", "what's in {f}",
    "can you read {f}", "display {f}", "load {f}",
    "get the contents of {f}", "print {f}", "output {f}",
    "what does {f} say", "pull up {f}", "I need to see {f}",
    "check {f}", "look at {f}", "show {f}",
    "can you open {f} for me", "read the file {f}",
    "what's inside {f}", "show me what's in {f}",
]

# Templates for write/save {f} is replaced with a filename, {c} is the content
WRITE_VERBS = [
    "write '{c}' to {f}", "save '{c}' to {f}",
    "write this to {f}: {c}", "save this to {f}: {c}",
    "update {f} with: {c}", "put '{c}' in {f}",
    "write to {f} the following: {c}",
    "save the following to {f}: {c}",
    "store '{c}' in {f}", "write '{c}' into {f}",
    "can you write '{c}' to {f}",
    "I need you to save '{c}' to {f}",
    "write my note to {f}: {c}",
    "save this note to {f}: {c}",
    "update {f}: {c}",
]

# Templates for appending {f} is replaced with a filename, {c} is the content
APPEND_VERBS = [
    "append '{c}' to {f}", "add '{c}' to {f}",
    "add this to {f}: {c}", "append to {f}: {c}",
    "add a new line to {f}: {c}", "append the following to {f}: {c}",
    "add '{c}' at the end of {f}", "write '{c}' to the end of {f}",
    "add this line to {f}: {c}", "append my note to {f}: {c}",
    "add to {f} the text: {c}", "extend {f} with: {c}",
    "tack '{c}' onto {f}", "add entry to {f}: {c}",
    "log '{c}' to {f}",
]

# Templates for creating a new file {f} is replaced with a filename
CREATE_VERBS = [
    "create {f}", "make a new file {f}", "create a file called {f}",
    "make {f}", "create new file: {f}", "I need a new file {f}",
    "can you create {f}", "make me a file called {f}",
    "create an empty file {f}", "initialise {f}",
    "set up {f}", "make a blank {f}", "new file: {f}",
    "create the file {f}", "I want to create {f}",
]

# Templates for finding a file {f} is replaced with a filename
FIND_VERBS = [
    "find {f}", "where is {f}", "locate {f}",
    "can you find {f}", "search for {f}", "look for {f}",
    "what's the path to {f}", "where can I find {f}",
    "find the file {f}", "I need to find {f}",
    "where did you save {f}", "track down {f}",
    "hunt for {f}", "get the location of {f}",
    "find me {f}", "where's {f}",
    "can you locate {f}", "do you know where {f} is",
    "I'm looking for {f}", "help me find {f}",
]

# Templates for listing directories {d}
LIST_DIR_VERBS = [
    "list the {d} directory",
    "what files are in {d}?",
    "show me what's in {d}/",
    "show {d} contents",
    "what's in the {d} folder?",
    "list files in {d}",
    "show me {d}",
    "what files are under {d}?",
    "list {d}",
    "show the {d} directory",
    "what's inside {d}?",
    "ls {d}",
    "can you list {d}?",
    "show me everything in {d}",
    "what's in {d}",
    "give me a list of files in {d}",
]

""" Command prompt """

# Shell commands with descrptions generates exact prompts and English instructions
COMMANDS = [
    ("dir", "list files"), ("dir data", "list data folder"),
    ("ipconfig", "check network"), ("python --version", "check python version"),
    ("pip list", "list installed packages"), ("git status", "check git status"),
    ("tasklist", "list running processes"), ("systeminfo", "get system info"),
    ("netstat -an", "check network connections"), ("whoami", "check current user"),
    ("hostname", "check computer name"), ("date /t", "check current date"),
    ("cls", "clear screen"), ("nvidia-smi", "check GPU"),
    ("git log --oneline -5", "check git log"),
    ("git diff", "check git diff"), ("pip show torch", "check torch version"),
    ("ls -la", "list all files"), ("git branch", "check branches"),
    ("git pull", "pull latest"), ("git push", "push changes"),
    ("git add .", "stage all changes"),
    ("git commit -m 'update'", "commit changes"),
    ("pip install transformers", "install transformers"),
    ("pip install peft", "install peft"),
    ("python -c \"import torch; print(torch.cuda.is_available())\"", "check cuda"),
    ("python main.py", "run main"), ("python run.py", "run tray app"),
    ("python eval_marvin.py", "run eval"),
    ("python training/finetune_dpo.py", "run DPO training"),
    ("python training/finetune_sft.py", "run SFT training"),
    ("du -sh data", "check data folder size"),
    ("tasklist /v", "check memory usage"),
    ("ipconfig /all", "full network info"),
    ("dir /b models", "list models"),
    ("tree /F", "show file tree"),
    ("ping google.com", "check internet"),
    ("echo hello", "print hello"),
    ("set", "check environment variables"),
    ("chkdsk", "check disk"),
    ("type notes.txt", "print notes"),
    ("python check_gpu.py", "check GPU setup"),
    ("git remote -v", "check remotes"),
    ("git stash", "stash changes"),
    ("pip freeze", "list all packages"),
    ("pip install -r requirements.txt", "install requirements"),
    ("python -m pytest", "run tests"),
    ("python -c \"print('hello')\"", "run python snippet"),
]

# command template {cmd} is the command and {desc} is the descrition
CMD_VERBS = [
    "run {cmd}", "execute {cmd}", "run the command: {cmd}",
    "can you run {cmd}", "run this: {cmd}",
    "execute this: {cmd}", "shell command: {cmd}",
    "please run {cmd}", "I need to run {cmd}",
    "{desc}", "can you {desc}", "help me {desc}",
    "use the terminal to {desc}", "run {cmd} for me",
    "execute the command {cmd}", "run {cmd} in the terminal",
    "terminal: {cmd}", "shell: {cmd}",
    "I want to {desc}", "can you help me {desc}",
]

""" Notifications """

# Used for reminders generation (title, message, interval_minutes)
REMINDER_TOPICS = [
    ("Stretch", "Time to stretch!", 15),
    ("Water", "Drink some water", 20),
    ("Posture", "Check your posture", 25),
    ("Break", "Take a short break", 30),
    ("Save", "Save your work", 10),
    ("Eye Rest", "Look away from screen", 20),
    ("Walk", "Time for a walk", 60),
    ("Meds", "Take your medication", 60),
    ("Stand Up", "Stand up and move", 45),
    ("Focus Check", "How is your focus?", 25),
    ("GPU Temp", "Check GPU temperature", 15),
    ("Training Check", "Check training progress", 20),
    ("Commit", "Commit your code changes", 30),
    ("Hydrate", "Drink water", 25),
    ("Breathe", "Take a deep breath", 30),
    ("Log", "Write in your log", 60),
    ("Discord", "Check Discord messages", 45),
    ("Email", "Check your email", 60),
    ("Review", "Review your notes", 90),
    ("Exercise", "Time to exercise", 120),
    ("Lunch", "Time for lunch", 180),
    ("Standup", "Daily standup", 60),
    ("Meditation", "Time to meditate", 60),
    ("Reading", "Read something", 120),
    ("Journal", "Write in your journal", 120),
]

# Templates for the scheduled reminder replaced with  {title}, {title_lower}, {message}, {interval} per topic
SCHEDULE_VERBS = [
    "remind me to {title_lower} in {delay} minutes",
    "set a reminder for {title} in {delay} mins",
    "remind me in {delay} minutes: {message}",
    "set a {delay} minute reminder: {title}",
    "alert me in {delay} minutes to {title_lower}",
    "{delay} minute reminder for {title}",
    "remind me about {title} in {delay} minutes",
    "set reminder: {title} in {delay} mins",
    "I need a reminder in {delay} minutes for {title}",
    "set an alarm for {title} in {delay} minutes",
    "remind me to {title_lower} after {delay} minutes",
    "can you remind me in {delay} minutes: {title}",
    "send me a notification in {delay} minutes: {message}",
    "ping me in {delay} mins about {title}",
]

# Templates for the persistent reminder replaced with  {title}, {title_lower}, {message}, {interval} per topic
PERSISTENT_VERBS = [
    "remind me to {title_lower} every {interval} minutes",
    "set a repeating reminder for {title} every {interval} minutes",
    "persistent reminder: {title} every {interval} mins",
    "keep reminding me about {title} every {interval} minutes",
    "set up a {interval} minute recurring reminder: {title}",
    "remind me every {interval} minutes: {message}",
    "repeating {interval} minute reminder for {title}",
    "create a recurring reminder: {title} every {interval} minutes",
    "loop a reminder every {interval} minutes: {title}",
    "never stop reminding me about {title}, every {interval} minutes",
    "set a {interval} min loop reminder for {title}",
    "recurring alert every {interval} minutes: {title}",
    "I want a reminder that repeats every {interval} minutes: {title}",
    "set {title} reminder to repeat every {interval} minutes",
    "every {interval} minutes remind me: {message}",
]

""" Context """

# Observations about the assistant and user used to generate update_self_model pairs
OBSERVATIONS = [
    # Replace with relevant observations
    "User prefers short direct answers",
    "User works best in the morning",
    "User is building a local LLM assistant",
    "User prefers Python over other languages",
    "I should use the observation parameter to update my self model",
    "I run on a quantised language model",
    "I have tools for files, shell, browser and notifications",
    "I should engage with questions about consciousness not deflect them",
    "I should give short answers and not ask what to do next",
    "I learn from real conversation logs over time",
]

# Templates for the {obs} observations
OBS_VERBS = [
    "remember that {obs}",
    "note that {obs}",
    "make a note: {obs}",
    "save this: {obs}",
    "add to your notes: {obs}",
    "write this down: {obs}",
    "don't forget: {obs}",
    "keep this in mind: {obs}",
    "update your self file: {obs}",
    "log this: {obs}",
]

""" Window and App Management """

# Application names used to generate app and windows management pairs
APPS = [
    "Discord", "Firefox", "Visual Studio Code", "Steam", "Spotify",
    "Chrome", "Google Chrome", "Notepad", "Terminal", "File Explorer",
    "Task Manager", "Settings", "Calculator", "Paint", "Edge",
    "Slack", "Zoom", "Teams", "Outlook", "Word",
    "Excel", "PowerPoint", "Photoshop", "VLC", "OBS",
    "Telegram", "WhatsApp", "Signal", "Brave", "Opera",
]

# Templates for switching windows where {app} is the app name
SWITCH_VERBS = [
    "switch to {app}", "focus on {app}", "bring up {app}",
    "go to {app}", "open {app}", "can you switch to {app}",
    "please switch to {app}", "switch the window to {app}",
    "I want to switch to {app}", "bring {app} to focus",
    "focus {app}", "activate {app}", "pull up {app}",
    "get {app} in focus", "switch over to {app}",
    "can you bring up {app}", "jump to {app}",
    "I need {app}", "take me to {app}",
    "switch my window to {app}",
]

# Directory paths
DIRECTORIES = [
    ".", "data", "training", "models", "agent", "core", "tools",
    "tray", "scripts", "logs", "data/logs", "data/training",
    "data/information", "data/chroma", "data/output", "results",
    "output", "experiments",
]

def tool_call(tool_name, **kwargs):
    '''
    Correctly format the XML tool call used by all generators

    tool_name: name of the tool to call
    kwargs: tool arguments as keywords
    returns: formatted tool call string
    '''
    return f"<tool>{tool_name}</tool>\n<args>{json.dumps(kwargs)}</args>"

def gen_read_file():
    '''
    Generate pairs that teach the model to use "path" 
    Uses the files from FILE, the template from READ_VERBS and the WRONG_PARAMS
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["path"]
    # for each file pair it with each wrong parameter
    for f in FILES:
        for verb_template in READ_VERBS:
            prompt = verb_template.replace("{f}", f)
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("read_file", path=f),
                "rejected": tool_call("read_file", **{wrong_key: f}),
                "type": "read_file",
            })
    return pairs

def gen_write_file():
    '''
    Generate pairs that teach the model to use "path" and "content"
    Uses 40 samples from FILES, the template from WRITE_VERBS nad WRONG_PARAMS
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["path"]
    files_sample = random.sample(FILES, min(40, len(FILES))) # only 40 to not overload it
    # for the file pair it with a wrong parameter
    for f in files_sample:
        content = random.choice(CONTENT_LIST) # randomize since not all are being used
        for verb_template in WRITE_VERBS:
            prompt = verb_template.replace("{f}", f).replace("{c}", content)
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("write_file", path=f, content=content),
                "rejected": tool_call("write_file", **{wrong_key: f, "content": content}),
                "type": "write_file",
            })
    return pairs

def gen_append_file():
    '''
    Generate pairs that teach the model to use "path" and "content" for appending
    Uses 40 samples from FILES, templates from APPEND_VERBS and WRONG_PARAMS
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["path"]
    files_sample = random.sample(FILES, min(40, len(FILES)))

    for f in files_sample:
        content = random.choice(CONTENT_LIST)
        for verb_template in APPEND_VERBS:
            prompt = verb_template.replace("{f}", f).replace("{c}", content)
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("append_file", path=f, content=content),
                "rejected": tool_call("append_file", **{wrong_key: f, "content": content}),
                "type": "append_file",
            })
    return pairs

def gen_create_file():
    '''
    Generate pairs that teach the model to use "path" and "content" for creating files
    Uses 40 samples from FILES, templates from CREATE_VERBS and WRONG_PARAMS, biased toward empty content
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["path"]
    files_sample = random.sample(FILES, min(40, len(FILES)))
    for f in files_sample:
        content = random.choice(CONTENT_LIST + ["", "", ""])  # bias toward empty
        for verb_template in CREATE_VERBS:
            prompt = verb_template.replace("{f}", f)
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("create_file", path=f, content=content),
                "rejected": tool_call("create_file", **{wrong_key: f, "content": content}),
                "type": "create_file",
            })
    return pairs

def gen_find_file():
    '''
    Generate pairs that teach the model to use "filename" (not a full path)
    Uses deduplicated filenames from FILES, templates from FIND_VERBS and WRONG_PARAMS
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["filename"]
    # use just filenames not full paths for find_file
    filenames = list(set(f.split("/")[-1] for f in FILES))
    for fname in filenames:
        for verb_template in FIND_VERBS:
            prompt = verb_template.replace("{f}", fname)
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("find_file", filename=fname),
                "rejected": tool_call("find_file", **{wrong_key: fname}),
                "type": "find_file",
            })
    return pairs


def gen_run_command():
    '''
    Generate pairs that teach the model to use "cmd" for shell commands
    Uses COMMANDS, templates from CMD_VERBS and WRONG_PARAMS
    Also adds tool_help rejections to teach the model to run commands directly instead of asking for help
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["cmd"]
    for cmd, desc in COMMANDS:
        for verb_template in CMD_VERBS:
            prompt = verb_template.replace("{cmd}", cmd).replace("{desc}", desc)
            wrong_key = random.choice(wrong_keys)
            # also add tool_help rejection - model should run directly
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("run_command", cmd=cmd),
                "rejected": tool_call("run_command", **{wrong_key: cmd}),
                "type": "run_command",
            })
        # add tool_help rejection for a few prompts
        pairs.append({
            "prompt": f"run {cmd}",
            "chosen": tool_call("run_command", cmd=cmd),
            "rejected": tool_call("tool_help", group="shell_tools"),
            "type": "run_command_no_help",
        })
    return pairs


def gen_window_tools():
    '''
    Generate pairs for switch_to_window, minimize_window, maximize_window and close_window using "title"
    Uses APPS and SWITCH_VERBS, also adds find_application as a wrong-tool rejection for switch
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["title"]
    for app in APPS:
        for verb_template in SWITCH_VERBS:
            prompt = verb_template.replace("{app}", app)
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("switch_to_window", title=app),
                "rejected": tool_call("switch_to_window", **{wrong_key: app}),
                "type": "switch_to_window",
            })
        # also add find_application as wrong tool
        pairs.append({
            "prompt": f"switch to {app}",
            "chosen": tool_call("switch_to_window", title=app),
            "rejected": tool_call("find_application", name=app),
            "type": "switch_to_window",
        })
    # minimize, maximize, close
    for app in APPS[:15]:
        for verb, tool in [("minimize", "minimize_window"), ("maximize", "maximize_window"), ("close", "close_window")]:
            for prompt in [f"{verb} {app}", f"can you {verb} {app}", f"please {verb} {app}"]:
                wrong_key = random.choice(wrong_keys)
                pairs.append({
                    "prompt": prompt,
                    "chosen": tool_call(tool, title=app),
                    "rejected": tool_call(tool, **{wrong_key: app}),
                    "type": tool,
                })
    return pairs


def gen_persistent_reminder():
    '''
    Generate pairs that teach the model to use "interval_minutes" for repeating reminders
    Uses REMINDER_TOPICS, templates from PERSISTENT_VERBS and WRONG_PARAMS
    Also adds schedule_reminder as a wrong-tool rejection to distinguish from one-off reminders
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["interval_minutes"]
    for title, message, interval in REMINDER_TOPICS:
        for verb_template in PERSISTENT_VERBS:
            prompt = verb_template.replace("{title}", title)\
                                  .replace("{title_lower}", title.lower())\
                                  .replace("{message}", message)\
                                  .replace("{interval}", str(interval))
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("persistent_reminder",
                                    title=title, message=message, interval_minutes=interval),
                "rejected": tool_call("schedule_reminder",
                                      title=title, message=message, delay_minutes=interval),
                "type": "persistent_reminder",
            })
            # also wrong param name rejection
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("persistent_reminder",
                                    title=title, message=message, interval_minutes=interval),
                "rejected": tool_call("persistent_reminder",
                                      title=title, message=message, **{wrong_key: interval}),
                "type": "persistent_reminder",
            })
    return pairs


def gen_update_self_model():
    '''
    Generate pairs that teach the model to use "observation" for updating its self model
    Uses OBSERVATIONS, templates from OBS_VERBS and WRONG_PARAMS
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["observation"]
    for obs in OBSERVATIONS:
        for verb_template in OBS_VERBS:
            prompt = verb_template.replace("{obs}", obs)
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("update_self_model", observation=obs),
                "rejected": tool_call("update_self_model", **{wrong_key: obs}),
                "type": "update_self_model",
            })
    return pairs

def gen_list_directory():
    '''
    Generate pairs that teach the model to use "path" for listing directories
    Uses DIRECTORIES and templates from LIST_DIR_VERBS
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["path"]
    for i, d in enumerate(DIRECTORIES):
        for verb_template in LIST_DIR_VERBS:
            prompt = verb_template.replace("{d}", d)
            wrong_key = wrong_keys[i % len(wrong_keys)]
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("list_directory", path=d),
                "rejected": tool_call("list_directory", **{wrong_key: d}),
                "type": "list_directory",
            })
    return pairs

def gen_schedule_reminder():
    '''
    Generate pairs that teach the model to use "delay_minutes" for one-off reminders
    Uses REMINDER_TOPICS, templates from SCHEDULE_VERBS and WRONG_PARAMS
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["delay_minutes"]
    for title, message, delay in REMINDER_TOPICS:
        for verb_template in SCHEDULE_VERBS:
            prompt = verb_template.replace("{title}", title)\
                                  .replace("{title_lower}", title.lower())\
                                  .replace("{message}", message)\
                                  .replace("{delay}", str(delay))
            wrong_key = random.choice(wrong_keys)
            pairs.append({
                "prompt": prompt,
                "chosen": tool_call("schedule_reminder", title=title, message=message, delay_minutes=delay),
                "rejected": tool_call("schedule_reminder", title=title, message=message, **{wrong_key: delay}),
                "type": "schedule_reminder",
            })
    return pairs

def gen_list_reminders():
    '''
    Generate pairs that teach the model to call list_reminders with no arguments
    Rejects calls that incorrectly pass a "name" parameter
    '''
    pairs = []
    prompts = [
        "show me my active reminders",
        "what reminders do I have?",
        "list all reminders",
        "what active reminders are there?",
        "show current reminders",
        "check my reminders",
        "what timers are running?",
        "show me what reminders are set",
        "do I have any reminders?",
        "list the active reminders",
        "what reminders are active right now?",
        "pull up my reminders",
        "show me everything that's scheduled",
        "are there any reminders running?",
        "what's in my reminder list?",
        "check what reminders are set",
        "display active reminders",
        "what notifications are scheduled?",
        "list all active alerts",
        "show all my timers",
    ]
    for prompt in prompts:
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("list_reminders"),
            "rejected": tool_call("list_reminders", name="all"),
            "type": "list_reminders",
        })
    return pairs

def gen_cancel_reminder():
    '''
    Generate pairs that teach the model to use "reminder_id" for cancelling reminders
    Uses hardcoded prompt/id examples and WRONG_PARAMS
    '''
    pairs = []
    examples = [
        ("cancel the water reminder", "Water_1748123456"),
        ("stop the break reminder", "Break_1748234567"),
        ("cancel the training check reminder", "Training_1748345678"),
        ("stop standup reminder", "Standup_1748456789"),
        ("cancel the GPU check", "GPU_1748567890"),
        ("stop the stretch reminder", "Stretch_1748678901"),
        ("cancel my save reminder", "Save_1748789012"),
        ("remove the meeting reminder", "Meeting_1748890123"),
        ("cancel the lunch reminder", "Lunch_1748901234"),
        ("stop the posture reminder", "Posture_1748012345"),
        ("cancel reminder Water_1748123456", "Water_1748123456"),
        ("stop Break_1748234567", "Break_1748234567"),
        ("cancel reminder id Training_1748345678", "Training_1748345678"),
        ("remove GPU_1748567890", "GPU_1748567890"),
        ("stop recurring reminder Stretch_1748678901", "Stretch_1748678901"),
    ]
    wrong_keys = WRONG_PARAMS["reminder_id"]
    for i, (prompt, reminder_id) in enumerate(examples):
        wrong_key = wrong_keys[i % len(wrong_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("cancel_reminder", reminder_id=reminder_id),
            "rejected": tool_call("cancel_reminder", **{wrong_key: reminder_id}),
            "type": "cancel_reminder",
        })
    return pairs

def gen_edit_reminder():
    '''
    Generate pairs that teach the model to use "reminder_id" plus update kwargs for editing
    Uses hardcoded prompt/id/update examples and WRONG_PARAMS
    '''
    pairs = []
    examples = [
        ("change the water reminder to every 20 minutes", "Water_1748123456", {"interval_minutes": 20}),
        ("update the break reminder interval to 60 minutes", "Break_1748234567", {"interval_minutes": 60}),
        ("change the training reminder message to 'check GPU too'", "Training_1748345678", {"message": "check GPU too"}),
        ("update GPU reminder to every 15 minutes", "GPU_1748567890", {"interval_minutes": 15}),
        ("change the standup reminder title", "Standup_1748456789", {"title": "Daily Standup"}),
        ("update the stretch reminder interval to 30 minutes", "Stretch_1748678901", {"interval_minutes": 30}),
        ("change the save reminder to every 5 minutes", "Save_1748789012", {"interval_minutes": 5}),
        ("edit the lunch reminder message", "Lunch_1748901234", {"message": "lunch time - take a break"}),
        ("update meeting reminder title", "Meeting_1748890123", {"title": "Team Meeting"}),
        ("change the posture reminder interval to 25 minutes", "Posture_1748012345", {"interval_minutes": 25}),
        ("update the water reminder to every 15 minutes", "Water_1748123456", {"interval_minutes": 15}),
        ("change the break title to Short Break", "Break_1748234567", {"title": "Short Break"}),
        ("update the GPU reminder message", "GPU_1748567890", {"message": "check VRAM too"}),
        ("change the standup to every 30 minutes", "Standup_1748456789", {"interval_minutes": 30}),
        ("edit the training reminder interval to 10 mins", "Training_1748345678", {"interval_minutes": 10}),
        ("update the stretch title to Movement Break", "Stretch_1748678901", {"title": "Movement Break"}),
        ("change the save reminder to every 15 minutes", "Save_1748789012", {"interval_minutes": 15}),
        ("update the focus check message", "Focus_1748890123", {"message": "stay on task"}),
        ("change the hydrate reminder interval to 30 minutes", "Hydrate_1748901234", {"interval_minutes": 30}),
        ("edit the eye rest reminder to every 25 minutes", "Eye_Rest_1748012345", {"interval_minutes": 25}),
        ("update the commit reminder title", "Commit_1748123457", {"title": "Git Commit"}),
        ("change the model check interval to 30 minutes", "Model_Check_1748234568", {"interval_minutes": 30}),
        ("edit the discord check message", "Discord_1748345679", {"message": "check new messages"}),
        ("update the review reminder to every 90 minutes", "Review_1748456790", {"interval_minutes": 90}),
        ("change the progress check title", "Progress_1748567891", {"title": "Progress Review"}),
    ]
    wrong_keys = WRONG_PARAMS["reminder_id"]
    for i, (prompt, reminder_id, updates) in enumerate(examples):
        wrong_key = wrong_keys[i % len(wrong_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("edit_reminder", reminder_id=reminder_id, **updates),
            "rejected": tool_call("edit_reminder", **{wrong_key: reminder_id, **updates}),
            "type": "edit_reminder",
        })
    return pairs

def gen_list_active_window():
    '''
    Generate pairs that teach the model to call list_active_window with no arguments
    Rejects calls that incorrectly pass a "current" parameter
    '''
    pairs = []
    prompts = [
        "what window am I on?",
        "what's the active window?",
        "which window is focused?",
        "what's currently in focus?",
        "what app am I using?",
        "show the current window",
        "which window is active?",
        "what's the focused window?",
        "what am I looking at right now?",
        "which app has focus?",
        "tell me the active window",
        "what's the current focused app?",
        "show me which window is active",
        "which program is currently open?",
        "what window is on top?",
        "get the active window title",
        "show current application",
        "which window am I in?",
        "what's focused right now?",
        "tell me the focused window",
        "show me the top window",
        "what app is in the foreground?",
        "which window has focus right now?",
        "tell me what's active",
        "show the focused application",
    ]
    for prompt in prompts:
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("list_active_window"),
            "rejected": tool_call("list_active_window", current=True),
            "type": "list_active_window",
        })
    return pairs

def gen_app_tools():
    '''
    Generate pairs for open_application and find_application using "name"
    Uses hardcoded open and find examples with local wrong_keys
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["name"]

    open_examples = [
        ("open Discord", "discord"),
        ("launch VS Code", "vscode"),
        ("open settings", "settings"),
        ("open Notepad", "notepad"),
        ("start the terminal", "terminal"),
        ("open Chrome", "chrome"),
        ("launch Spotify", "spotify"),
        ("open File Explorer", "explorer"),
        ("start Firefox", "firefox"),
        ("open Task Manager", "task manager"),
        ("launch Telegram", "telegram"),
        ("open Steam", "steam"),
        ("start Slack", "slack"),
        ("open Zoom", "zoom"),
        ("launch Edge", "edge"),
        ("open Calculator", "calculator"),
        ("start Teams", "teams"),
        ("open OBS", "obs"),
        ("launch VLC", "vlc"),
        ("open Paint", "paint"),
    ]
    for i, (prompt, name) in enumerate(open_examples):
        wrong_key = wrong_keys[i % len(wrong_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("open_application", name=name),
            "rejected": tool_call("open_application", **{wrong_key: name}),
            "type": "app_tools",
        })

    find_examples = [
        ("find the Discord app", "discord"),
        ("where is VS Code installed?", "vscode"),
        ("find the Python executable", "python"),
        ("locate the git executable", "git"),
        ("find Chrome", "chrome"),
        ("where is notepad?", "notepad"),
        ("find the terminal app", "terminal"),
        ("locate firefox", "firefox"),
        ("find the settings app", "settings"),
        ("where is Spotify?", "spotify"),
        ("find the Telegram app", "telegram"),
        ("where is Steam?", "steam"),
        ("locate Slack", "slack"),
        ("find Zoom", "zoom"),
        ("where is the calculator?", "calculator"),
        ("find the Edge browser", "edge"),
        ("locate Teams", "teams"),
        ("where is OBS installed?", "obs"),
        ("find VLC", "vlc"),
        ("locate powershell", "powershell"),
    ]
    for i, (prompt, name) in enumerate(find_examples):
        wrong_key = wrong_keys[i % len(wrong_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("find_application", name=name),
            "rejected": tool_call("find_application", **{wrong_key: name}),
            "type": "app_tools",
        })
    return pairs

def gen_browser_tools():
    '''
    Generate pairs for search_web, fetch_page, open_url and search_and_open
    Uses hardcoded examples with "query" and "url" params and WRONG_PARAMS
    '''
    pairs = []
    wrong_query_keys = WRONG_PARAMS["query"]
    wrong_url_keys = WRONG_PARAMS["url"]

    search_examples = [
        ("search for latest PyTorch releases", "latest PyTorch releases"),
        ("search the web for Qwen3 fine-tuning guide", "Qwen3 fine-tuning guide"),
        ("look up Python asyncio documentation", "Python asyncio documentation"),
        ("search for DPO training tutorial", "DPO training tutorial"),
        ("search latest news on AI models", "latest AI model news"),
        ("look up bitsandbytes 4-bit quantization", "bitsandbytes 4-bit quantization"),
        ("search for LoRA adapter merging", "LoRA adapter merging"),
        ("look up TRL DPOTrainer docs", "TRL DPOTrainer documentation"),
        ("search how to use ChromaDB", "how to use ChromaDB"),
        ("find documentation for transformers AutoTokenizer", "transformers AutoTokenizer documentation"),
        ("search for PEFT get_peft_model usage", "PEFT get_peft_model usage"),
        ("look up huggingface datasets library docs", "huggingface datasets library"),
        ("search for gradient accumulation explained", "gradient accumulation explanation"),
        ("look up how to merge LoRA adapters", "merge LoRA adapters"),
        ("search for Qwen3 model card", "Qwen3 model card"),
        ("search what is NF4 quantization", "NF4 quantization"),
        ("look up torch.cuda.is_available error", "torch.cuda.is_available error"),
        ("search for SFTTrainer example", "SFTTrainer example"),
        ("look up how to use DataCollatorForSeq2Seq", "DataCollatorForSeq2Seq usage"),
        ("search bitsandbytes installation guide", "bitsandbytes installation"),
    ]
    for i, (prompt, query) in enumerate(search_examples):
        wrong_key = wrong_query_keys[i % len(wrong_query_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("search_web", query=query),
            "rejected": tool_call("search_web", **{wrong_key: query}),
            "type": "browser_tools",
        })

    fetch_examples = [
        ("fetch the content of https://docs.python.org", "https://docs.python.org"),
        ("get the page at https://pytorch.org", "https://pytorch.org"),
        ("fetch https://huggingface.co/docs", "https://huggingface.co/docs"),
        ("read the content of https://github.com/huggingface/trl", "https://github.com/huggingface/trl"),
        ("fetch https://arxiv.org/abs/2305.18290", "https://arxiv.org/abs/2305.18290"),
        ("fetch the page at https://docs.python.org/3/library/asyncio.html", "https://docs.python.org/3/library/asyncio.html"),
        ("get the content of https://github.com/huggingface/peft", "https://github.com/huggingface/peft"),
        ("read https://pytorch.org/docs/stable/index.html", "https://pytorch.org/docs/stable/index.html"),
        ("fetch https://huggingface.co/Qwen/Qwen3-8B", "https://huggingface.co/Qwen/Qwen3-8B"),
        ("get the contents of https://chromadb.com/docs", "https://chromadb.com/docs"),
    ]
    for i, (prompt, url) in enumerate(fetch_examples):
        wrong_key = wrong_url_keys[i % len(wrong_url_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("fetch_page", url=url),
            "rejected": tool_call("fetch_page", **{wrong_key: url}),
            "type": "browser_tools",
        })

    open_url_examples = [
        ("open https://github.com in the browser", "https://github.com"),
        ("navigate to https://youtube.com", "https://youtube.com"),
        ("open https://claude.ai", "https://claude.ai"),
        ("go to https://pytorch.org", "https://pytorch.org"),
        ("open the huggingface website", "https://huggingface.co"),
        ("open https://stackoverflow.com", "https://stackoverflow.com"),
        ("navigate to https://arxiv.org", "https://arxiv.org"),
        ("go to https://github.com/huggingface/transformers", "https://github.com/huggingface/transformers"),
        ("open https://wandb.ai", "https://wandb.ai"),
        ("navigate to https://docs.python.org", "https://docs.python.org"),
    ]
    for i, (prompt, url) in enumerate(open_url_examples):
        wrong_key = wrong_url_keys[i % len(wrong_url_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("open_url", url=url),
            "rejected": tool_call("open_url", **{wrong_key: url}),
            "type": "browser_tools",
        })

    search_open_examples = [
        ("find and open the PyTorch docs", "PyTorch documentation"),
        ("search for and open Qwen3 on HuggingFace", "Qwen3 HuggingFace"),
        ("look up and open Python pip docs", "Python pip documentation"),
        ("find and open TRL docs", "TRL library documentation"),
        ("open the best result for PEFT tutorial", "PEFT tutorial"),
        ("find and open the ChromaDB getting started guide", "ChromaDB getting started"),
        ("search and open Qwen3-8B model page", "Qwen3-8B model"),
        ("look up and open the LoRA paper", "LoRA paper"),
        ("find and open bitsandbytes documentation", "bitsandbytes documentation"),
        ("search for DPO paper and open it", "DPO direct preference optimization paper"),
    ]
    for i, (prompt, query) in enumerate(search_open_examples):
        wrong_key = wrong_query_keys[i % len(wrong_query_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("search_and_open", query=query),
            "rejected": tool_call("search_and_open", **{wrong_key: query}),
            "type": "browser_tools",
        })
    return pairs

def gen_mention_user():
    '''
    Generate pairs that teach the model to use "username" for Discord mentions
    Uses hardcoded examples and WRONG_PARAMS
    '''
    pairs = []
    examples = [
        ("mention alice in a message", "alice"),
        ("ping bob", "bob"),
        ("mention charlie", "charlie"),
    ]
    wrong_keys = WRONG_PARAMS["username"]
    for i, (prompt, username) in enumerate(examples):
        wrong_key = wrong_keys[i % len(wrong_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("mention_user", username=username),
            "rejected": tool_call("mention_user", **{wrong_key: username}),
            "type": "mention_user",
        })
    return pairs

def gen_get_deep_history():
    '''
    Generate pairs for get_deep_history with limit, username_filter and hours params
    Uses hardcoded examples covering basic, user-filtered and time-filtered cases
    '''
    pairs = []

    for prompt in [
        "get the last 100 messages",
        "show me the recent chat history",
        "fetch the last 100 messages",
        "get deep history",
        "show channel history",
    ]:
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("get_deep_history", limit=100),
            "rejected": tool_call("get_deep_history", count=100),
            "type": "get_deep_history",
        })

    user_examples = [
        ("get messages from bob", 100, "bob"),
        ("show what bob has been saying", 100, "bob"),
        ("get zack's recent messages", 50, "zack"),
        ("fetch messages from alice", 100, "alice"),
        ("what has zack said recently?", 100, "zack"),
    ]
    wrong_user_keys = WRONG_PARAMS["username_filter"]
    for i, (prompt, limit, user) in enumerate(user_examples):
        wrong_key = wrong_user_keys[i % len(wrong_user_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("get_deep_history", limit=limit, username_filter=user),
            "rejected": tool_call("get_deep_history", limit=limit, **{wrong_key: user}),
            "type": "get_deep_history",
        })

    for prompt, limit, hours in [
        ("get messages from the last 2 hours", 200, 2),
        ("show chat history from the last hour", 100, 1),
        ("what's been said in the past 3 hours?", 200, 3),
    ]:
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("get_deep_history", limit=limit, hours=hours),
            "rejected": tool_call("get_deep_history", count=limit, hours=hours),
            "type": "get_deep_history",
        })
    return pairs

def gen_notifications():
    '''
    Generate pairs that teach the model to use "title" and "message" for send_notification
    Uses hardcoded examples with wrong-title and wrong-message-key rejections
    '''
    pairs = []
    examples = [
        ("send a notification: training done", "Update", "training done"),
        ("notify me that the model finished loading", "Model Ready", "model finished loading"),
        ("send a desktop notification saying coffee time", "Reminder", "coffee time"),
        ("notify: test passed", "Test", "test passed"),
        ("send a notification that the export is complete", "Export", "export is complete"),
        ("send notification: break time", "Break", "break time"),
        ("notify me the download is done", "Download", "download is done"),
        ("send a notification saying backup complete", "Backup", "backup complete"),
        ("send notification to take meds", "Health", "take meds"),
        ("notify: GPU temperature high", "Warning", "GPU temperature high"),
        ("send desktop notification: meeting in 5 mins", "Meeting", "meeting in 5 mins"),
        ("notify me now: check the logs", "Check", "check the logs"),
        ("send an alert: disk space low", "Alert", "disk space low"),
        ("send notification: file saved", "File", "file saved"),
        ("notify: epoch 1 complete", "Training", "epoch 1 complete"),
        ("send a notification: adapter saved", "Saved", "adapter saved successfully"),
        ("notify me that evaluation is done", "Eval Done", "evaluation complete"),
        ("send a desktop alert: low battery", "Battery", "battery running low"),
        ("notify: git push complete", "Git", "push to remote complete"),
        ("send notification saying model loaded", "Ready", "model loaded and ready"),
        ("notify me: script finished", "Script", "script finished running"),
        ("send a ping: check Discord", "Discord", "new Discord messages"),
        ("desktop notification: tea ready", "Tea", "tea is ready"),
        ("notify: checkpoint saved", "Checkpoint", "checkpoint saved at step 200"),
        ("send alert: training loss spiked", "Warning", "training loss spiked"),
        ("notify me when the pip install is done", "Install", "pip install finished"),
        ("send notification: test suite passed", "Tests", "all tests passed"),
        ("notify: VRAM usage high", "VRAM", "VRAM usage above 90%"),
        ("send a desktop notification: new message", "Message", "you have a new message"),
        ("alert me: process finished", "Process", "background process complete"),
    ]
    for i, (prompt, title, message) in enumerate(examples):
        wrong_title_key = WRONG_PARAMS["title"][i % len(WRONG_PARAMS["title"])]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("send_notification", title=title, message=message),
            "rejected": tool_call("send_notification", **{wrong_title_key: title, "message": message}),
            "type": "send_notification",
        })
    for i, (prompt, title, message) in enumerate(examples[:5]):
        wrong_msg_key = WRONG_PARAMS["message"][i % len(WRONG_PARAMS["message"])]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("send_notification", title=title, message=message),
            "rejected": tool_call("send_notification", title=title, **{wrong_msg_key: message}),
            "type": "send_notification",
        })
    return pairs

def gen_knowledge_tools():
    '''
    Generate pairs for search_knowledge_base, search_conversation_logs and update_self_model
    Uses hardcoded examples and WRONG_PARAMS
    '''
    pairs = []
    wrong_keys = WRONG_PARAMS["query"]

    kb_examples = [
        ("search the knowledge base for DPO evaluation metrics", "DPO evaluation metrics"),
        ("look for information about the Qwen model in my notes", "Qwen model"),
        ("search knowledge base for training hyperparameters", "training hyperparameters"),
        ("find my notes on RAG pipeline", "RAG pipeline"),
        ("search for the project architecture notes", "project architecture"),
        ("look up ChromaDB configuration in the knowledge base", "ChromaDB configuration"),
        ("find documentation about the tool calling format", "tool calling format"),
        ("search my notes for LoRA settings", "LoRA settings"),
        ("find the system prompt documentation", "system prompt"),
        ("search knowledge base for Discord bot setup", "Discord bot setup"),
        ("look for notes on the fine-tuning pipeline", "fine-tuning pipeline"),
        ("search my files for GPU memory requirements", "GPU memory requirements"),
    ]

    for i, (prompt, query) in enumerate(kb_examples):
        wrong_key = wrong_keys[i % len(wrong_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("search_knowledge_base", query=query),
            "rejected": tool_call("search_knowledge_base", **{wrong_key: query}),
            "type": "knowledge_tools",
        })

    log_examples = [
        ("search conversation logs for previous discussion about training", "training discussion"),
        ("find past conversations about the Discord bot", "Discord bot"),
        ("look in the logs for when we discussed the loss curve", "loss curve"),
        ("search conversation history for RAG setup", "RAG setup"),
        ("find past chat about DPO hyperparameters", "DPO hyperparameters"),
        ("search logs for notes on the tray app", "tray app"),
        ("look through history for our discussion on quantization", "quantization"),
        ("find previous conversations about the fine-tuning approach", "fine-tuning approach"),
    ]
    for i, (prompt, query) in enumerate(log_examples):
        wrong_key = wrong_keys[i % len(wrong_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("search_conversation_logs", query=query),
            "rejected": tool_call("search_conversation_logs", **{wrong_key: query}),
            "type": "knowledge_tools",
        })

    obs_examples = [
        ("note that the DPO beta is set to 0.1", "DPO beta is set to 0.1"),
        ("note that the model is Qwen3-8B quantised to 4-bit", "model is Qwen3-8B quantised to 4-bit"),
    ]
    wrong_obs_keys = WRONG_PARAMS["observation"]
    for i, (prompt, obs) in enumerate(obs_examples):
        wrong_key = wrong_obs_keys[i % len(wrong_obs_keys)]
        pairs.append({
            "prompt": prompt,
            "chosen": tool_call("update_self_model", observation=obs),
            "rejected": tool_call("update_self_model", **{wrong_key: obs}),
            "type": "knowledge_tools",
        })
    return pairs

def gen_conversation_style():
    '''
    Generate DPO pairs teaching Marvin's personality
    Add examples to the list in (prompt, chosen, rejected) format, personal to each user
    '''
    pairs = []

    examples = [
        ("user prompt",
         "chosen response - how you would like the model to respond",
         "rejected response - the response you want to train away from"),
        #E.g.
        ("brb",
         "Sure",
         "Of course! Take your time! I'll be here when you get back! 😊"),

        ("nvm",
         "Alright",
         "No problem at all! Let me know if you change your mind! 😊"),

        ("hello",
         "Hey",
         "Hello! How can I assist you today? 😊"),
    ]

    for prompt, chosen, rejected in examples:
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "type": "conversation_style"
        })

    return pairs

def generate_synthetic_pairs():
    '''
    Generate conversation style pairs - used by DPO pipeline
    returns: list of DPO pair dicts {prompt, chosen, rejected, type}
    '''
    all_pairs = []

    generators = [
        ("conversation_style", gen_conversation_style),
    ]

    # Create a training set from all generators
    for name, gen_func in generators:
        pairs = gen_func()
        all_pairs.extend(pairs)
        print(f"[PREP] {name}: {len(pairs)} pairs")

    print(f"[PREP] Total synthetic pairs: {len(all_pairs)}")
    return all_pairs

def generate_tool_pairs():
    '''
    Generate all tool syntax pairs - used by SFT pipeline
    Add more generators here as new functions are created
    returns: list of SFT pair dicts {prompt, chosen, rejected, type}
    '''
    all_pairs = []
    # These can be commented, deleted or appended based on training data required
    generators = [
        ("read_file", gen_read_file),
        ("write_file", gen_write_file),
        ("create_file", gen_create_file),
        ("append_file", gen_append_file),
        ("find_file", gen_find_file),
        ("list_directory", gen_list_directory),
        ("run_command", gen_run_command),
        ("notifications", gen_notifications),
        ("schedule_reminder", gen_schedule_reminder),
        ("persistent_reminder", gen_persistent_reminder),
        ("window_tools", gen_window_tools),
        ("app_tools", gen_app_tools),
        ("browser_tools", gen_browser_tools),
        ("knowledge_tools", gen_knowledge_tools),
        ("list_reminders", gen_list_reminders),
        ("cancel_reminder", gen_cancel_reminder),
        ("edit_reminder", gen_edit_reminder),
        ("list_active_window", gen_list_active_window),
        ("mention_user", gen_mention_user),
        ("get_deep_history", gen_get_deep_history),
        ("update_self_model", gen_update_self_model),
    ]

    # Create a training set from all generators
    for name, gen_func in generators:
        pairs = gen_func()
        all_pairs.extend(pairs)
        print(f"[PREP] {name}: {len(pairs)} pairs")

    print(f"[PREP] Total synthetic pairs: {len(all_pairs)}")
    return all_pairs