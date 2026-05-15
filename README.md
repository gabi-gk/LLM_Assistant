Base chat — Qwen3-8B, 4-bit, simple chat loop
RAG — ChromaDB, index a folder of your notes - long term external memory
Basic tools — file read/write, notifications, shell commands (low risk)
Agent loop — tool calling parser, confirmation system
Tray app — always-on, hotkey, hide-on-close
Window management tools — switch tabs, get open windows
Discord bot API
Conversation logger + fine-tune queue
LoRA/SFT pipeline — adapt existing DPO setup
Adapter versioning
Linux port — OS-aware tool layer, Arch-specific tools
browser extension to control active browser tabs
Let Marvin access and modify his own file describing himself for personality direction

26.04.2026
CREATED check_gpu.py - verify CUDA and VRAM before running
CREATED download_model.py - one-time download of Qwen3-8B weights to local disk
CREATED chat.py - basic chat loop, Qwen3-8B quantized to 4-bit nf4
EDIT chat.py - conversation history with context window management
EDIT chat.py - history compaction via summarisation for long conversations
EDIT chat.py - conversation saving with training queue flag on quit

27.04.2026
CREATED rag.py - ChromaDB vector database for long term external memory
EDIT rag.py - Python AST chunking splitting by function/class boundaries
EDIT rag.py - PDF chunking with adaptive splitting strategy

KNOWN ISSUES (resolved):
- PDF section retrieval — fixed with heading-aware chunking
- System prompt too restrictive — fixed with knowledge base clarification

28.04.2026
EDIT chat.py - system prompt: fixed over-restriction on general knowledge questions
EDIT chat.py - debug flag for RAG output visibility
EDIT chat.py - merged similarity + keyword search with deduplication
EDIT chat.py - dynamic k based on query length
EDIT chat.py - build_search_query adds recent conversation context to RAG search
EDIT rag.py - heading-aware chunking so section content stays together
EDIT rag.py - TOC line filtering using spaced dot pattern
EDIT rag.py - section metadata tagging on PDF chunks
EDIT rag.py - search_by_keyword with stop word filtering and partial matching
EDIT rag.py - .txt and .md similarity score boost

29.04.2026
EDIT file hierarchy changed - split code into modules:
  core/model.py — model loading and inference
  core/history.py — conversation compaction and saving
  core/rag.py — ChromaDB indexing and search
  core/chunking.py — all file chunking logic (new)
  config.py — all constants and system prompt
  main.py — entry point (replaces chat.py)

30.04.2026
CREATED tools/files.py - read, write, append, create, list directory
CREATED tools/shell.py - shell command execution with risk classification
CREATED tools/notifications.py - immediate, scheduled and persistent reminders
CREATED agent/registry.py - tool registry and descriptions for model
CREATED agent/loop.py - agent loop with tool call parsing and execution
EDIT core/history.py - load_last_session restores context on restart
EDIT tools/notifications.py - persistent reminders survive restart via disk
EDIT config.py - RESTORE_LAST_SESSION, MAX_RESTORE_MESSAGES, REMINDERS_FILE
EDIT config.py - COMPACTION_THRESHOLD, COMPACTION_KEEP_RECENT

01.05.2026
EDIT core/history.py - always save conversation on quit, mark session as closed
EDIT core/history.py - clear marks session state as cleared, skips restore on next startup  
EDIT core/history.py - load_last_session restores previous session if closed cleanly
EDIT core/history.py - save_session_state and load_session_state added
EDIT main.py - quit always saves and marks closed, clear marks cleared without saving
EDIT config.py - added SESSION_FILE path
EDIT agent/registry.py - clarified schedule_reminder vs persistent_reminder descriptions
EDIT tools/notifications.py - added edit_reminder
EDIT agent/registry.py - registered edit_reminder
EDIT agent/registry.py - reminder ID clarification, model must call list_reminders for the ID before cancel or edit

02.05.2026
EDIT agent/registry.py - lazy tool loading, base prompt now small index only
EDIT agent/registry.py - full tool details split into FILE/SHELL/NOTIFICATION groups
EDIT agent/loop.py - tool_help handled directly in agent loop

03.05.2026
CREATED tray/window.py - tkinter terminal-style dark chat window
CREATED tray/app.py - system tray icon, Alt+Space hotkey, app lifecycle
CREATED run.py - GUI entry point (main.py kept for terminal/debug use)
CREATED core/search.py - shared RAG search logic extracted from main.py
EDIT main.py - uses core/search.py
EDIT tray/app.py - uses core/search.py

04.05.2026
EDIT detailed comments added
 
08.05.2026
EDIT main.py - added always auto-save the conversation
EDIT tray/window.py - added the log display in the chatbox
EDIT agent/loop.py - added json parsing 
CREATED tools/knowledge.py - RAG is now controlled by the model
EDIT agent/registry.py - added tools to search the knowledge base
EDIT tray/app.py - removed automatic RAG

09.05.2026
CREATED tools/windows.py - window listing, switching, minimizing, maximizing
CREATED tools/apps/system.py - open and find installed applications
CREATED tools/apps/discord_bot.py - Discord bot with per-user conversation history
EDIT tray/app.py - Discord bot starts on startup with shared model
CREATED tools/apps/discord_bot.py - added clear and improved logging
EDIT tools/app/discord_bot.py - added deep search option for full channel history


10.05.2026
EDIT tools/app/discord_bot.py - added hourly filter
EDIT tools/app/discord_bot.py - added mention only toggle
EDIT config.py - added current time to the prompt
CREATED tools/browser.py - web search via SearXNG and DuckDuckGo

13.05.2026
CREATED finetune_dpo.py - DPO training pipeline (adapted from MEng dissertation project)
CREATED prepare_training_data.py - generate synthetic fine-tuning pairs 
EDIT finetune_dpo.py - fixed to work with prepare_training.py data

14.05.2025
TRAINED models/Marvin_v4 - via DPO fine-tune 
EDIT config.py - updated SYSTEM_PROMPT with personality instructions
  personality emerging, system prompt + DPO working together
  TODO: real conversation logs needed for full personality development
  TODO: self model tool, SFT pipeline for tool syntax

KNOWN ISSUES (resolved):
- the model adds "How can I help" and a smiling face emoji in every message

KNOWN ISSUES:
- Tool calling format inconsistent (XML vs JSON) - to finetune
- Model inputs wrong arguments occasionally (e.g. list_reminders with args) - to finetune
- The model will be inconsistent with the tools (e.g. file_path instead of path) - to finetune
- the model runs incorrect commands when searching the RAG (search_knowdge base 90% of the time) - to finetune
- Alt+Space does not toggle window close if already open
- Tray window confirmation prompts (write_file etc) appear in terminal not in GUI
- No way to see tool execution in tray window (confirmation is terminal only)
- schedule reminders is not tracked in list_reminders
- if something is saved during the session, it will not be accessible to the model until restart
- Persistent reminder timer resets on code reset
- format_context truncates large documents at 3000 chars
- doesnt execute discord commands properly without help (once finetuned search_conversation_logs can be added again) - to finetune
- timezone mismatch - check later
- the log does not appear correctly on the app/restore issues

EXTRA TASKS:
- Hotkey definition to be added to config
- move confirmation function to a separate utility script
- make the reminders remain on the screen until snoozed/confirmed
- Reminder escalation — auto open chat if unconfirmed after X minutes with contextual proactive message from Marvin
- Allow the data from RAG to be deleted
- Synthetically generated data to finetune the model with correct commands and their syntax
- on save default to data/information unless the user specifies otherwise - to finetune
- image generation
- time awareness with datetime
- timestamped logs and a timeframe filter
- make discord start config (autostart vs start on demand)
- duckduck links need to be stripped from the duck duck
- let discord bot verion of Marvin use the internet tools (maybe)