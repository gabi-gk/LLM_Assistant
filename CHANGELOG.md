Base chat — Qwen3-8B, 4-bit, simple chat loop
RAG — ChromaDB, index a folder of your notes - long term external memory
Basic tools — file read/write, notifications, shell commands (low risk)
Agent loop — tool calling parser, confirmation system
Tray app — always-on, hotkey, hide-on-close
Window management tools — switch tabs, get open windows
Discord bot API
Conversation logger + fine-tune queue
LoRA/SFT pipeline — adapt existing DPO setup
Let Marvin access and modify his own file describing himself for personality direction
Adapter versioning
Linux port — OS-aware tool layer, Arch-specific tools
browser extension to control active browser tabs
Actual screen tracker
Separate self model into identity file (Marvin) and people files (per person)
Different Qwen models for different devices based on hardware with access to his file - still himself
Merge all the adapters for one clear Marvin 

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

14.05.2026
TRAINED models/Marvin_v4 - via DPO fine-tune 
EDIT config.py - updated SYSTEM_PROMPT with personality instructions
  personality emerging, system prompt + DPO working together
  TODO: real conversation logs needed for full personality development
  TODO: self model tool, SFT pipeline for tool syntax
CREATED data/information/marvin_self_model.md - Marvin's identity and self knowledge file
EDIT tools/knowledge.py - added update_self_model tool
EDIT tray/window.py - GUI confirmation dialog replacing terminal input

KNOWN ISSUES (resolved):
- the model adds "How can I help" and a smiling face emoji in every message
- Tray window confirmation prompts (write_file etc) appear in terminal not in GUI

18.05.2026
EDIT history.py - made logs save full history not the summary
EDIT tray/app.py - compaction only touches conversation_history, logs always full
CREATED log_review.py - to quickly sorft through logs and pick up the training data

20.05.2026
CREATED training/prepare_SFT_data.py - generated SFT training data
CREATED training/prepare_DPO_data.py - separated prepare_training_data.py to be usable by both methods
CREATED finetune_sft.py - copied the DPO framework and adjusted it for SFT
MOVED finetune_dpo.py to training/
CREATED log_review.py - simple log clear up script
EDIT tray/app.py - self model injected directly into context on startup (no agent loop)
EDIT config.py - identity moved from SYSTEM_PROMPT to self model file

21.05.2026
EDIT knowledge.py - added ability for Marvin to edit his self model
CREATE core/utils.py - added for functions used by multiple scripts
EDIT core/utils.py - add confirmation function
EDIT core/utils.py - added correct time and timezone information and injected into the prompt
EDIT config.py - Hotkey definition added to config

KNOWN ISSUES (resolved):
- Alt+Space does not toggle window close if already open
- schedule reminders is not tracked in list_reminders
- close_window confirmation not triggering in tray
- persistant reminder time tracker after reset

22.05.2026
EDIT notifications.py - fixed schedule reminder tracking
EDIT notifications.py - fixed reminders trigger on reset

24.05.2026
EDIT tray/app.py - added discord bot toggle
EDIT tools/notifications.py - added confirmation, snooze and escalation
EDIT tray/app.py - connect the notifications to the tray app

27.05.2026
EDIT loop.py - increase the turn limit

28.05.2026
EDIT app.py - added self injection also on session restore
EDIT files.py - added reindex on write/append to file
EDIT knowledge.py - added callable reindex_knowledge_base and delete_from_knowledge_base
EDIT files.py - added set save path
EDIT utils.py - added self model injection function

KNOWN ISSUES (resolved):
- Session restore message on top even on clear chat
- Data saved during the session is accissible to marvin following a reindex
- Scheduled reminder cannot be edited

KNOWN ISSUES:
- doesnt execute discord commands properly without help (once finetuned search_conversation_logs can be added again) - to finetune
- Marvin sometimes says he completed a task before actually doing it (fabricated confirmation)
- Marvin schedules reminders instead of answering quesions often
- Marvin insists on adding a second parameter when trying to update his self model

EXTRA TASKS:
- image generation
- duckduck links need to be stripped from the duck duck
- save_page tool — fetch full page without char cap for archiving articles
- merge DPO and SFT models into a single one
- Real conversation log DPO 
- home server user detection based on authentication