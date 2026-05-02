Base chat — Qwen3-8B, 4-bit, simple chat loop
RAG — ChromaDB, index a folder of your notes - long term external memory
Basic tools — file read/write, notifications, shell commands (low risk)
Agent loop — tool calling parser, confirmation system
Tray app — always-on, hotkey, hide-on-close
Window management tools — switch tabs, get open windows
Conversation logger + fine-tune queue
LoRA/SFT pipeline — adapt existing DPO setup
Adapter versioning
Linux port — OS-aware tool layer, Arch-specific tools

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

KNOWN ISSUES:
- persistent_reminder failing silently — error not surfaced to model  
- Model verbose on tool calls — to be addressed via fine-tuning with conversation logs
- Model sometimes calls wrong tool name before self-correcting (schedule_reminder vs scheduled_reminder)
- Model loops on list_reminders instead of acting on reminder creation
- Model does not always execture the command it says it will be executing

02.05.2026
EDIT agent/registry.py - lazy tool loading, base prompt now small index only
EDIT agent/registry.py - full tool details split into FILE/SHELL/NOTIFICATION groups
EDIT agent/loop.py - tool_help handled directly in agent loop