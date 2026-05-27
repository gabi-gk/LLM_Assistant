Marvin - Local AI Assistant 
    Designed as a persistent personal AI with an evolving identity rather than a stateless assistant

Personal AI running locally on a private machine
Built using a Qwen3-8B model with 4-bit quantisation 
Includes a DPO and SFT setup for personality and tool training 

Features
- System tray app with configurable hotkey (current Alt+Space)
- Long-term memory via ChromaDB RAG
- File management, commandline execution, browser tools, notification scheduling
- Personal model memory - Persistent identity system - the model can read and write to his own self-model file, allowing personality and knowledge to evolve over time
- Fine-tuning pipelines - DPO for personality and SFT for tool syntax
- Optional discord bot - configurable in the config

Stack
- Model Qwen3-8B, 4-bit nf4, bitsandbytes
- RAG: ChromaDB + all-MiniLM-L6-v2
- GUI: tkinter + pystray
- Training: TRL DPOTrainer + SFTTrainer, PEFT LoRA

Config
- The files do not include any RAG data
- The files do not include actual Qwen model nor its fine-tuned version
- The training scripts do not include personalization data but do have a general setup for SFT tuning generation
- The path settings and model definitions can be edited in the config.py script
- System prompt is derived from the self-model file which is not included
- discord bot can be disabled and re-enabled via DISCORD_ENABLED in config.py

Setup
- Check for GPU availability using check_gpu.py the model requires at least 8GB VRAM to run 
- download the Qwen model from huggingface using download_model.py 
- run run.py for the GUI version of the model
    / or run main.py for the terminal/debug mode
- for finetuning the general SFT setup can be run however DPO requires manual data generation/import

Project Structure
    agent/ - the main agent loop and tool definition registry
    core/ - the main functions of the model including RAG
    data/ - generated on startup - will include logs and is where ChromaDB indexes data from
    models/ - generated on model download - path used in config to load the model
    tools/ - addition tools including notifications, file systems, window management and the extra discord bot
    training/ - the training scripts
    tray/ - the GUI application
    run.py - GUI entry point
    main.py - terminal/debug entry point
    config.py - all constants, paths and system prompts