Marvin - Local AI Assistant 
    Designed as a persistent personal AI with an evolving identity rather than a stateless assistant

- Personal AI running locally on a private machine
- Built using a Qwen3-8B model with 4-bit quantisation 
- Includes a DPO and SFT setup for personality and tool training 

Design
- Self-model file. - a quantised local model is stateless, it loses all the context between sessions. Marvin keeps a file that lets him persist his identity as well as user relevant context across conversations which gets injected into his prompt on each session to keep high and relevant data on each run
- Identity vs prompt separation - the system prompt stays short and holds only the core, essential instructions, the identity and context stay in the editable self-file. This separates the fluid part of the model from the fixed part, letting the personality evolve while keeping the prompt mostly untouched
- Additional capabilities of the model were inspired by failure and exploration. The tool set is not fully sketched and gets improved and expanded through testing. Marvin is run, testing for any gaps and errors with his abilities which are then solved by new tools and upgrades.
- Split training pipeline - Adapted from the dissertation, initial tests proved that DPO is wrong for tool syntax, where there's a single correct answer. This led to dataset being split into DPO's conversational style and SFT's exact tool syntax correctness letting Marvin learn to use correct parameters while keeping desired response style.
- Reliability work - Marvin used to claim tasks were done when the tool had actually failed (fabricated confirmation). The agent loop now judges the success from the actual tool result as well as the real state, stops after 3 consecutive fail attempts and reports the failure.
- Merged final model - the SFT and DPO generated models were expanded on from being adapters to merging into a full model, making it set to be usable across a household and easy to manage.

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
- The BASE_MODEL is the unedited Qwen model that is used during training and will be a fallback if a trained model doesn't exist
- System prompt is derived from the self-model file which is not included
- discord bot can be disabled and re-enabled via DISCORD_ENABLED in config.py

Setup
- Check for GPU availability using check_gpu.py the model requires at least 8GB VRAM to run 
- Training and merging needs additional 16 GB of RAM
- download the Qwen model from huggingface using download_model.py 
- run run.py for the GUI version of the model
    / or run main.py for the terminal/debug mode
- for finetuning the general SFT setup can be run however DPO requires manual data generation/import

Training
- training can be done using training_combined.py which automatically runs both trainings and combines them into a single final version of the AI
- When new training data is added, the model is rebuilt from the baseline

SERVER
- server.py runs the model as an OpenAI-compatible HTTP service
- it is pure inference only - the self model and tools stay client sided
- Dockerfile builds an image, mounts the weights at /models so the image can use different models
- Requires an NVIDIA GPU and nvidia-container-toolkit to host
- Defaults to localhost (127.0.0.1), can be modified to LAN (0.0.0.0) but there is no authentication so it is not advised
- To build and run:
    docker build -t marvin_server .
    docker run --rm --gpus all -p 8000:8000 -v ./models:/models:ro marvin_server    

Project Structure
    agent/ - the main agent loop and tool definition registry
    core/ - the main functions of the model including RAG
    data/ - generated on startup - will include logs and is where ChromaDB indexes data from
    models/ - generated on model download - path used in config to load the model
    tools/ - additional tools including notifications, file systems, window management and the extra discord bot
    training/ - the training scripts
    tray/ - the GUI application
    run.py - GUI entry point
    main.py - terminal/debug entry point
    config.py - all constants, paths and system prompts
    server.py - entry point for the server
