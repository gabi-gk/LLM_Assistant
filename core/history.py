'''
A script for managing conversation history
- saves and loads conversations
- compacts history when it gets too long by summarising the older messages with the model itself
- keeps track of the sessions state
'''

import json
import os
import torch
from datetime import datetime
from pathlib import Path
from config import LOGS_DIR, SESSION_FILE, RESTORE_LAST_SESSION, MAX_RESTORE_MESSAGES

def save_session_state(state):
    """
    Save session state to disk so the app can decide whether to restore the last session
    
    state: closed, cleared - closed means shutdown without clearning the conversation - restore required
    """
    os.makedirs("./data", exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump({"state": state}, f)

def load_session_state():
    """
    Read the last session state, default to closed if file doesn't exist
    """
    try:
        with open(SESSION_FILE) as f:
            return json.load(f).get("state", "closed")
    except Exception:
        return "closed"
    
def save_conversation(conversation_history, summary=None):
    """
    Save the conversation to a json file on quit - can be used by RAG, future training

    conversation_history: list of message dicts with "role" and "content" keys
    summary: optional string summary of the conversation if it was compacted during the session
    """
    if not conversation_history:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(LOGS_DIR, exist_ok=True)

    log = {
        "timestamp": timestamp,
        "conversation": conversation_history,
        "summary": summary # optional, if the conversation was compacted
    }

    path = f"{LOGS_DIR}/{timestamp}.json"
    with open(path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[SESSION] Conversation saved to {path}")    

def load_last_session():
    """
    Load the most recent conversation log back into memory on startup if not cleared before power down
    Gives the model immediate context from the last session without needing to search RAG
    
    Returns empty list if no logs exist or RESTORE_LAST_SESSION is False
    or the conversation log if present and RESTORE_LAST_SESSION is True
    """
    if not RESTORE_LAST_SESSION:
        return []

    state = load_session_state()
    if state == "cleared":
        print("[SESSION] Last session was cleared, starting fresh.")
        return []

    logs_path = Path(LOGS_DIR)
    if not logs_path.exists():
        return []

    # find the most recent log by filename — they're timestamped
    logs = sorted(logs_path.glob("*.json"), reverse=True)
    if not logs:
        return []

    try:
        with open(logs[0], encoding="utf-8") as f:
            data = json.load(f)

        conversation = data.get("conversation", [])
        if not conversation:
            return []

        # only restore the last N messages (Config)
        restored = conversation[-MAX_RESTORE_MESSAGES:]

        print(f"[SESSION] Restored {len(restored)} messages from {logs[0].name}")
        return restored

    except Exception as e:
        print(f"[SESSION] Could not restore last session: {e}")
        return []

def summarise_history(model, tokenizer, history_chunk):
    """ 
    Summarize the conversation to not lose any important details during a medium length converstaion

    model: model to use for summarization (same as the main model, just with a different prompt)
    tokenizer: tokenizer for the model
    history_chunk: list of message dicts to summarise, the older part of the conversation when compacting history
    
    returns a concise summary string of the conversation chunk
    """
    
    # format the chunk into readable text
    formatted = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in history_chunk
    ])
    
    # prompt the model to summarize the user message
    summary_prompt = [{
        "role": "user",
        "content": f"Summarise this conversation excerpt concisely, "
                   f"preserving any important facts, decisions, or context:\n\n{formatted}"
    }]
    
    prompt = tokenizer.apply_chat_template( # generate the prompt with the same template as the main conversation
        summary_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device) # full input
    
    with torch.no_grad(): # no need to calculate gradients for summarization
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256, # concise summaries shouldn't need to be long
            temperature=0.3, # low temp for factual not creative answers
            do_sample=True, # for some variation in the summaries if the same chunk comes up multiple times
            top_p=0.95 # allow some creativity but keep it focused on the most likely tokens
        )
    
    output = output_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output, skip_special_tokens=True).strip()

def compact_history(model, tokenizer, history, threshold=16, keep_recent=6):
    """
    When history gets long, summarise the older messages and keep the recent exchanges intact
    
    threshold — how many messages before compaction triggers
    keep_recent — how many recent messages to always preserve 

    returns a new history with the older messages replaced by a summary
    """
    # do nothing if not enough messages yet
    if len(history) < threshold:
        return history 
    
    # split into old chunk and recent messages
    old_chunk = history[:-keep_recent]
    recent = history[-keep_recent:]
    
    print("\n[Compacting conversation history...]\n")
    summary = summarise_history(model, tokenizer, old_chunk)
    
    # replace old chunk with a single summary message so the model treats it as context
    summary_message = {
        "role": "user",
        "content": f"[Earlier conversation summary: {summary}]"
    }
    summary_ack = { # optional so the assistant doesnt confuse the summary with a new user message
        "role": "assistant", 
        "content": "Understood, I have context from our earlier conversation."
    }
    
    return [summary_message, summary_ack] + recent

