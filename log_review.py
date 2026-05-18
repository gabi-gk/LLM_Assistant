# review_logs.py
import json
from pathlib import Path

LOGS_DIR = Path("data/logs")
TRAINING_DIR = Path("data/training/training_log")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

def preview_log(log_path):
    """
    Print a readable preview of a conversation log
    
    log_path: the location on the drive of the logs
    """
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)
    
    conv = data.get("conversation", [])
    if not conv:
        return False
    
    print(f"File:     {log_path.name}")
    print(f"Messages: {len(conv)}")
    
    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content", "")
            
        if role == "user":
            prefix = "You:    "
        elif role == "assistant":
            prefix = "Marvin: "
        else:
            continue
        
        # wrap long messages
        if len(content) > 200:
            content = content[:200] + "..."
        
        print(f"\n{prefix}{content}")
    
    return True


def sort_log():
    '''
    Allow the user to move the logs to training
    '''
    logs = sorted(
        [f for f in LOGS_DIR.glob("*.json") 
         if "current" not in f.name and "discord" not in f.name],
        reverse=True  # newest first
    )
    
    if not logs:
        print("No logs found.")
        return
    
    print(f"Found {len(logs)} logs to review.")
    print("Commands: y = keep for training, n = skip, d = delete, q = quit\n")
    
    kept = 0
    skipped = 0
    
    for log in logs:
        if not preview_log(log):
            continue
        
        choice = input("Keep? (y/n/d/q): ").strip().lower()
        
        if choice == 'y':
            dest = TRAINING_DIR / log.name
            log.rename(dest)
            print(f"Moved to training")
            kept += 1
        elif choice == 'd':
            log.unlink()
            print(f"Deleted")
        elif choice == 'q':
            break
        else:
            print(f"Skipped")
            skipped += 1
    
    print(f"\nDone. Kept: {kept}, Skipped: {skipped}")
    print(f"Training logs in: {TRAINING_DIR}")


if __name__ == "__main__":
    sort_log()