'''
Generates DPO training pairs for Marvin conversation style fine-tuning
- Targets response tone and personality
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import pandas as pd
from datasets import Dataset
from datetime import datetime
from training.prepare_training_data import generate_synthetic_pairs

OUTPUT_DIR = Path("./data/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_dpo_dataset():
    """
    Prepare the full DPO training dataset from synthetic examples.
    
    returns: Dataset object compatible with the existing DPO pipeline
    """
    pairs = generate_synthetic_pairs()

    if not pairs:
        return None

    random.shuffle(pairs) # shuffle the training pairs

    training_data = [
        {
            "prompt": [{"role": "user",  "content": p["prompt"]}],
            "chosen": [{"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "assistant", "content": p["rejected"]}]
        }
        for p in pairs
    ]

    flat_data = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # timestamp the file name
    csv_path = OUTPUT_DIR / f"marvin_dpo_{timestamp}.csv"
    pd.DataFrame(flat_data).to_csv(csv_path, index=False) # save the csv
    print(f"\n[PREP] Saved to {csv_path}")
    print(f"[PREP] Total DPO pairs: {len(training_data)}")

    return Dataset.from_list(training_data)


if __name__ == "__main__":
    '''
    If run directly from here, prepares the dataset only - useful for inspection
    '''
    dataset = prepare_dpo_dataset()

    if dataset:
        print(f"\n[DPO] Dataset ready: {len(dataset)} examples")
        print(f"\n[DPO] Sample:")
        sample = dataset[0]
        print(f" Prompt: {sample['prompt'][:80]}")
        print(f" Chosen: {sample['chosen'][:80]}")
        print(f" Rejected: {sample['rejected'][:80]}")
        print(f"Types: {type(sample['prompt'])}, {type(sample['chosen'])}, {type(sample['rejected'])}")