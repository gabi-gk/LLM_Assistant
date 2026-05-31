'''
SFT training data for Marvin tool syntax fine-tuning
- Imports tool generators 
- Formats as SFT conversation examples for TRL SFTTrainer
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import pandas as pd
from datasets import Dataset
from datetime import datetime
from training.prepare_training_data import generate_tool_pairs

OUTPUT_DIR = Path("./data/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_sft_dataset():
    """
    Prepare SFT dataset from tool syntax examples

    Returns Dataset in SFTTrainer messages format
    """
    pairs = generate_tool_pairs()

    if not pairs:
        return None

    random.shuffle(pairs) # shuffle the training pairs

    training_data = [
        {
            "messages": [
                {"role": "user", "content": p["prompt"]},
                {"role": "assistant", "content": p["chosen"]}
            ]
        }
        for p in pairs
    ]

    flat_data = [{"prompt": p["prompt"], "response": p["chosen"], "type": p.get("type", "")} for p in pairs]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # timestamp the file name
    csv_path = OUTPUT_DIR / f"marvin_sft_{timestamp}.csv"
    pd.DataFrame(flat_data).to_csv(csv_path, index=False) # save the csv
    print(f"\n[SFT] Saved to {csv_path}")
    print(f"[SFT] Total SFT examples: {len(training_data)}")

    # count, sort and print the generated pairs to check the dataset
    type_counts = {}
    for p in pairs:
        t = p.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"[SFT] By type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f" {t}: {count}")

    return Dataset.from_list(training_data)


if __name__ == "__main__":#
    '''
    If run directly from here, prepares the dataset only - useful for inspection
    '''
    dataset = prepare_sft_dataset()
    if dataset:
        print(f"\n[SFT] Dataset ready: {len(dataset)} examples")
        sample = dataset[0]
        print(f" Messages: {sample['messages'][0]['content'][:80]}")
        print(f" {sample['messages'][1]['content'][:80]}")