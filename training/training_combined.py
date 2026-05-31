'''
Full SFT and DPO training plus merge in one go

Creates a single marvin model at the end
 
On a new tool or training data addition this will delete the model and retrain it
'''
import sys
import time
import shutil
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from config import BASE_MODEL  # stock Qwen3-8B path
from training import finetune_sft as sft_mod
from training import finetune_dpo as dpo_mod
from training.prepare_SFT_data import prepare_sft_dataset
from training.prepare_DPO_data import prepare_dpo_dataset
from training.merge import merge_adapter
 
MODELS = Path("./models")
 
# save training data alongside the final model
SFT_ADAPTER = MODELS / "marvin_sft_adapter" # just sft
SFT_MERGED = MODELS / "marvin_sft_merged" # 
DPO_ADAPTER = MODELS / "marvin_dpo_adapter"# DPO only
FINAL_MODEL = MODELS / "marvin" 
 
# Delete the model and train from the beginning when new training data is added
CLEAN_REBUILD = False
 
 
def clean():
    '''
    On retrain, remove all existing adapters 
    '''
    for p in (SFT_ADAPTER, SFT_MERGED, DPO_ADAPTER, FINAL_MODEL):
        if p.exists():
            print(f"Removing old {p}")
            shutil.rmtree(p)

if __name__ == "__main__":
    '''
    Will call prepare both datasets and both training functions and finally merge them all together
    '''
    start = time.time()
 
    if CLEAN_REBUILD:
        # remove existing trained models
        clean()
 
    # SFT on the original qwen model
    print("\n SFT on Qwen")
    sft_mod.base_model_name = BASE_MODEL
    sft_data = prepare_sft_dataset()
    sft_mod.train_sft(sft_data, SFT_ADAPTER.name)
 
    # Merge SFT with the base
    print("\n Merging SFT adapter into base")
    merge_adapter(BASE_MODEL, SFT_ADAPTER, SFT_MERGED)
 
    # Train DPO on the SFT model
    print("\n Applying DPO adapter on SFT model ===")
    dpo_mod.base_model_name = str(SFT_MERGED)
    dpo_data = prepare_dpo_dataset()
    dpo_mod.train_dpo(dpo_data, DPO_ADAPTER.name)
 
    # Merge the models
    print("\n Merge DPO adapter")
    merge_adapter(SFT_MERGED, DPO_ADAPTER, FINAL_MODEL)
 
    end = time.time()
    training_time = (end - start) / 60
    print(f" Training time: {training_time} minutes")
    print(f"Final model at: {FINAL_MODEL}")
 
