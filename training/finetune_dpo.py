'''
DPO fine-tuning pipeline for Marvin's conversational style
- Trains on preference pairs
- Teaches style and personality
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import time
import pandas as pd
import matplotlib.pyplot as plt
from config import BASE_MODEL
from training.prepare_DPO_data import prepare_dpo_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

base_model_name = BASE_MODEL
path = Path("./models") 

def get_lora_config():

    ''' 
    0.5B-1.5B: r=8-16, alpha=16-32, 3B-7B: r=16-32, alpha=32-64
    Attention Modules (Most Important) 
    q_proj # Query projection - what we're looking for
    k_proj # Key projection - what we're looking at
    v_proj # Value projection - what information to extract
    o_proj # Output projection - final attention output
    ''' 
    lora_configs = {
    "Qwen": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], # Attention layers (only add adapters to these layers)
        "r": 16, # rank (size of the adapter)
        "lora_alpha": 32, # learning rate scaling
    },
    }

    # Get model type from the full name
    config = lora_configs["Qwen"]
    
    return LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

# Actual fine-tunning of the model

def check_model(model_path):
    """
    Check if a trained model already exists to avoid retraining
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print("Model not found, proceeding to training")
        return False
    
    print(f"Model exists at {model_path} - delete folder to retrain")
    return True

def train_dpo(train_data, model_name): 
    '''
    Train DPO on conversation style examples

    train_data: Dataset with prompt/chosen/rejected column
    model_name: name for the output adapter folder under ./models/
    returns: path to the saved adapter
    '''
    # If the model already exists return it's path
    model_path = path/model_name

    if check_model(model_path):
        return model_path

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # load model with 4-bit quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # bfloat16 for training stability
        bnb_4bit_use_double_quant=True, # double quantization saves extra VRAM
        bnb_4bit_quant_type="nf4" # NormalFloat4 - designed for neural network weights
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # Apply LoRA adapters
    lora_config = get_lora_config()
    print(f"Target modules: {lora_config.target_modules}")
    
    model = prepare_model_for_kbit_training(model) # prepare for 4-bit training
    model = get_peft_model(model, lora_config) # add LoRA adapters
    
    print("\n Trainable parameters:")
    model.print_trainable_parameters()

    train = train_data if train_data is not None else prepare_dpo_dataset()
    print(f" Loaded {len(train)} examples")

    # Training arguments

    training_args = DPOConfig(
        output_dir=str(model_path), # weights and checkpoints save points
        num_train_epochs=5, # loop n times
        per_device_train_batch_size=2, # training examples processed at once
        gradient_accumulation_steps=8, # waits n steps until updating weight
        learning_rate=5e-6, # step size for an error
        logging_steps=20, # print current loss log
        save_steps=200, # checkpoint to go back to in case of a crash
        save_total_limit=3, # keep only 3 most recent checkpoints
        bf16=True, # use bfloat16 for 4-bit training stability 
        remove_unused_columns=False, 
        report_to="none", # don't log into external sites
        beta=0.1, # how much the model is allowed to deviate
        max_length=1024, # cut off too long text
        max_prompt_length=512,
        warmup_steps=10, # LR starts at 0 and slowly climbs up to prevement immediate crashing
    )
    
    # Initialize DPO Trainer 
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        processing_class=tokenizer,
    )
    
    # train the model
    trainer.train() 

    # generate and save loss graph
    print("Generating loss graph...")
    history = trainer.state.log_history
    steps = [e["step"] for e in history if "loss" in e]
    losses = [e["loss"] for e in history if "loss" in e]

    if steps:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label="DPO Training Loss", color="#2c3e50", linewidth=2)
        plt.title(f"Training Loss: {model_name}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Save it to the results folder
        graph_path = Path("./results") / f"{model_name}_loss.png"
        Path("./results").mkdir(parents=True, exist_ok=True)
        plt.savefig(graph_path)
        print(f"Loss graph saved to: {graph_path}")
        plt.close() # Close plot to free memory

    # Save the model andthe adapter
    print(f"Model saved to: {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model_path

if __name__ == "__main__":
    '''
    Will call prepare_DPO_data.py to get the dataset and use it to finetune the original model.
    The new model will be saved as a new file - must be changed in config to use
    '''
    start = time.time()
    print(f"DPO fine-tunning using: {base_model_name} model")
    train_data = prepare_dpo_dataset()
    model_path = train_dpo(train_data, "marvin_v5") 

    end = time.time()
    training_time = (end - start) / 60

    print(f" Training time: {training_time} minutes")