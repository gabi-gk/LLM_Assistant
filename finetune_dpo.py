import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import time
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from config import BASE_MODEL
from prepare_DPO_data import prepare_dpo_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

base_model_name = BASE_MODEL
path = Path("./models") 

def get_lora_config():

    ''' 0.5B-1.5B: r=8-16, alpha=16-32, 3B-7B: r=16-32, alpha=32-64, 14B+: r=32-64, alpha=64-128 
    Attention Modules (Most Important) 
    q_proj  # Query projection - what we're looking for
    k_proj  # Key projection - what we're looking at
    v_proj  # Value projection - what information to extract
    o_proj  # Output projection - final attention output
    MLP Layer
    gate_proj  # Controls information flow (like a gate)
    up_proj    # Expands dimension
    down_proj  # Reduces dimension back
    
    ''' 
    lora_configs = {
    "Qwen": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], # Attention layers (only add adapters to these layers)
        "r": 16, # rank (size of the adapter)
        "lora_alpha": 32, # learning rate scaling
    },
    }

    # Get model type from the full name
    model_type = "Qwen"
    config = lora_configs[model_type]
    
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
    # check if the model has already been generated
    model_path = Path(model_path)
    if not model_path.exists():
        print("Model not found, proceeding to training")
        return False
    
    print(f"Model exists, returning path")
    return True

def train_dpo(train_data, model_name, sample_size = None, bo_data = False, **kwargs): 
    '''
    Takes the name of the model, checks whether a trained model already exists
    if it doesn't trains and saves the model and returns the path
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
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    # Apply LoRA
    lora_config = get_lora_config()
    print(f"Target modules: {lora_config.target_modules}")
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print("\n Trainable parameters:")
    model.print_trainable_parameters()

    if train_data is not None:
        train = train_data
    elif bo_data:
        train_df = pd.read_csv("bo_pairs_train.csv") # BO version
        train = Dataset.from_pandas(train_df)
    else:
        train = prepare_dpo_dataset()
    
    print(f" Loaded {len(train)} examples")

    # Training arguments

    training_args = DPOConfig(
        output_dir=str(model_path), # weights and checkpoints save points
        num_train_epochs=5, # loop n times
        per_device_train_batch_size=2, # training examples processed at once
        gradient_accumulation_steps=8, # waits 8 steps until updating weight
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
    train_result = trainer.train() 

    # save the model
    print("Generating loss graph...")
    history = trainer.state.log_history
    steps = []
    losses = []

    for entry in history:
        if "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    if steps:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label="DPO Training Loss", color="#2c3e50", linewidth=2)
        plt.title(f"Training Loss: {model_name}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Save it to your results folder
        graph_path = Path("./results") / f"{model_name}_loss.png"
        Path("./results").mkdir(parents=True, exist_ok=True)
        plt.savefig(graph_path)
        print(f"Loss graph saved to: {graph_path}")
        plt.close() # Close plot to free memory

    # Save the model and tokenizer
    print(f"Model saved to: {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model_path

if __name__ == "__main__":
    start = time.time()
    print(f"DPO fine-tunning using: {base_model_name} model")
    train_data = prepare_dpo_dataset()
    model_path = train_dpo(train_data, "marvin_v5")

    end = time.time()
    training_time = (end - start) / 60

    print(f" Training time: {training_time} minutes")