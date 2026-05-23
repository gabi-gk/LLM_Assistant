'''
SFT fine-tuning pipeline for Marvin tool syntax
- Trains on correct tool call examples
- Teaches strict parameter name compliance: path not file_path, cmd not command
- Adapted from MEng dissertation DPO pipeline
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
import time
import matplotlib.pyplot as plt
from training.prepare_SFT_data import prepare_sft_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

base_model_name = "./models/marvin_v5"
path = Path("./models")


def get_lora_config():
    '''
    14B+: r=32-64, alpha=64-128 
    Attention Modules (Most Important) 
    q_proj # Query projection - what we're looking for
    k_proj # Key projection - what we're looking at
    v_proj # Value projection - what information to extract
    o_proj # Output projection - final attention output
    MLP Layer
    gate_proj # Controls information flow (like a gate)
    up_proj # Expands dimension
    down_proj # Reduces dimension back
    
    ''' 
    lora_configs = {
        "Qwen": {
            # Attention layers and MLP (only add adapters to these layers)
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "r": 64, # rank (size of the adapter)
            "lora_alpha": 128, # learning rate scaling
        },
    }

    # Get model type from the full name
    config = lora_configs["Qwen"]

    return LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=0.05,
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

def train_sft(train_data, model_name):
    '''
    Train SFT adapter on correct tool call examples

    train_data: Dataset with "messages" column
    model_name: name for the output adapter folder under ./models/
    returns: path to the saved adapter
    '''
    # If the model already exists return it's path
    model_path = path / model_name

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

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    train = train_data if train_data is not None else prepare_sft_dataset()
    train = preprocess_dataset(train, tokenizer)
    print(f"Loaded {len(train)} examples")

    # SFT training arguments
    training_args = SFTConfig(
        output_dir=str(model_path), # weights and checkpoints save points
        num_train_epochs=1, # loop n times
        eval_strategy="steps",
        eval_steps=100,
        per_device_train_batch_size=2, # training examples processed at once
        gradient_accumulation_steps=8, # waits n steps until updating weight
        learning_rate=2e-5, # step size for an error
        logging_steps=20, # print current loss log
        save_steps=200, # checkpoint to go back to in case of a cras
        save_total_limit=3, # keep only 3 most recent checkpoints
        bf16=True, # use bfloat16 for 4-bit training stabilit
        remove_unused_columns=False,
        report_to="none", # don't log into external sites
        warmup_steps=10, # LR starts at 0 and slowly climbs up to prevement immediate crashing
        max_length=1024, # cut off too long text
        dataset_kwargs={"skip_prepare_dataset": True}, # dataset is pre-tokenized
    )

    # pads input_ids, attention_mask, and labels (-100 pad keeps loss masking intact)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    # split into train and verifiction 
    split = train.train_test_split(test_size=0.1)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        data_collator=collator,
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
        plt.plot(steps, losses, label="SFT Training Loss", color="#2c3e50", linewidth=2)
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

def preprocess_dataset(dataset, tokenizer, max_length=1024):
    '''
    Tokenize the dataset for SFT, strip the Qwen's think tokens

    dataset: Dataset with "messages" column
    tokenizer: model tokenizer
    max_length: truncate if longer than this
    returns: tokenized dataset with input_ids, attention_mask and labels
    '''
    response_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def tokenize_and_mask(example):
        '''
        Apply chat template, strip think tokens, tokenize and mask user prompt tokens from loss
        '''
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        text = re.sub(r'<think>.*?</think>\n*', '', text, flags=re.DOTALL) # strip Qwen's think tokens

        tokenized = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        input_ids = tokenized["input_ids"]

        # mask everything - then unmask from the first assistant response onwards
        labels = [-100] * len(input_ids)
        n = len(response_ids)
        for i in range(len(input_ids) - n + 1):
            if input_ids[i:i + n] == response_ids:
                for j in range(i + n, len(input_ids)):
                    labels[j] = input_ids[j]
                break

        return {"input_ids": input_ids, "attention_mask": tokenized["attention_mask"], "labels": labels}

    return dataset.map(tokenize_and_mask).remove_columns(["messages"])

if __name__ == "__main__":
    '''
    Will call prepare_SFT_data.py to get the dataset and use it to finetune the original model.
    The new model will be saved as a new file - must be changed in config to use
    '''
    start = time.time()
    print(f"SFT fine-tuning using: {base_model_name} model")

    train_data = prepare_sft_dataset()
    model_path = train_sft(train_data, "marvin_sft_v4")

    end = time.time()
    training_time = (end - start) / 60
    print(f"Training time: {training_time:.1f} minutes")