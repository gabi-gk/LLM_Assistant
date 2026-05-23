from transformers import AutoTokenizer
from training.prepare_SFT_data import prepare_sft_dataset
from training.finetune_sft import preprocess_dataset

tokenizer = AutoTokenizer.from_pretrained("./models/marvin_v5")
dataset = prepare_sft_dataset()
processed = preprocess_dataset(dataset, tokenizer)

print(repr(processed[0]["text"]))