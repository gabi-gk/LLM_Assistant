# test_merged.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import BASE_MODEL, ADAPTER_PATH

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()

# test without system prompt 
messages = [{"role": "user", "content": "hi"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)

output = outputs[0][len(inputs.input_ids[0]):]
print(tokenizer.decode(output, skip_special_tokens=True))