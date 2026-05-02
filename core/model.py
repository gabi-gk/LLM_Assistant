from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
from config import BASE_MODEL

def load_model(model_name=BASE_MODEL):
    # load the pre-trained model and its tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)# text to numbers 
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, # compress the weights from 16 pits per number to 4 bits per number
        bnb_4bit_compute_dtype=torch.bfloat16, # keep the maths accurate to the smaller storage
        bnb_4bit_use_double_quant=True, # quantize the quantization constraints
        bnb_4bit_quant_type="nf4" # NormalFloat4 - compression designed for NNs following a normal distribution approximately
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config, 
        device_map="auto", # CPU/GPU
        trust_remote_code=False, # do not pull and run any remote code
    )
    return model, tokenizer

def generate_response(model, tokenizer, conversation_history, system_prompt, streamer):
    # Prepare the model input, include chat history
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}] + conversation_history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False      
    )

    # Convert string to token ID for the model and return as torch tensors
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad(): # no gradients needed for inference
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9, # cut off the unlinkely tokens
            do_sample=True, # tokens picked probabilistically (True) - varied responses vs deterministically (False) - same output every time
            streamer=streamer # print while thinking rather than dump everything at once
        )
    
    # Keep only the newly generated tokens and decode
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

def create_streamer(tokenizer):
    return TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)