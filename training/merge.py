'''
Merge a LoRA adapter into its base model and save full fp16 weights
 
Used twice in the Marvin pipeline:
  FT adapter 
  DPO adapter
'''
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
 
def merge_adapter(base_model_path, adapter_path, output_path):
    '''
    base_model_path: model the adapter was trained on
    adapter_path: folder holding adapter_config.json + adapter_model.safetensors
    output_path: where to write the merged full-weight fp16 model
    returns: output_path
    '''
    base_model_path = str(base_model_path)
    adapter_path = str(adapter_path)
    output_path = Path(output_path)
 
    print(f"Merging adapter:\n  base = {base_model_path}\n  adapter = {adapter_path}")
 
    # Load base in fp16 on CPU 
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
 
    # Attach the adapter, then fold it into the weights
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # returns a plain CausalLM, adapter folded in
 
    # Save full merged weights amd tokenizer 
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(str(output_path))
 
    # Free CPU RAM before the next stage loads anything
    del model
    print(f"Merged model saved to: {output_path}\n")
    return output_path
 
 
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge.py <base_model> <adapter> <output>")
        sys.exit(1)
    merge_adapter(sys.argv[1], sys.argv[2], sys.argv[3])
 
