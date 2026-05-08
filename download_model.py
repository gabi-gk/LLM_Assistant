from huggingface_hub import snapshot_download

'''
.safetensors vs .pt/.bin store floating point numbers that represent model weights
.pt and .bin use python's pickle format - serialisation format storing arbitrary objects and executables
.safetensors only downloads the objects without executables meaning it loads faster and will not contain any malicious .bin files

'''

snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir="./models/qwen3-8b",
    ignore_patterns=["*.pt", "*.bin"] # skip any .pt or .bin files to avoid accidental executables - safety measure
)

print("Download complete.")