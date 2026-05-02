import torch

'''
Test for GPU avaliability before running heavy code
'''

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

free, total = torch.cuda.mem_get_info()
print(f"VRAM free: {free / 1e9:.2f} GB")
print(f"VRAM total: {total / 1e9:.2f} GB")