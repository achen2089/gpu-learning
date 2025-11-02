import torch

# Check how many GPUs you have
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Check current default GPU
print(f"Current device: {torch.cuda.current_device()}")  # Usually 0

# Get device name
print(f"Device 0: {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    print(f"Device 1: {torch.cuda.get_device_name(1)}")