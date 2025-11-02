import torch

# Check how many GPUs you have
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Check current default GPU
print(f"Current device: {torch.cuda.current_device()}")  # Usually 0

# Get device name
print(f"Device 0: {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    for i in range(1, torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        