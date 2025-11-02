import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cuda()
print(f"Tensor on GPU: {x}")
print(f"Device: {x.device}")

# Perform an operation on the GPU
y = x * 2 + 1
print(f"After operation (x * 2 + 1): {y}")

# Move the tensor back to the CPU
y = y.cpu()
print(f"Tensor on CPU: {y}")
print(f"Device: {y.device}")