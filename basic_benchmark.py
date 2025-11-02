import torch
import time

# Large operation - GPU wins
x_large = torch.randn(10000, 10000)  # 100 million elements

# CPU
start = time.time()
y_cpu = x_large * 2 + 1
print(f"CPU time: {time.time() - start:.4f}s")

# GPU
x_gpu = x_large.cuda()
torch.cuda.synchronize()  # Wait for transfer
start = time.time()
y_gpu = x_gpu * 2 + 1
torch.cuda.synchronize()  # Wait for GPU to finish
print(f"GPU time: {time.time() - start:.4f}s")

# GPU is 10-100x faster for large tensors!