import torch
import time

# Two large matrices
size = 100000
A_cpu = torch.randn(size, size)
B_cpu = torch.randn(size, size)

# CPU Matrix Multiplication
print(f"Multiplying two {size}x{size} matrices on CPU...")
print(f"\nCPU:")
start = time.time()
C_cpu = A_cpu @ B_cpu
cpu_tim = time.time() - start
print(f" Time: {cpu_time:.4f} seconds")
print(f"Result shape: {C_cpu.shape}")

# GPU Matrix Multiplication
print(f"\nGPU:")
A_gpu = A_cpu.cuda()
B_gpu = B_cpu.cuda()
torch.cuda.synchronize()

start = time.time()
C_gpu = A_gpu @ B_gpu
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f" Time: {gpu_time:.4f} seconds")
print(f"Result shape: {C_gpu.shape}")

print(f"GPU is {cpu_time / gpu_time:.2f}x faster than CPU")
print(f"   (Results match: {torch.allclose(C_cpu, C_gpu.cpu())})")