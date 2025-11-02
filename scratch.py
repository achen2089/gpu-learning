import torch

x = torch.randn(10000, 10000, device="cuda")

with torch.profiler.profile() as prof:
  y = x @ x

print(prof)

