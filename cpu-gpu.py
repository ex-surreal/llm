import torch
import time
import numpy as np

mps = torch.device("mps")

gpu_x = torch.rand(10000, 10000).to(mps)
gpu_y = torch.rand(10000, 10000).to(mps)

cpu_x = torch.rand(10000, 10000)
cpu_y = torch.rand(10000, 10000)

start_time = time.time()
gpu_xy = gpu_x @ gpu_y
took = time.time() - start_time
print(f"GPU took {took:.10f}")


start_time = time.time()
cpu_xy = np.multiply(cpu_x, cpu_y)
took = time.time() - start_time
print(f"CPU with np took {took:.10f}")
