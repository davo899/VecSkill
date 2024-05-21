import torch

free, total = torch.cuda.mem_get_info()
print("GPU Memory:")
print(f"  Free  - {round(free / 1_000_000_000.0, 1)}GB")
print(f"  Total - {round(total / 1_000_000_000.0, 1)}GB")
