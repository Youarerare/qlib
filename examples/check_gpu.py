import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.0f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.0f} MB")
