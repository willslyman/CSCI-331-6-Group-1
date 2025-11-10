import torch

# Check if PyTorch detects a GPU
print("PyTorch CUDA available:", torch.cuda.is_available())

# Check which device Ultralytics is using
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Ultralytics using device:", device)

# Detailed GPU info if available
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
else:
    print("No GPU detected. Using CPU.")
