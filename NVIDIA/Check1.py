import torch

# Check if CUDA (NVIDIA GPU support) is available
if torch.cuda.is_available():
    # Use the NVIDIA GPU by default
    device = torch.device('cuda')
    print('Using NVIDIA GPU')
else:
    # Use CPU if CUDA is not available
    device = torch.device('cpu')
    print('CUDA not available, using CPU')

# Run your code on the selected device
# For example:
tensor = torch.randn(3, 3).to(device)
result = tensor.mm(tensor.T)
print(result)
