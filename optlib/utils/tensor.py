# Device and dtype utilities
import torch
import os

# Detect best available device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Precision policy: use float64 on CPU/CUDA, float32 on MPS
if device.type == 'mps':
    torch.set_default_dtype(torch.float32)
    tensor_dtype = torch.float32
    complex_dtype = torch.complex64
else:
    torch.set_default_dtype(torch.float64)
    tensor_dtype = torch.float64
    complex_dtype = torch.complex128

