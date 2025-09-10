# Tensor/device utilities
try:
    # Prefer importing from the existing engine for now
    from options_engine_torch import tensor_dtype as _tensor_dtype
except Exception:
    import torch
    _tensor_dtype = torch.float32

# Public alias
tensor_dtype = _tensor_dtype

