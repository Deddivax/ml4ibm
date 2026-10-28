import torch
import numpy as np

def _tensor(x, device, dtype=torch.float64):
    """Convert array-like to a torch tensor on the target device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)