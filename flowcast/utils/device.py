"""Device auto-detection: CUDA -> Apple MPS -> CPU."""
import torch


def get_device(verbose=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple MPS"
    else:
        device = torch.device("cpu")
        name = "CPU"
    if verbose:
        print(f"[device] using {device} ({name})")
    return device
