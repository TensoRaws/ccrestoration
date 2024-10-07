import sys

import torch


def default_device() -> torch.device:
    if sys.platform != "darwin":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        except Exception as e:
            print(f"Err: {e}, MPS is not available, use CPU instead.")
            return torch.device("cpu")


DEFAULT_DEVICE = default_device()
