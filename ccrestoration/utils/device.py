import sys

import torch


def default_device() -> torch.device:
    if sys.platform != "darwin":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("mps")
