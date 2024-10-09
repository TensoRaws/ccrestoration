import random

import numpy as np
import torch


def set_random_seed(seed: int = 0) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
