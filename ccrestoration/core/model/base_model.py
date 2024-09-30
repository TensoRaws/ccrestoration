from typing import Any, Optional

import torch

from ccrestoration.core.config import BaseConfig
from ccrestoration.utils.device import default_device


class BaseModelInterface:
    def __init__(self, config: BaseConfig, device: Optional[torch.device] = None):
        self.config = config
        if device is None:
            device = default_device()
        self.device: torch.device = device
        self.module: torch.nn.Module = self.load_model()

    def load_model(self) -> torch.nn.Module:
        raise NotImplementedError

    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        return self.module(img)

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any) -> Any:
        raise NotImplementedError
