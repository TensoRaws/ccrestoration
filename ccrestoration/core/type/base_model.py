from typing import Any, Optional

import torch

from ccrestoration.utils.device import DEFAULT_DEVICE


class BaseModelInterface:
    def __init__(self, config: Any, device: Optional[torch.device] = None):
        self.config = config
        if device is None:
            device = DEFAULT_DEVICE
        self.device: torch.device = device
        self.model: torch.nn.Module = self.load_model()

    def load_model(self) -> torch.nn.Module:
        raise NotImplementedError

    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any) -> Any:
        raise NotImplementedError
