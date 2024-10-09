from typing import Any

import numpy as np
import torch

from ccrestoration.model.sr_base_model import SRBaseModel


class AuxiliaryBaseModel(SRBaseModel):
    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Auxiliary model should use self.model to load in the main model")

    @torch.inference_mode()  # type: ignore
    def inference_image(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Auxiliary model has no inference_image method")

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any) -> Any:
        raise NotImplementedError("Auxiliary model has no inference_video method")
