from typing import Any

import numpy as np
import torch
from torchvision import transforms

from ccrestoration.cache_models import load_file_from_url
from ccrestoration.core.type import BaseConfig, BaseModelInterface


class SRBaseModel(BaseModelInterface):
    def get_state_dict(self) -> Any:
        """
        Load the state dict of the model from config

        :return: The state dict of the model
        """
        cfg: BaseConfig = self.config

        if cfg.path is not None:
            model_path = str(cfg.path)
        else:
            try:
                model_path = load_file_from_url(cfg)
            except Exception as e:
                print(f"Error: {e}, try force download the model...")
                model_path = load_file_from_url(cfg, force_download=True)

        return torch.load(model_path, map_location=self.device, weights_only=True)

    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any) -> Any:
        raise NotImplementedError

    @torch.inference_mode()  # type: ignore
    def inference_image(self, img: np.ndarray) -> np.ndarray:
        # 参考real-ESRGAN重写

        if not self.fp16:
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        else:
            img = transforms.ToTensor()(img).unsqueeze(0).half().to(self.device)

        img = self.inference(img)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        return img
