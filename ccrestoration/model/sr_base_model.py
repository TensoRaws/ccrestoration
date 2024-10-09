from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms

from ccrestoration.cache_models import load_file_from_url
from ccrestoration.model.tile import tile_sr
from ccrestoration.type import BaseConfig, BaseModelInterface


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
        cfg: BaseConfig = self.config

        if self.tile is None:
            return self.model(img)

        # tile processing
        return tile_sr(
            model=self.model,
            scale=cfg.scale,
            img=img,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pad_img=self.pad_img,
        )

    @torch.inference_mode()  # type: ignore
    def inference_image(self, img: np.ndarray) -> np.ndarray:
        """
        Inference the image(BGR) with the model

        :param img: The input image(BGR), can use cv2 to read the image
        :return:
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        if self.fp16:
            img = img.half()

        img = self.inference(img)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any) -> Any:
        """
        Inference the video with the model, the clip should be a vapoursynth clip

        :param clip: vs.VideoNode
        :return:
        """
        cfg: BaseConfig = self.config

        from ccrestoration.vs import inference_sr

        return inference_sr(inference=self.inference, clip=clip, scale=cfg.scale, device=self.device)
