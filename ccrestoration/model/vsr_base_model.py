from typing import Any, List

import cv2
import numpy as np
import torch
from torchvision import transforms

from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.model.tile import tile_vsr
from ccrestoration.type import BaseConfig


class VSRBaseModel(SRBaseModel):
    @torch.inference_mode()  # type: ignore
    def inference_image(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("VSR model has no inference_image method")

    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        cfg: BaseConfig = self.config

        if self.tile is None:
            return self.model(img)

        # tile processing
        return tile_vsr(
            model=self.model,
            scale=cfg.scale,
            img=img,
            one_frame_out=self.one_frame_out,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pad_img=self.pad_img,
        )

    @torch.inference_mode()  # type: ignore
    def inference_image_list(self, img_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Inference the image list with the VSR model

        :param img_list: List[np.ndarray]
        :return: List[np.ndarray]
        """
        new_img_list = []
        for img in img_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            new_img_list.append(img)

        # b, n, c, h, w
        img_tensor_stack = torch.stack(new_img_list, dim=1)

        out = self.inference(img_tensor_stack)

        if len(out.shape) == 5:
            res_img_list = []

            for i in range(out.shape[1]):
                img = out[0, i, :, :, :]
                img = img.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).clip(0, 255).astype("uint8")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                res_img_list.append(img)

            return res_img_list

        elif len(out.shape) == 4:
            img = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            return [img]
        else:
            raise ValueError(f"Unexpected output shape: {out.shape}")

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any) -> Any:
        """
        Inference the video with the model, the clip should be a vapoursynth clip

        :param clip: vs.VideoNode
        :return:
        """

        from ccrestoration.vs import inference_vsr, inference_vsr_one_frame_out

        cfg: BaseConfig = self.config

        if self.one_frame_out:
            return inference_vsr_one_frame_out(
                inference=self.inference, clip=clip, scale=cfg.scale, length=cfg.length, device=self.device
            )
        else:
            return inference_vsr(
                inference=self.inference, clip=clip, scale=cfg.scale, length=cfg.length, device=self.device
            )
