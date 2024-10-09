from typing import List

import cv2
import numpy as np
import torch
from torchvision import transforms

from ccrestoration.model.sr_base_model import SRBaseModel


class VSRBaseModel(SRBaseModel):
    @torch.inference_mode()  # type: ignore
    def inference_image(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("VSR model has no inference_image method")

    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = img.shape
        if b != 1:
            raise ValueError("Batch size must be 1 when VSR inference")

        return self.model(img)

    @torch.inference_mode()  # type: ignore
    def inference_image_list(self, img_list: List[np.ndarray]) -> List[np.ndarray]:
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
                img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
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
