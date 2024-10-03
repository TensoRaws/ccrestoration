from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms

from ccrestoration.core.arch import UpCunet
from ccrestoration.core.config import RealCUGANConfig
from ccrestoration.core.model import MODEL_REGISTRY
from ccrestoration.core.model.sr_model import SRBaseModel
from ccrestoration.core.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.RealCUGAN)
class RealCUGANModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: RealCUGANConfig = self.config
        state_dict = self.get_state_dict()

        if cfg.pro:
            del state_dict["pro"]

        new_state_dict = {}
        for key, value in state_dict.items():
            # 修改键，添加"unet."前缀
            new_key = "unet." + key
            new_state_dict[new_key] = value

        model = UpCunet(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            scale=cfg.scale,
            cache_mode=cfg.cache_mode,
            alpha=cfg.alpha,
            pro=cfg.pro,
        )

        model.load_state_dict(new_state_dict, assign=True)
        model.eval().to(self.device)
        return model

    @torch.inference_mode()  # type: ignore
    def inference_image(self, img: np.ndarray) -> np.ndarray:
        """
        Inference the image(BGR) with the model

        :param img: The input image(BGR), can use cv2 to read the image
        :return:
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)

        cfg: RealCUGANConfig = self.config
        if cfg.pro:
            img = img * 0.7 + 0.15

        if self.fp16:
            img = img.half()

        img = self.inference(img)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @torch.inference_mode()  # type: ignore
    def inference_video(self, clip: Any) -> Any:
        """
        Inference the video with the model, the clip should be a vapoursynth clip

        :param clip: vs.VideoNode
        :return:
        """

        import vapoursynth as vs

        from ccrestoration.vs import inference_sr

        cfg: RealCUGANConfig = self.config

        def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
            t = torch.stack(
                [torch.from_numpy(np.asarray(frame[plane])).to(device) for plane in range(frame.format.num_planes)]
            ).clamp(0.0, 1.0)
            if cfg.pro:
                t = t * 0.7 + 0.15

            return t

        def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
            array = tensor.squeeze(0).detach().cpu().numpy()
            for plane in range(frame.format.num_planes):
                np.copyto(np.asarray(frame[plane]), array[plane])
            return frame

        return inference_sr(
            inference=self.inference,
            clip=clip,
            scale=self.config.scale,
            device=self.device,
            _frame_to_tensor=frame_to_tensor,
            _tensor_to_frame=tensor_to_frame,
        )
