from typing import Any

import cv2
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
        import vapoursynth as vs

        from ccrestoration.utils.vs import frame_to_tensor, tensor_to_frame

        if not isinstance(clip, vs.VideoNode):
            raise vs.Error("Only vapoursynth clip is supported")

        if clip.format.id not in [vs.RGBH, vs.RGBS]:
            raise vs.Error("Only vs.RGBH and vs.RGBS formats are supported")

        scale = self.config.scale

        @torch.inference_mode()  # type: ignore
        def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
            img = frame_to_tensor(f[0], self.device).unsqueeze(0)

            output = self.inference(img)

            return tensor_to_frame(output, f[1].copy())

        new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
        return new_clip.std.FrameEval(
            lambda n: new_clip.std.ModifyFrame([clip, new_clip], _inference), clip_src=[clip, new_clip]
        )
