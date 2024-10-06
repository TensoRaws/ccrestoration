from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms

from ccrestoration.arch import SwinIR
from ccrestoration.config import SwinIRConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_model import SRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.SwinIR)
class SwimIRModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: SwinIRConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = SwinIR(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=cfg.norm_layer,
            ape=cfg.ape,
            patch_norm=cfg.patch_norm,
            use_checkpoint=cfg.use_checkpoint,
            upscale=cfg.scale,
            img_range=cfg.img_range,
            upsampler=cfg.upsampler,
            resi_connection=cfg.resi_connection,
        )

        model.load_state_dict(state_dict, assign=True)
        model.eval().to(self.device)
        return model

    @torch.inference_mode()  # type: ignore
    def inference_image(self, img: np.ndarray) -> np.ndarray:
        """
        Inference the image(BGR) with the model

        :param img: The input image(BGR), can use cv2 to read the image
        :return:
        """
        cfg: SwinIRConfig = self.config

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        if self.fp16:
            img = img.half()

        _, _, h_old, w_old = img.size()
        window_size = cfg.window_size
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w_old + w_pad]

        print(img.size())
        img = self.inference(img)
        print(img.size())

        img = img[..., : h_old * cfg.scale, : w_old * cfg.scale]

        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
