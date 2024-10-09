from typing import Any

import torch
import torch.nn.functional as F

from ccrestoration.arch import EDVR
from ccrestoration.config import EDVRConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.vsr_base_model import VSRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.EDVR)
class EDVRModel(VSRBaseModel):
    def load_model(self) -> Any:
        cfg: EDVRConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = EDVR(
            num_in_ch=cfg.num_in_ch,
            num_out_ch=cfg.num_out_ch,
            num_feat=cfg.num_feat,
            num_frame=cfg.num_frame,
            deformable_groups=cfg.deformable_groups,
            num_extract_block=cfg.num_extract_block,
            num_reconstruct_block=cfg.num_reconstruct_block,
            center_frame_idx=cfg.center_frame_idx,
            hr_in=cfg.hr_in,
            with_predeblur=cfg.with_predeblur,
            with_tsa=cfg.with_tsa,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model

    @torch.inference_mode()  # type: ignore
    def inference(self, img: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = img.shape
        if b != 1:
            raise ValueError("Batch size must be 1 when VSR inference")

        pad_h = 16 - h % 16 if h % 16 != 0 else 0
        pad_w = 16 - w % 16 if w % 16 != 0 else 0

        img = img.squeeze(0)
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
        img = img.unsqueeze(0)

        img = self.model(img)

        img = img[:, :, : h * self.config.scale, : w * self.config.scale]

        return img
