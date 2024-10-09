from typing import Any

from ccrestoration.arch import EDSR
from ccrestoration.config import EDSRConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.EDSR)
class EDSRModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: EDSRConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = EDSR(
            num_in_ch=cfg.num_in_ch,
            num_out_ch=cfg.num_out_ch,
            num_feat=cfg.num_feat,
            num_block=cfg.num_block,
            upscale=cfg.scale,
            res_scale=cfg.res_scale,
            img_range=cfg.img_range,
            rgb_mean=cfg.rgb_mean,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
