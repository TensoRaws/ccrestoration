from typing import Any

from ccrestoration.arch import RRDBNet, SRVGGNetCompact
from ccrestoration.config import RealESRGANConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.type import ArchType, ModelType


@MODEL_REGISTRY.register(name=ModelType.RealESRGAN)
class RealESRGANModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: RealESRGANConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        elif "model_state_dict" in state_dict:
            # For APISR's model
            state_dict = state_dict["model_state_dict"]

        if cfg.arch == ArchType.RRDB:
            model = RRDBNet(
                num_in_ch=cfg.num_in_ch,
                num_out_ch=cfg.num_out_ch,
                scale=cfg.scale,
                num_feat=cfg.num_feat,
                num_block=cfg.num_block,
                num_grow_ch=cfg.num_grow_ch,
            )
        elif self.config.arch == ArchType.SRVGG:
            model = SRVGGNetCompact(
                num_in_ch=cfg.num_in_ch,
                num_out_ch=cfg.num_out_ch,
                upscale=cfg.scale,
                num_feat=cfg.num_feat,
                num_conv=cfg.num_conv,
                act_type=cfg.act_type,
            )
        else:
            raise NotImplementedError(f"Arch {cfg.arch} is not implemented.")

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
