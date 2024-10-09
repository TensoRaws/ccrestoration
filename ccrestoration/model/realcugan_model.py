from typing import Any

from ccrestoration.arch import UpCunet
from ccrestoration.config import RealCUGANConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.type import ModelType


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

        model.load_state_dict(new_state_dict)
        model.eval().to(self.device)
        return model
