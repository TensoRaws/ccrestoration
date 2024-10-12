from typing import Any

from ccrestoration.arch import SCUNet
from ccrestoration.config import SCUNetConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.SCUNet)
class SCUNetModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: SCUNetConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = SCUNet(
            in_nc=cfg.in_nc,
            config=cfg.config,
            dim=cfg.dim,
            drop_path_rate=cfg.drop_path_rate,
            input_resolution=cfg.input_resolution,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
