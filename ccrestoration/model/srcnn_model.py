from typing import Any

from ccrestoration.arch import SRCNN
from ccrestoration.config import SRCNNConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.SRCNN)
class SRCNNModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: SRCNNConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = SRCNN(
            num_channels=cfg.num_channels,
            scale=cfg.scale,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
