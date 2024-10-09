from typing import Any

from ccrestoration.arch import SpyNet
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.auxiliary_base_model import AuxiliaryBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.SpyNet)
class SpyNetModel(AuxiliaryBaseModel):
    def load_model(self) -> Any:
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = SpyNet()

        model.load_state_dict(state_dict, strict=False)
        model.eval().to(self.device)
        return model
