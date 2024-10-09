from typing import Any

from ccrestoration.arch import EDVRFeatureExtractor
from ccrestoration.config import EDVRFeatureExtractorConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.auxiliary_base_model import AuxiliaryBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.EDVRFeatureExtractor)
class EDVRFeatureExtractorModel(AuxiliaryBaseModel):
    def load_model(self) -> Any:
        state_dict = self.get_state_dict()

        cfg: EDVRFeatureExtractorConfig = self.config

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = EDVRFeatureExtractor(num_input_frame=cfg.num_input_frame, num_feat=cfg.num_feat)

        model.load_state_dict(state_dict)

        model.eval().to(self.device)
        return model
