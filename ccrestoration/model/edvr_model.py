from typing import Any

from ccrestoration.arch import EDVR, EDVRFeatureExtractor
from ccrestoration.config import EDVRConfig, EDVRFeatureExtractorConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.auxiliary_base_model import AuxiliaryBaseModel
from ccrestoration.model.vsr_base_model import VSRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.EDVR)
class EDVRModel(VSRBaseModel):
    def load_model(self) -> Any:
        self.one_frame_out = True

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
            num_frame=cfg.length,
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
