from typing import Any

from ccrestoration.arch import IconVSR
from ccrestoration.config import CONFIG_REGISTRY, IconVSRConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.edvr_model import EDVRFeatureExtractorModel
from ccrestoration.model.spynet_model import SpyNetModel
from ccrestoration.model.vsr_base_model import VSRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.IconVSR)
class IconVSRModel(VSRBaseModel):
    def load_model(self) -> Any:
        cfg: IconVSRConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        spynet = SpyNetModel(
            config=CONFIG_REGISTRY.get(cfg.spynet),
            device=self.device,
            fp16=self.fp16,
            compile=self.compile,
            compile_backend=self.compile_backend,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pad_img=self.pad_img,
        )

        edvr_feature_extractor = EDVRFeatureExtractorModel(
            config=CONFIG_REGISTRY.get(cfg.edvr_feature_extractor),
            device=self.device,
            fp16=self.fp16,
            compile=self.compile,
            compile_backend=self.compile_backend,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pad_img=self.pad_img,
        )

        model = IconVSR(
            num_feat=cfg.num_feat,
            num_block=cfg.num_block,
            keyframe_stride=cfg.keyframe_stride,
            temporal_padding=cfg.temporal_padding,
            spynet=spynet.model,
            edvr_feature_extractor=edvr_feature_extractor.model,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
