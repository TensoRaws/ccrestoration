from typing import Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class EDVRFeatureExtractorConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.EDVRFEATUREEXTACTOR
    model: Union[ModelType, str] = ModelType.EDVRFeatureExtractor
    num_input_frame: int = 5
    num_feat: int = 64


EDVRFeatureExtractorConfigs = [
    # BasicSR SpyNet
    EDVRFeatureExtractorConfig(
        name=ConfigType.EDVRFeatureExtractor_REDS_pretrained_for_IconVSR,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/EDVRFeatureExtractor_REDS_pretrained_for_IconVSR.pth",
        hash="f62a2f1ea25dfa0d38d82b58beef3888edcc5433b8d39ae599bc4e1d1b261e15",
        num_input_frame=5,
        num_feat=64,
    ),
    EDVRFeatureExtractorConfig(
        name=ConfigType.EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR.pth",
        hash="ee48ee9240055f86a1c55cf3fbe8d7802bf017409fda9f9ea2ca09cfb57bcfe7",
        num_input_frame=7,
        num_feat=64,
    ),
]

for cfg in EDVRFeatureExtractorConfigs:
    CONFIG_REGISTRY.register(cfg)
