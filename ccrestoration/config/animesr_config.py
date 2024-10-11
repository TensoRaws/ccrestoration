from typing import Tuple, Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class AnimeSRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.MSRSWVSR
    model: Union[ModelType, str] = ModelType.AnimeSR
    scale: int = 4
    length: int = 3
    num_feat: int = 64
    num_block: Tuple[int, int, int] = (5, 3, 2)


AnimeSRConfigs = [
    AnimeSRConfig(
        name=ConfigType.AnimeSR_v1_PaperModel_4x,
        path="/Users/tohru/Downloads/AnimeSR_v1_PaperModel_4x.pth",
        scale=4,
    ),
    AnimeSRConfig(
        name=ConfigType.AnimeSR_v2_4x,
        path="/Users/tohru/Downloads/AnimeSR_v2_4x.pth",
        scale=4,
    ),
]

for cfg in AnimeSRConfigs:
    CONFIG_REGISTRY.register(cfg)
