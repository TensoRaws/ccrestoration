from typing import Optional, Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class EDVRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.EDVR
    model: Union[ModelType, str] = ModelType.EDVR
    scale: int = 4
    num_in_ch: int = 3
    num_out_ch: int = 3
    num_feat: int = 64
    num_frame: int = 5
    deformable_groups: int = 8
    num_extract_block: int = 5
    num_reconstruct_block: int = 10
    center_frame_idx: Optional[int] = None
    hr_in: bool = False
    with_predeblur: bool = False
    with_tsa: bool = True


EDVRConfigs = [
    # Official Medium size models
    EDVRConfig(
        name=ConfigType.EDVR_M_SR_REDS_official_4x,
        path="/Users/tohru/Downloads/EDVR_M_x4_SR_REDS_official-32075921.pth",
        scale=4,
    ),
]

for cfg in EDVRConfigs:
    CONFIG_REGISTRY.register(cfg)
