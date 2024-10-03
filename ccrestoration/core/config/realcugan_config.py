from typing import Union

from pydantic import field_validator

from ccrestoration.core.config import CONFIG_REGISTRY
from ccrestoration.core.type import ArchType, BaseConfig, ConfigType, ModelType


class RealCUGANConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.UPCUNET
    model: Union[ModelType, str] = ModelType.RealCUGAN
    scale: int = 2
    in_channels: int = 3
    out_channels: int = 3
    cache_mode: int = 0
    alpha: float = 1
    pro: bool = False

    @field_validator("scale")
    def scale_match(cls, v: int) -> int:
        if v not in [2, 3, 4]:
            raise ValueError("real cugan scale must be one of 2, 3, 4")
        return v

    @field_validator("cache_mode")
    def cache_mode_match(cls, v: int) -> int:
        if v not in [0, 1, 2, 3]:
            raise ValueError("cache mode must be one of 0, 1, 2, 3")
        return v

    @field_validator("alpha")
    def alpha_match(cls, v: float) -> float:
        if v < 0 or v > 2:
            raise ValueError("alpha must be in [0, 2]")
        return v


RealCUGANConfigs = [
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Pro_Conservative_2x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealCUGAN_Pro_Conservative_2x.pth",
        hash="b8ae5225d2d515aa3c33ef1318aadc532a42ea5ed8d564471b5a5b586783e964",
        scale=2,
        pro=True,
    ),
    RealCUGANConfig(
        name=ConfigType.RealCUGAN_Conservative_2x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealCUGAN_Conservative_2x.pth",
        hash="6cfe3b23687915d08ba96010f25198d9cfe8a683aa4131f1acf7eaa58ee1de93",
        scale=2,
    ),
]

for cfg in RealCUGANConfigs:
    CONFIG_REGISTRY.register(cfg)
