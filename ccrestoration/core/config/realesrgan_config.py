from pydantic import field_validator

from ccrestoration.core.config import CONFIG_REGISTRY
from ccrestoration.core.type import ArchType, BaseConfig, ConfigType, ModelType


class RealESRGANConfig(BaseConfig):
    scale: int = 2
    num_in_ch: int = 3
    num_out_ch: int = 3
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    num_conv: int = 16
    act_type: str = "prelu"

    @field_validator("arch")
    def arch_match(cls, v: str) -> str:
        if v not in [ArchType.RRDB, ArchType.SRVGG]:
            raise ValueError("real esrgan arch must be one of 'RRDB', 'SRVGG'")
        return v

    @field_validator("act_type")
    def act_type_match(cls, v: str) -> str:
        if v not in ["relu", "prelu", "leakyrelu"]:
            raise ValueError("act_type must be one of 'relu', 'prelu', 'leakyrelu'")
        return v


RealESRGANConfigs = [
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_RealESRGAN_x4plus_anime_6B_4x,
        url="https://github.com/HolyWu/vs-realesrgan/releases/download/model/RealESRGAN_x4plus_anime_6B.pth",
        hash="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        arch=ArchType.RRDB,
        model=ModelType.RealESRGAN,
        scale=4,
        num_block=6,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_realesr_animevideov3_4x,
        url="https://github.com/HolyWu/vs-realesrgan/releases/download/model/realesr_animevideov3.pth",
        hash="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        arch=ArchType.SRVGG,
        model=ModelType.RealESRGAN,
        scale=4,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
        url="https://github.com/HolyWu/vs-realesrgan/releases/download/model/AnimeJaNai_HD_V3_Compact_2x.pth",
        hash="af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f",
        arch=ArchType.SRVGG,
        model=ModelType.RealESRGAN,
        scale=2,
    ),
]

for cfg in RealESRGANConfigs:
    CONFIG_REGISTRY.register(cfg)
