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
        name=ConfigType.RealESRGAN_RealESRGAN_x4plus_4x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealESRGAN_RealESRGAN_x4plus_4x.pth",
        hash="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        arch=ArchType.RRDB,
        model=ModelType.RealESRGAN,
        scale=4,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_RealESRGAN_x4plus_anime_6B_4x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealESRGAN_RealESRGAN_x4plus_anime_6B_4x.pth",
        hash="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        arch=ArchType.RRDB,
        model=ModelType.RealESRGAN,
        scale=4,
        num_block=6,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_RealESRGAN_x2plus_2x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealESRGAN_RealESRGAN_x2plus_2x.pth",
        hash="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        arch=ArchType.RRDB,
        model=ModelType.RealESRGAN,
        scale=2,
        num_block=23,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_realesr_animevideov3_4x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealESRGAN_realesr_animevideov3_4x.pth",
        hash="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        arch=ArchType.SRVGG,
        model=ModelType.RealESRGAN,
        scale=4,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth",
        hash="af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f",
        arch=ArchType.SRVGG,
        model=ModelType.RealESRGAN,
        scale=2,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_AniScale_2_Compact_2x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealESRGAN_AniScale_2_Compact_2x.pth",
        hash="916ddf99eac77008834a8aeb3dc74b64b17eee02932c18bca93cfa093106e85d",
        arch=ArchType.SRVGG,
        model=ModelType.RealESRGAN,
        scale=2,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_Ani4Kv2_Compact_2x,
        url="https://github.com/TensoRaws/ccrestoration/releases/download/model_zoo/RealESRGAN_Ani4Kv2_Compact_2x.pth",
        hash="fe99290e9e4f95424219566dbe159184a123587622cc00bc632b1eecbd07d7a4",
        arch=ArchType.SRVGG,
        model=ModelType.RealESRGAN,
        scale=2,
    ),
]

for cfg in RealESRGANConfigs:
    CONFIG_REGISTRY.register(cfg)
