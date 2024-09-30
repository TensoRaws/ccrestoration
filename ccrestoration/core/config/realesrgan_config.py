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

    @field_validator("act_type")
    def passwords_match(cls, v: str) -> str:
        if v not in ["relu", "prelu", "leakyrelu"]:
            raise ValueError("act_type must be one of 'relu', 'prelu', 'leakyrelu'")
        return v


for cfg in [
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_RealESRGAN_x4plus_anime_6B_2x,
        url="https://github.com/HolyWu/vs-realesrgan/releases/download/model/RealESRGAN_x4plus_anime_6B.pth",
        arch=ArchType.RRDB,
        model=ModelType.RealESRGAN,
        scale=4,
        num_block=6,
    ),
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_realesr_animevideov3_4x,
        url="https://github.com/HolyWu/vs-realesrgan/releases/download/model/realesr_animevideov3.pth",
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
]:
    CONFIG_REGISTRY.register(cfg)
