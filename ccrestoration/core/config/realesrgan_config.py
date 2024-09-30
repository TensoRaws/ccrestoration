from ccrestoration.core.arch import ArchType
from ccrestoration.core.config import CONFIG_REGISTRY, ConfigType
from ccrestoration.core.config.base_config import BaseConfig
from ccrestoration.core.model import ModelType


class RealESRGANConfig(BaseConfig):
    scale: int
    denoise_strength: float


for cfg in [
    RealESRGANConfig(
        name=ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
        url="https://github.com/HolyWu/vs-realesrgan/releases/download/model/AnimeJaNai_HD_V3_Compact_2x.pth",
        hash="af7307eee19e5982a8014dd0e4650d3bde2e25aa78d2105a4bdfd947636e4c8f",
        scale=2,
        denoise_strength=0.1,
        arch=ArchType.SRVGG,
        model=ModelType.RealESRGAN,
    ),
    RealESRGANConfig(
        name="1111",
        scale=2,
        denoise_strength=0.1,
        arch=ArchType.SRVGG,
        model=ModelType.RealESRGAN,
    ),
]:
    CONFIG_REGISTRY.register(cfg)
