from enum import StrEnum

from ccrestoration.utils.registry import RegistryConfigInstance


class ConfigType(StrEnum):
    RealESRGAN_AnimeJaNai_HD_V3_Compact_2x = "RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth"


CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from ccrestoration.core.config.base_config import BaseConfig  # noqa
from ccrestoration.core.config.realesrgan_config import RealESRGANConfig  # noqa
