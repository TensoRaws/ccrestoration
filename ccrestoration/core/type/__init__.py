from enum import Enum

from ccrestoration.core.type.base_config import BaseConfig  # noqa
from ccrestoration.core.type.base_model import BaseModelInterface  # noqa


class ArchType(str, Enum):
    RRDB = "RRDB"
    SRVGG = "SRVGG"


class ModelType(str, Enum):
    RealESRGAN = "RealESRGAN"


class ConfigType(str, Enum):
    # RealESRGAN
    RealESRGAN_realesr_animevideov3_4x = "RealESRGAN_realesr_animevideov3_4x.pth"
    RealESRGAN_AnimeJaNai_HD_V3_Compact_2x = "RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth"
