from enum import Enum


class ConfigType(str, Enum):
    # RealESRGAN
    RealESRGAN_RealESRGAN_x4plus_anime_6B_2x = "RealESRGAN_RealESRGAN_x4plus_anime_6B_2x.pth"
    RealESRGAN_realesr_animevideov3_4x = "RealESRGAN_realesr_animevideov3_4x.pth"
    RealESRGAN_AnimeJaNai_HD_V3_Compact_2x = "RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth"
