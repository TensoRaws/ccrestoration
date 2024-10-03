from enum import Enum


class ConfigType(str, Enum):
    # RealESRGAN
    RealESRGAN_RealESRGAN_x4plus_4x = "RealESRGAN_RealESRGAN_x4plus_4x.pth"
    RealESRGAN_RealESRGAN_x4plus_anime_6B_4x = "RealESRGAN_RealESRGAN_x4plus_anime_6B_4x.pth"
    RealESRGAN_RealESRGAN_x2plus = "RealESRGAN_RealESRGAN_x2plus.pth"
    RealESRGAN_realesr_animevideov3_4x = "RealESRGAN_realesr_animevideov3_4x.pth"

    RealESRGAN_AnimeJaNai_HD_V3_Compact_2x = "RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth"
