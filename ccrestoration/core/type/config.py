from enum import Enum


class ConfigType(str, Enum):
    # RealESRGAN
    RealESRGAN_RealESRGAN_x4plus_4x = "RealESRGAN_RealESRGAN_x4plus_4x.pth"
    RealESRGAN_RealESRGAN_x4plus_anime_6B_4x = "RealESRGAN_RealESRGAN_x4plus_anime_6B_4x.pth"
    RealESRGAN_RealESRGAN_x2plus_2x = "RealESRGAN_RealESRGAN_x2plus_2x.pth"
    RealESRGAN_realesr_animevideov3_4x = "RealESRGAN_realesr_animevideov3_4x.pth"

    RealESRGAN_AnimeJaNai_HD_V3_Compact_2x = "RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth"
    RealESRGAN_AniScale_2_Compact_2x = "RealESRGAN_AniScale_2_Compact_2x.pth"
    RealESRGAN_Ani4Kv2_Compact_2x = "RealESRGAN_Ani4Kv2_Compact_2x.pth"
