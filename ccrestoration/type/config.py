from enum import Enum


# Enum for config type, {ModelType.model}_{config_name}_{scale}x.pth
# For the Auxiliary Network, {ModelType.model}_{config_name}.pth
class ConfigType(str, Enum):
    # ------------------------------------- Single Image Super-Resolution ----------------------------------------------

    # RealESRGAN
    RealESRGAN_RealESRGAN_x4plus_4x = "RealESRGAN_RealESRGAN_x4plus_4x.pth"
    RealESRGAN_RealESRGAN_x4plus_anime_6B_4x = "RealESRGAN_RealESRGAN_x4plus_anime_6B_4x.pth"
    RealESRGAN_RealESRGAN_x2plus_2x = "RealESRGAN_RealESRGAN_x2plus_2x.pth"
    RealESRGAN_realesr_animevideov3_4x = "RealESRGAN_realesr_animevideov3_4x.pth"

    RealESRGAN_AnimeJaNai_HD_V3_Compact_2x = "RealESRGAN_AnimeJaNai_HD_V3_Compact_2x.pth"
    RealESRGAN_AniScale_2_Compact_2x = "RealESRGAN_AniScale_2_Compact_2x.pth"
    RealESRGAN_Ani4Kv2_Compact_2x = "RealESRGAN_Ani4Kv2_Compact_2x.pth"
    RealESRGAN_APISR_RRDB_GAN_generator_2x = "RealESRGAN_APISR_RRDB_GAN_generator_2x.pth"
    RealESRGAN_APISR_RRDB_GAN_generator_4x = "RealESRGAN_APISR_RRDB_GAN_generator_4x.pth"

    # RealCUGAN
    RealCUGAN_Conservative_2x = "RealCUGAN_Conservative_2x.pth"
    RealCUGAN_Denoise1x_2x = "RealCUGAN_Denoise1x_2x.pth"
    RealCUGAN_Denoise2x_2x = "RealCUGAN_Denoise2x_2x.pth"
    RealCUGAN_Denoise3x_2x = "RealCUGAN_Denoise3x_2x.pth"
    RealCUGAN_No_Denoise_2x = "RealCUGAN_No_Denoise_2x.pth"
    RealCUGAN_Conservative_3x = "RealCUGAN_Conservative_3x.pth"
    RealCUGAN_Denoise3x_3x = "RealCUGAN_Denoise3x_3x.pth"
    RealCUGAN_No_Denoise_3x = "RealCUGAN_No_Denoise_3x.pth"
    RealCUGAN_Conservative_4x = "RealCUGAN_Conservative_4x.pth"
    RealCUGAN_Denoise3x_4x = "RealCUGAN_Denoise3x_4x.pth"
    RealCUGAN_No_Denoise_4x = "RealCUGAN_No_Denoise_4x.pth"
    RealCUGAN_Pro_Conservative_2x = "RealCUGAN_Pro_Conservative_2x.pth"
    RealCUGAN_Pro_Denoise3x_2x = "RealCUGAN_Pro_Denoise3x_2x.pth"
    RealCUGAN_Pro_No_Denoise_2x = "RealCUGAN_Pro_No_Denoise_2x.pth"
    RealCUGAN_Pro_Conservative_3x = "RealCUGAN_Pro_Conservative_3x.pth"
    RealCUGAN_Pro_Denoise3x_3x = "RealCUGAN_Pro_Denoise3x_3x.pth"
    RealCUGAN_Pro_No_Denoise_3x = "RealCUGAN_Pro_No_Denoise_3x.pth"

    # EDSR
    EDSR_Mx2_f64b16_DIV2K_official_2x = "EDSR_Mx2_f64b16_DIV2K_official_2x.pth"
    EDSR_Mx3_f64b16_DIV2K_official_3x = "EDSR_Mx3_f64b16_DIV2K_official_3x.pth"
    EDSR_Mx4_f64b16_DIV2K_official_4x = "EDSR_Mx4_f64b16_DIV2K_official_4x.pth"

    # SwinIR
    SwinIR_classicalSR_DF2K_s64w8_SwinIR_M_2x = "SwinIR_classicalSR_DF2K_s64w8_SwinIR_M_2x.pth"
    SwinIR_lightweightSR_DIV2K_s64w8_SwinIR_S_2x = "SwinIR_lightweightSR_DIV2K_s64w8_SwinIR_S_2x.pth"
    SwinIR_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR_L_GAN_4x = "SwinIR_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR_L_GAN_4x.pth"
    SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_2x = "SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_2x.pth"
    SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_4x = "SwinIR_realSR_BSRGAN_DFO_s64w8_SwinIR_M_GAN_4x.pth"

    SwinIR_Bubble_AnimeScale_SwinIR_Small_v1_2x = "SwinIR_Bubble_AnimeScale_SwinIR_Small_v1_2x.pth"

    # SCUNet
    SCUNet_color_50_1x = "SCUNet_color_50_1x.pth"
    SCUNet_color_real_psnr_1x = "SCUNet_color_real_psnr_1x.pth"
    SCUNet_color_real_gan_1x = "SCUNet_color_real_gan_1x.pth"

    # ------------------------------------- Auxiliary Network ----------------------------------------------------------

    # SpyNet
    SpyNet_spynet_sintel_final = "SpyNet_spynet_sintel_final.pth"

    # EDVR Feature Extractor
    EDVRFeatureExtractor_REDS_pretrained_for_IconVSR = "EDVRFeatureExtractor_REDS_pretrained_for_IconVSR.pth"
    EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR = "EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR.pth"

    # ------------------------------------- Video Super-Resolution -----------------------------------------------------

    # EDVR
    EDVR_M_SR_REDS_official_4x = "EDVR_M_SR_REDS_official_4x.pth"
    EDVR_M_woTSA_SR_REDS_official_4x = "EDVR_M_woTSA_SR_REDS_official_4x.pth"

    # BasicVSR
    BasicVSR_REDS_4x = "BasicVSR_REDS_4x.pth"
    BasicVSR_Vimeo90K_BD_4x = "BasicVSR_Vimeo90K_BD_4x.pth"
    BasicVSR_Vimeo90K_BI_4x = "BasicVSR_Vimeo90K_BI_4x.pth"

    # IconVSR
    IconVSR_REDS_4x = "IconVSR_REDS_4x.pth"
    IconVSR_Vimeo90K_BD_4x = "IconVSR_Vimeo90K_BD_4x.pth"
    IconVSR_Vimeo90K_BI_4x = "IconVSR_Vimeo90K_BI_4x.pth"

    # AnimeSR
    AnimeSR_v1_PaperModel_4x = "AnimeSR_v1_PaperModel_4x.pth"
    AnimeSR_v2_4x = "AnimeSR_v2_4x.pth"
