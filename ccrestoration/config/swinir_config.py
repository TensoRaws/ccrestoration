from typing import Any, Optional, Tuple, Union

from pydantic import field_validator
from torch import nn

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import ArchType, BaseConfig, ConfigType, ModelType


class SwinIRConfig(BaseConfig):
    arch: Union[ArchType, str] = ArchType.SWINIR
    model: Union[ModelType, str] = ModelType.SwinIR
    scale: int = 2
    img_size: Union[int, Tuple[int, ...]] = 64
    patch_size: Union[int, Tuple[int, ...]] = 1
    in_chans: int = 3
    embed_dim: int = 96
    depths: Tuple[int, ...] = (6, 6, 6, 6)
    num_heads: Tuple[int, ...] = (6, 6, 6, 6)
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Any = nn.LayerNorm
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    img_range: float = 1.0
    upsampler: str = ""
    resi_connection: str = "1conv"

    @field_validator("scale")
    def scale_match(cls, v: int) -> int:
        if v not in [1, 2, 3, 4, 8]:
            raise ValueError("Upscale factor must be one of [1, 2, 3, 4, 8]")
        return v

    @field_validator("upsampler")
    def upsampler_match(cls, v: str) -> str:
        if v not in ["pixelshuffle", "pixelshuffledirect", "nearest+conv", ""]:
            raise ValueError("Upsampler must be one of ['pixelshuffle','pixelshuffledirect','nearest+conv', '']")
        return v

    @field_validator("resi_connection")
    def resi_connection_match(cls, v: str) -> str:
        if v not in ["1conv", "3conv"]:
            raise ValueError("Residual connection must be one of ['1conv', '3conv']")
        return v


SwinIRConfigs = [
    # official models
    SwinIRConfig(
        name=ConfigType.SwinIR_classicalSR_DF2K_s64w8_SwinIR_M_2x,
        path="/Users/tohru/Downloads/SwinIR_classicalSR_DF2K_s64w8_SwinIR_M_2x.pth",
        scale=2,
        window_size=8,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        embed_dim=180,
        mlp_ratio=2,
        upsampler="pixelshuffle",
    ),
    # community models
    SwinIRConfig(
        name=ConfigType.SwinIR_Bubble_AnimeScale_SwinIR_Small_v1_2x,
        path="/Users/tohru/Downloads/SwinIR_Bubble_AnimeScale_SwinIR_Small_v1_2x.pth",
        scale=2,
        img_size=32,
        window_size=8,
        embed_dim=60,
        mlp_ratio=2,
        upsampler="pixelshuffle",
    ),
]

for cfg in SwinIRConfigs:
    CONFIG_REGISTRY.register(cfg)
