from typing import Any

from ccrestoration.arch import HAT
from ccrestoration.config import HATConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.HAT)
class HATModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: HATConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        elif "model_state_dict" in state_dict:
            # For APISR's model
            state_dict = state_dict["model_state_dict"]

        model = HAT(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            compress_ratio=cfg.compress_ratio,
            squeeze_factor=cfg.squeeze_factor,
            conv_scale=cfg.conv_scale,
            overlap_ratio=cfg.overlap_ratio,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=cfg.norm_layer,
            ape=cfg.ape,
            patch_norm=cfg.patch_norm,
            upscale=cfg.scale,
            img_range=cfg.img_range,
            upsampler=cfg.upsampler,
            resi_connection=cfg.resi_connection,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
