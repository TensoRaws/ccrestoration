from typing import Any

from ccrestoration.arch import DAT
from ccrestoration.config import DATConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.DAT)
class DATModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: DATConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        elif "model_state_dict" in state_dict:
            # For APISR's model
            state_dict = state_dict["model_state_dict"]

        model = DAT(
            img_size=cfg.img_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            split_size=cfg.split_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            act_layer=cfg.act_layer,
            norm_layer=cfg.norm_layer,
            use_chk=cfg.use_chk,
            upscale=cfg.scale,
            img_range=cfg.img_range,
            upsampler=cfg.upsampler,
            resi_connection=cfg.resi_connection,
            expansion_factor=cfg.expansion_factor,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model
