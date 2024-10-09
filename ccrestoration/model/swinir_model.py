from typing import Any

from ccrestoration.arch import SwinIR
from ccrestoration.config import SwinIRConfig
from ccrestoration.model import MODEL_REGISTRY
from ccrestoration.model.sr_base_model import SRBaseModel
from ccrestoration.type import ModelType


@MODEL_REGISTRY.register(name=ModelType.SwinIR)
class SwinIRModel(SRBaseModel):
    def load_model(self) -> Any:
        cfg: SwinIRConfig = self.config
        state_dict = self.get_state_dict()

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model = SwinIR(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            norm_layer=cfg.norm_layer,
            ape=cfg.ape,
            patch_norm=cfg.patch_norm,
            use_checkpoint=cfg.use_checkpoint,
            upscale=cfg.scale,
            img_range=cfg.img_range,
            upsampler=cfg.upsampler,
            resi_connection=cfg.resi_connection,
        )

        model.load_state_dict(state_dict)
        model.eval().to(self.device)
        return model

    # the latest version of SwinIR has auto padding, so we don't need to pad the image
    # @torch.inference_mode()  # type: ignore
    # def inference(self, img: torch.Tensor) -> torch.Tensor:
    #     cfg: SwinIRConfig = self.config
    #
    #     _, _, h_old, w_old = img.size()
    #     window_size = cfg.window_size
    #     h_pad = (h_old // window_size + 1) * window_size - h_old
    #     w_pad = (w_old // window_size + 1) * window_size - w_old
    #     img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h_old + h_pad, :]
    #     img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w_old + w_pad]
    #
    #     img = self.model(img)
    #
    #     img = img[..., : h_old * cfg.scale, : w_old * cfg.scale]
    #     return img
