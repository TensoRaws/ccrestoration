import torch

from ccrestoration.cache_models import load_file_from_url
from ccrestoration.core.arch import RRDBNet, SRVGGNetCompact
from ccrestoration.core.config import RealESRGANConfig
from ccrestoration.core.model import MOEDL_REGISTRY
from ccrestoration.core.type import ArchType, BaseModelInterface, ModelType


@MOEDL_REGISTRY.register(name=ModelType.RealESRGAN)
class RealESRGANModel(BaseModelInterface):
    def load_model(self) -> torch.nn.Module:
        cfg: RealESRGANConfig = self.config

        if cfg.path is not None:
            model_path = str(cfg.path)
        else:
            try:
                model_path = load_file_from_url(cfg)
            except Exception as e:
                print(f"Error: {e}, try force download the model...")
                model_path = load_file_from_url(cfg, force_download=True)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        with torch.device("meta"):
            if cfg.arch == ArchType.RRDB:
                model = RRDBNet(
                    num_in_ch=cfg.num_in_ch,
                    num_out_ch=cfg.num_out_ch,
                    scale=cfg.scale,
                    num_feat=cfg.num_feat,
                    num_block=cfg.num_block,
                    num_grow_ch=cfg.num_grow_ch,
                )
            elif self.config.arch == ArchType.SRVGG:
                model = SRVGGNetCompact(
                    num_in_ch=cfg.num_in_ch,
                    num_out_ch=cfg.num_out_ch,
                    upscale=cfg.scale,
                    num_feat=cfg.num_feat,
                    num_conv=cfg.num_conv,
                    act_type=cfg.act_type,
                )
            else:
                raise NotImplementedError(f"Arch {cfg.arch} is not implemented.")

        model.load_state_dict(state_dict, assign=True)
        model.eval().to(self.device)
        return model
