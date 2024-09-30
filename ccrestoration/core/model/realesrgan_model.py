import torch

from ccrestoration.cache_models import load_file_from_url
from ccrestoration.core.arch import ARCH_REGISTRY, ArchType, RRDBNet, SRVGGNetCompact
from ccrestoration.core.model.base_model import BaseModelInterface


class RealESRGANModel(BaseModelInterface):
    def load_model(self) -> torch.nn.Module:
        if self.config.path is not None:
            model_path = str(self.config.path)
        else:
            try:
                model_path = load_file_from_url(self.config)
            except Exception as e:
                print(f"Error: {e}, try force download the model...")
                model_path = load_file_from_url(self.config, force_download=True)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        with torch.device("meta"):
            if self.config.arch == ArchType.RRDB:
                model_rrdb: RRDBNet = ARCH_REGISTRY.get(self.config.arch)
                model = model_rrdb(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
            elif self.config.arch == ArchType.SRVGG:
                model_srvgg: SRVGGNetCompact = ARCH_REGISTRY.get(self.config.arch)
                model = model_srvgg(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
            else:
                raise NotImplementedError(f"Arch {self.config.arch} is not implemented.")

        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        model.half()
        return model
