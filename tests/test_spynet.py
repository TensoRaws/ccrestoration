from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.model import VSRBaseModel

from .util import get_device


class Test_SpyNet:
    def test_load(self) -> None:
        for k in [ConfigType.SpyNet_spynet_sintel_final]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            assert model is not None
