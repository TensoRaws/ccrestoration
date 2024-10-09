from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.model import VSRBaseModel

from .util import get_device


class Test_EDVRFeatureExtractor:
    def test_load(self) -> None:
        for k in [
            ConfigType.EDVRFeatureExtractor_REDS_pretrained_for_IconVSR,
            ConfigType.EDVRFeatureExtractor_Vimeo90K_pretrained_for_IconVSR,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            assert model is not None
