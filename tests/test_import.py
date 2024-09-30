from ccrestoration.core.arch import ARCH_REGISTRY
from ccrestoration.core.config import CONFIG_REGISTRY, RealESRGANConfig
from ccrestoration.core.type import ConfigType


def test_import() -> None:
    @ARCH_REGISTRY.register()
    class A:
        pass

    print(ARCH_REGISTRY)
    for k, v in ARCH_REGISTRY:
        print(k, v)

    print(ARCH_REGISTRY.get("A"))

    for k, v in CONFIG_REGISTRY:
        print(k, v)

    cfg: RealESRGANConfig = CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x)
    print(cfg.arch)
