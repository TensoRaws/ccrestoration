from ccrestoration import ARCH_REGISTRY, CONFIG_REGISTRY, BaseConfig, ConfigType


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

    cfg: BaseConfig = CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x)
    print(cfg.arch)
