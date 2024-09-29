from ccrestoration.core.arch import SISR_ARCH_REGISTRY


def test_auto_import() -> None:
    @SISR_ARCH_REGISTRY.register()
    class A:
        pass

    print(SISR_ARCH_REGISTRY)
    for k, v in SISR_ARCH_REGISTRY:
        print(k, v)

    print(SISR_ARCH_REGISTRY.get("A"))
