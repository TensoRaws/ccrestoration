from ccrestoration.core.arch import ARCH_REGISTRY


def test_auto_import() -> None:
    @ARCH_REGISTRY.register()
    class A:
        pass

    print(ARCH_REGISTRY)
    for k, v in ARCH_REGISTRY:
        print(k, v)

    print(ARCH_REGISTRY.get("A"))
