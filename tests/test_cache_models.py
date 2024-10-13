from ccrestoration import CONFIG_REGISTRY, ConfigType
from ccrestoration.cache_models import load_file_from_url


def test_cache_models() -> None:
    load_file_from_url(CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x))


def test_cache_models_with_gh_proxy() -> None:
    load_file_from_url(
        config=CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x),
        force_download=True,
        gh_proxy="https://github.abskoop.workers.dev/",
    )
    load_file_from_url(
        config=CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x),
        force_download=True,
        gh_proxy="https://github.abskoop.workers.dev",
    )
