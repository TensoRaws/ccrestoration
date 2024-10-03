from ccrestoration import CONFIG_REGISTRY, ConfigType
from ccrestoration.cache_models import load_file_from_url


def test_cache_models() -> None:
    load_file_from_url(CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x))
