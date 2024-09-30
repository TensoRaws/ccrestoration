from ccrestoration.cache_models import load_file_from_url
from ccrestoration.core.config import CONFIG_REGISTRY
from ccrestoration.core.type import ConfigType


def test_cache_models() -> None:
    load_file_from_url(CONFIG_REGISTRY.get(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x))
