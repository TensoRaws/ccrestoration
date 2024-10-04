from ccrestoration.utils.registry import RegistryConfigInstance

CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from ccrestoration.core.config.realesrgan_config import RealESRGANConfig  # noqa
from ccrestoration.core.config.realcugan_config import RealCUGANConfig  # noqa
