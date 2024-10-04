from ccrestoration.utils.registry import RegistryConfigInstance

CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from ccrestoration.config.realesrgan_config import RealESRGANConfig  # noqa
from ccrestoration.config.realcugan_config import RealCUGANConfig  # noqa
