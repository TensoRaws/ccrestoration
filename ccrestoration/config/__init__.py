from ccrestoration.util.registry import RegistryConfigInstance

CONFIG_REGISTRY: RegistryConfigInstance = RegistryConfigInstance("CONFIG")

from ccrestoration.config.realesrgan_config import RealESRGANConfig  # noqa
from ccrestoration.config.realcugan_config import RealCUGANConfig  # noqa
from ccrestoration.config.edsr_config import EDSRConfig  # noqa
from ccrestoration.config.swinir_config import SwinIRConfig  # noqa
from ccrestoration.config.edvr_config import EDVRConfig  # noqa
from ccrestoration.config.spynet_config import SpyNetConfig  # noqa
from ccrestoration.config.basicvsr_config import BasicVSRConfig  # noqa
