from ccrestoration.utils.registry import Registry

MOEDL_REGISTRY: Registry = Registry("MODEL")

from ccrestoration.core.model.realesrgan_model import RealESRGANModel  # noqa
