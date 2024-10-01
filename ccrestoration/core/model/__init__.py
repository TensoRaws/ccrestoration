from ccrestoration.utils.registry import Registry

MODEL_REGISTRY: Registry = Registry("MODEL")

from ccrestoration.core.model.sr_model import SRBaseModel  # noqa
from ccrestoration.core.model.realesrgan_model import RealESRGANModel  # noqa
