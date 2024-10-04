from ccrestoration.utils.registry import Registry

MODEL_REGISTRY: Registry = Registry("MODEL")

from ccrestoration.model.sr_model import SRBaseModel  # noqa
from ccrestoration.model.realesrgan_model import RealESRGANModel  # noqa
from ccrestoration.model.realcugan_model import RealCUGANModel  # noqa
