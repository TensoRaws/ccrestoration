from enum import Enum

from ccrestoration.utils.registry import Registry


class ModelType(str, Enum):
    RealESRGAN = "RealESRGAN"


MOEDL_REGISTRY = Registry("MODEL")
