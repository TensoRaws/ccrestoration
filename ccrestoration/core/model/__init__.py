from enum import StrEnum

from ccrestoration.utils.registry import Registry


class ModelType(StrEnum):
    RealESRGAN = "RealESRGAN"


MOEDL_REGISTRY = Registry("MODEL")
