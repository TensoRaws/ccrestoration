from enum import Enum


class ModelType(str, Enum):
    RealESRGAN = "RealESRGAN"
    RealCUGAN = "RealCUGAN"
    EDSR = "EDSR"
