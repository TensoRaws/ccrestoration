from enum import Enum


# Enum for model type, use original name
class ModelType(str, Enum):
    RealESRGAN = "RealESRGAN"
    RealCUGAN = "RealCUGAN"
    EDSR = "EDSR"
    SwinIR = "SwinIR"
