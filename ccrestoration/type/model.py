from enum import Enum


# Enum for model type, use original name
class ModelType(str, Enum):
    # ------------------------------------- Single Image Super-Resolution ----------------------------------------------

    RealESRGAN = "RealESRGAN"
    RealCUGAN = "RealCUGAN"
    EDSR = "EDSR"
    SwinIR = "SwinIR"

    # ------------------------------------- Auxiliary Network ----------------------------------------------------------

    SpyNet = "SpyNet"
    EDVRFeatureExtractor = "EDVRFeatureExtractor"

    # ------------------------------------- Video Super-Resolution -----------------------------------------------------

    EDVR = "EDVR"
    BasicVSR = "BasicVSR"
    IconVSR = "IconVSR"
    AnimeSR = "AnimeSR"
