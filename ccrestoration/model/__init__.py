from ccrestoration.util.registry import Registry

MODEL_REGISTRY: Registry = Registry("MODEL")

from ccrestoration.model.sr_base_model import SRBaseModel  # noqa
from ccrestoration.model.vsr_base_model import VSRBaseModel  # noqa
from ccrestoration.model.auxiliary_base_model import AuxiliaryBaseModel  # noqa
from ccrestoration.model.realesrgan_model import RealESRGANModel  # noqa
from ccrestoration.model.realcugan_model import RealCUGANModel  # noqa
from ccrestoration.model.edsr_model import EDSRModel  # noqa
from ccrestoration.model.swinir_model import SwinIRModel  # noqa
from ccrestoration.model.tile import tile_sr, calculate_pad_img_size  # noqa
from ccrestoration.model.edvr_model import EDVRModel  # noqa
