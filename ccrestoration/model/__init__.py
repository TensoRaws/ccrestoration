from ccrestoration.util.registry import Registry

MODEL_REGISTRY: Registry = Registry("MODEL")

from ccrestoration.model.sr_base_model import SRBaseModel  # noqa
from ccrestoration.model.vsr_base_model import VSRBaseModel  # noqa
from ccrestoration.model.auxiliary_base_model import AuxiliaryBaseModel  # noqa
from ccrestoration.model.realesrgan_model import RealESRGANModel  # noqa
from ccrestoration.model.realcugan_model import RealCUGANModel  # noqa
from ccrestoration.model.edsr_model import EDSRModel  # noqa
from ccrestoration.model.swinir_model import SwinIRModel  # noqa
from ccrestoration.model.edvr_model import EDVRModel, EDVRFeatureExtractorModel  # noqa
from ccrestoration.model.tile import tile_sr, tile_vsr  # noqa
from ccrestoration.model.spynet_model import SpyNetModel  # noqa
from ccrestoration.model.basicvsr_model import BasicVSRModel  # noqa
from ccrestoration.model.iconvsr_model import IconVSRModel  # noqa
from ccrestoration.model.animesr_model import AnimeSRModel  # noqa
