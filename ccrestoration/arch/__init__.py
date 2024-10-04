from ccrestoration.utils.registry import Registry

ARCH_REGISTRY: Registry = Registry("ARCH")

from ccrestoration.arch.rrdb_arch import RRDBNet  # noqa
from ccrestoration.arch.srvgg_arch import SRVGGNetCompact  # noqa
from ccrestoration.arch.upcunet_arch import UpCunet  # noqa
