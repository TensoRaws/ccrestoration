from ccrestoration.utils.registry import Registry

ARCH_REGISTRY: Registry = Registry("ARCH")

from ccrestoration.core.arch.rrdb_arch import RRDBNet  # noqa
from ccrestoration.core.arch.srvgg_arch import SRVGGNetCompact  # noqa
