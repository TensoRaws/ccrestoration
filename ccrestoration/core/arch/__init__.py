from enum import Enum

from ccrestoration.utils.registry import Registry


class ArchType(str, Enum):
    RRDB = "RRDB"
    SRVGG = "SRVGG"


ARCH_REGISTRY = Registry("ARCH")

from ccrestoration.core.arch.rrdb_arch import RRDBNet  # noqa
from ccrestoration.core.arch.srvgg_arch import SRVGGNetCompact  # noqa
