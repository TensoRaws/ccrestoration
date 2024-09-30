from enum import StrEnum

from ccrestoration.utils.registry import Registry


class ArchType(StrEnum):
    RRDB = "RRDB"
    SRVGG = "SRVGG"


ARCH_REGISTRY = Registry("ARCH")

from ccrestoration.core.arch.rrdb_arch import RRDBNet  # noqa
from ccrestoration.core.arch.srvgg_arch import SRVGGNetCompact  # noqa
