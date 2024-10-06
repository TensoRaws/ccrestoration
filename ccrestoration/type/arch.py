from enum import Enum


# Enum for the architecture type, use capital letters
class ArchType(str, Enum):
    RRDB = "RRDB"
    SRVGG = "SRVGG"
    UPCUNET = "UPCUNET"
    EDSR = "EDSR"
    SWINIR = "SWINIR"
