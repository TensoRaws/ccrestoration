from enum import Enum


# Enum for the architecture type, use capital letters
class ArchType(str, Enum):
    # ------------------------------------- Single Image Super-Resolution ----------------------------------------------

    RRDB = "RRDB"
    SRVGG = "SRVGG"
    UPCUNET = "UPCUNET"
    EDSR = "EDSR"
    SWINIR = "SWINIR"
    SCUNET = "SCUNET"

    # ------------------------------------- Auxiliary Network ----------------------------------------------------------

    SPYNET = "SPYNET"
    EDVRFEATUREEXTACTOR = "EDVRFEATUREEXTACTOR"

    # ------------------------------------- Video Super-Resolution -----------------------------------------------------

    EDVR = "EDVR"
    BASICVSR = "BASICVSR"
    ICONVSR = "ICONVSR"
    MSRSWVSR = "MSRSWVSR"
