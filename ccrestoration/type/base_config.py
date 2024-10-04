from typing import Optional, Union

from pydantic import BaseModel, FileUrl, HttpUrl

from ccrestoration.type.arch import ArchType
from ccrestoration.type.model import ModelType


class BaseConfig(BaseModel):
    name: str
    url: Optional[HttpUrl] = None
    path: Optional[FileUrl] = None
    hash: Optional[str] = None
    arch: Union[ArchType, str]
    model: Union[ModelType, str]