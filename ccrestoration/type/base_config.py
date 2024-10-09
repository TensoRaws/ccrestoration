from typing import Optional, Union

from pydantic import BaseModel, FilePath, HttpUrl

from ccrestoration.type.arch import ArchType
from ccrestoration.type.model import ModelType


class BaseConfig(BaseModel):
    name: str
    url: Optional[HttpUrl] = None
    path: Optional[FilePath] = None
    hash: Optional[str] = None
    arch: Union[ArchType, str]
    model: Union[ModelType, str]
    scale: int = 2
    length: int = 7
