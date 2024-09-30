from typing import Optional

from pydantic import BaseModel, FileUrl, HttpUrl

from ccrestoration.core.type import ArchType, ModelType


class BaseConfig(BaseModel):
    name: str
    url: Optional[HttpUrl] = None
    path: Optional[FileUrl] = None
    hash: Optional[str] = None
    arch: ArchType
    model: ModelType
