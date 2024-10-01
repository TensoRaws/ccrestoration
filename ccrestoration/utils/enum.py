from enum import Enum


class MyStrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value
