from typing import Any, Union

from ccrestoration.core.config import CONFIG_REGISTRY
from ccrestoration.core.type import BaseConfig, ConfigType


class AutoConfig:
    @staticmethod
    def from_pretrained(pretrained_model_name: Union[ConfigType, str]) -> Any:
        return CONFIG_REGISTRY.get(pretrained_model_name)

    @staticmethod
    def register(config: Union[BaseConfig, Any], name: str) -> None:
        CONFIG_REGISTRY.register(obj=config, name=name)
