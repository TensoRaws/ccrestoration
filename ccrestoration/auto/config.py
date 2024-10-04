from typing import Any, Optional, Union

from ccrestoration.config import CONFIG_REGISTRY
from ccrestoration.type import BaseConfig, ConfigType


class AutoConfig:
    @staticmethod
    def from_pretrained(pretrained_model_name: Union[ConfigType, str]) -> Any:
        """
        Get a config instance of a pretrained model configuration.

        :param pretrained_model_name: The name of the pretrained model configuration
        :return:
        """
        return CONFIG_REGISTRY.get(pretrained_model_name)

    @staticmethod
    def register(config: Union[BaseConfig, Any], name: Optional[str] = None) -> None:
        """
        Register the given config class instance under the name BaseConfig.name or the given name.
        Can be used as a function call. See docstring of this class for usage.

        :param config: The config class instance to register
        :param name: The name to register the config class instance under. If None, use BaseConfig.name
        :return:
        """
        # used as a function call
        CONFIG_REGISTRY.register(obj=config, name=name)
