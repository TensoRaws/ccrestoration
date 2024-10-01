from typing import Any, Optional, Union

import torch

from ccrestoration.core.config import CONFIG_REGISTRY
from ccrestoration.core.model import MODEL_REGISTRY
from ccrestoration.core.type import BaseConfig, ConfigType


class AutoModel:
    @staticmethod
    def from_pretrained(
        pretrained_model_name: Union[ConfigType, str],
        device: Optional[torch.device] = None,
        fp16: bool = True,
        compile: bool = False,
        compile_backend: Optional[str] = None,
    ) -> Any:
        config = CONFIG_REGISTRY.get(pretrained_model_name)
        return AutoModel.from_config(
            config=config,
            device=device,
            fp16=fp16,
            compile=compile,
            compile_backend=compile_backend,
        )

    @staticmethod
    def from_config(
        config: Union[BaseConfig, Any],
        device: Optional[torch.device] = None,
        fp16: bool = True,
        compile: bool = False,
        compile_backend: Optional[str] = None,
    ) -> Any:
        model = MODEL_REGISTRY.get(config.model)
        model = model(
            config=config,
            device=device,
            fp16=fp16,
            compile=compile,
            compile_backend=compile_backend,
        )

        return model

    @staticmethod
    def register(obj: Optional[Any] = None, name: Optional[str] = None) -> Any:
        """
        Register the given object under the name `obj.__name__` or the given name.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                _name = name
                if _name is None:
                    _name = func_or_class.__name__
                MODEL_REGISTRY.register(obj=func_or_class, name=_name)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        MODEL_REGISTRY.register(obj=obj, name=name)
