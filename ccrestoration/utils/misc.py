import os
import random
from os import path as osp
from typing import Generator, Optional, Tuple, Union

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scandir(
    dir_path: str, suffix: Optional[Union[str, Tuple]] = None, recursive: bool = False, full_path: bool = False
) -> Generator[str, None, None]:
    """Scan a directory to find the interested files.

    Args:
        dir_path: Path of the directory.
        suffix: File suffix that we are interested in. Default: None.
        recursive: If set to True, recursively scan the directory. Default: False.
        full_path: If set to True, include the dir_path. Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(_dir_path: str, _suffix: Optional[Union[str, Tuple]], _recursive: bool) -> Generator[str, None, None]:
        for entry in os.scandir(_dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if _suffix is None:
                    yield return_path
                elif return_path.endswith(_suffix):
                    yield return_path
            else:
                if _recursive:
                    yield from _scandir(entry.path, _suffix=_suffix, _recursive=_recursive)
                else:
                    continue

    return _scandir(dir_path, _suffix=suffix, _recursive=recursive)
