import importlib
from os import path as osp

from ccrestoration.utils.misc import scandir
from ccrestoration.utils.registry import Registry

SISR_ARCH_REGISTRY = Registry("SISR_ARCH")

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder: str = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(dir_path=arch_folder, suffix="_arch.py")]
# import all the arch modules
_arch_modules_ = [importlib.import_module(f"ccrestoration.core.arch.{file_name}") for file_name in arch_filenames]
