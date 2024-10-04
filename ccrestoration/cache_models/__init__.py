import hashlib
import os
from pathlib import Path
from typing import Optional

from tenacity import retry, stop_after_attempt, stop_after_delay, wait_random
from torch.hub import download_url_to_file

from ccrestoration.type import BaseConfig

CACHE_PATH = Path(__file__).resolve().parent.absolute()


def get_file_sha256(file_path: str, blocksize: int = 1 << 20) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(blocksize)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def load_file_from_url(
    config: BaseConfig, force_download: bool = False, progress: bool = True, model_dir: Optional[str] = None
) -> str:
    """
    Load file form http url, will download models if necessary.

    Reference: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    :param config: The config object.
    :param force_download: Whether to force download the file.
    :param progress: Whether to show the download progress.
    :param model_dir: The path to save the downloaded model. Should be a full path. If None, use default cache path.
    :return:
    """

    if model_dir is None:
        model_dir = str(CACHE_PATH)

    cached_file_path = os.path.abspath(os.path.join(model_dir, config.name))

    if not os.path.exists(cached_file_path) or force_download:
        print(f"Downloading: {config.url} to {cached_file_path}\n")

        @retry(wait=wait_random(min=3, max=5), stop=stop_after_delay(10) | stop_after_attempt(30))
        def _download() -> None:
            try:
                download_url_to_file(str(config.url), cached_file_path, hash_prefix=None, progress=progress)
            except Exception as e:
                print(f"Download failed: {e}, retrying...")
                raise e

        _download()

    if config.hash is not None:
        get_hash = get_file_sha256(cached_file_path)
        if get_hash != config.hash:
            raise ValueError(
                f"File {cached_file_path} hash mismatched with config hash {config.hash}, compare with {get_hash}"
            )

    return cached_file_path


if __name__ == "__main__":
    # get all model files sha256
    for root, _, files in os.walk(CACHE_PATH):
        for file in files:
            if not file.endswith(".pth") and not file.endswith(".pt"):
                continue
            file_path = os.path.join(root, file)
            name = os.path.basename(file_path)
            print(f"{name}: {get_file_sha256(file_path)}")
