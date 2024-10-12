import sys

import cv2
import pytest
import torch

from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.model import SRBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, get_device, load_image

print(f"PyTorch version: {torch.__version__}")
torch_2: bool = torch.__version__.startswith("2")


def test_inference() -> None:
    tensor1 = torch.rand(1, 3, 256, 256).to(get_device())

    k = ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x

    model: SRBaseModel = AutoModel.from_pretrained(pretrained_model_name=k, fp16=False, device=get_device())

    t2 = model(tensor1)
    t3 = model.inference(tensor1)
    assert t2.equal(t3)


@pytest.mark.skipif(not torch_2, reason="Skip test if PyTorch version is under 2.0.0")
def test_sr_fp16() -> None:
    img1 = load_image()
    k = ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x

    cfg: BaseConfig = AutoConfig.from_pretrained(k)
    model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=True, device=get_device(), tile=None)

    img2 = model.inference_image(img1)

    cv2.imwrite(str(ASSETS_PATH / f"test_fp16_{k}_out.jpg"), img2)

    assert calculate_image_similarity(img1, img2)
    assert compare_image_size(img1, img2, cfg.scale)


@pytest.mark.skipif(
    sys.platform == "win32" or not torch_2, reason="Skip test torch.compile on Windows or PyTorch version under 2.0.0"
)
def test_sr_compile() -> None:
    img1 = load_image()
    k = ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x

    model: SRBaseModel = AutoModel.from_pretrained(
        pretrained_model_name=k, fp16=True, compile=True, device=get_device(), tile=None
    )

    img2 = model.inference_image(img1)

    assert calculate_image_similarity(img1, img2)
