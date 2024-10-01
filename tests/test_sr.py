import cv2

from ccrestoration.core.config import CONFIG_REGISTRY
from ccrestoration.core.model import MODEL_REGISTRY, SRBaseModel
from ccrestoration.core.type import BaseConfig, ConfigType

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, load_image


def test_sr() -> None:
    img1 = load_image()

    for k, _ in CONFIG_REGISTRY:
        print(f"Testing {k}")
        cfg: BaseConfig = CONFIG_REGISTRY.get(k)

        model: SRBaseModel = MODEL_REGISTRY.get(cfg.model)

        model = model(config=cfg, fp16=False)  # type: ignore

        img2 = model.inference_image(img1)

        cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

        assert calculate_image_similarity(img1, img2)
        assert compare_image_size(img1, img2, cfg.scale)


def test_sr_fp16() -> None:
    img1 = load_image()
    k = ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x

    cfg: BaseConfig = CONFIG_REGISTRY.get(k)

    model: SRBaseModel = MODEL_REGISTRY.get(cfg.model)
    model = model(config=cfg, fp16=True)  # type: ignore

    img2 = model.inference_image(img1)

    cv2.imwrite(str(ASSETS_PATH / f"test_fp16_{k}_out.jpg"), img2)

    assert calculate_image_similarity(img1, img2)
    assert compare_image_size(img1, img2, cfg.scale)


def test_sr_compile() -> None:
    img1 = load_image()
    k = ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x

    cfg: BaseConfig = CONFIG_REGISTRY.get(k)

    model: SRBaseModel = MODEL_REGISTRY.get(cfg.model)

    m1 = model(config=cfg, fp16=True, compile=True)  # type: ignore

    img2 = m1.inference_image(img1)

    assert calculate_image_similarity(img1, img2)
    assert compare_image_size(img1, img2, cfg.scale)
