import cv2

from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.core.model import SRBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, load_image


def test_sr() -> None:
    img1 = load_image()

    for k in ConfigType:
        print(f"Testing {k}")
        cfg: BaseConfig = AutoConfig.from_pretrained(k)
        model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False)
        print(model.device)

        img2 = model.inference_image(img1)

        cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

        assert calculate_image_similarity(img1, img2)
        assert compare_image_size(img1, img2, cfg.scale)


def test_sr_fp16() -> None:
    img1 = load_image()
    k = ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x

    cfg: BaseConfig = AutoConfig.from_pretrained(k)
    model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=True)

    img2 = model.inference_image(img1)

    cv2.imwrite(str(ASSETS_PATH / f"test_fp16_{k}_out.jpg"), img2)

    assert calculate_image_similarity(img1, img2)
    assert compare_image_size(img1, img2, cfg.scale)


def test_sr_compile() -> None:
    img1 = load_image()
    k = ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x

    model: SRBaseModel = AutoModel.from_pretrained(pretrained_model_name=k, fp16=True, compile=True)

    img2 = model.inference_image(img1)

    assert calculate_image_similarity(img1, img2)
