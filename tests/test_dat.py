import os

import cv2
import pytest

from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.model import SRBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, get_device, load_image


class Test_DAT:
    def test_official_light(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.DAT_light_2x,
            ConfigType.DAT_light_3x,
            ConfigType.DAT_light_4x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)

    @pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") == "true", reason="Skip on CI test")
    def test_official(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.DAT_S_2x,
            ConfigType.DAT_S_3x,
            ConfigType.DAT_S_4x,
            ConfigType.DAT_2x,
            ConfigType.DAT_3x,
            ConfigType.DAT_4x,
            ConfigType.DAT_2_2x,
            ConfigType.DAT_2_3x,
            ConfigType.DAT_2_4x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)

    def test_custom(self) -> None:
        img1 = load_image()

        for k in [ConfigType.DAT_APISR_GAN_generator_4x]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)
