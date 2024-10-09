import os
import sys

import cv2
import pytest
import torch

from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.model import VSRBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, get_device, load_image

DEVICE = get_device() if sys.platform != "darwin" else torch.device("cpu")


class Test_IconVSR:
    def test_load(self) -> None:
        for k in [ConfigType.IconVSR_REDS_4x, ConfigType.IconVSR_Vimeo90K_BD_4x, ConfigType.IconVSR_Vimeo90K_BI_4x]:
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=DEVICE)
            assert model is not None

    @pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") != "true", reason="Skip on local test")
    def test_official(self) -> None:
        img = load_image()
        imgList = [img, img, img, img, img, img, img]

        for k in [ConfigType.IconVSR_REDS_4x, ConfigType.IconVSR_Vimeo90K_BD_4x, ConfigType.IconVSR_Vimeo90K_BI_4x]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=DEVICE)
            print(model.device)

            imgOutList = model.inference_image_list(imgList)

            assert len(imgOutList) == len(imgList)

            for i, v in enumerate(imgOutList):
                cv2.imwrite(str(ASSETS_PATH / f"test_{k}_{i}_out.jpg"), v)

                assert calculate_image_similarity(img, v)
                assert compare_image_size(img, v, cfg.scale)
