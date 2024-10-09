import sys

import cv2

from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.model import VSRBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, get_device, load_image

DEVICE = get_device() if sys.platform != "darwin" else "cpu"


class Test_EDVR:
    def test_official_M(self) -> None:
        img = load_image()
        imgList = [img, img, img, img, img]

        for k in [ConfigType.EDVR_M_SR_REDS_official_4x]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=DEVICE)
            print(model.device)

            img2 = model.inference_image_list(imgList)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2[0])

            assert calculate_image_similarity(img, img2[0])
            assert compare_image_size(img, img2[0], cfg.scale)
