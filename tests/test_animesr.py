import cv2

from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.model import VSRBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, get_device, load_image


class Test_AnimeSR:
    def test_official(self) -> None:
        img = load_image()
        imgList = [img, img, img]

        for k in [ConfigType.AnimeSR_v1_PaperModel_4x, ConfigType.AnimeSR_v2_4x]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: VSRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            imgOutList = model.inference_image_list(imgList)

            assert len(imgOutList) == len(imgList)

            for i, v in enumerate(imgOutList):
                cv2.imwrite(str(ASSETS_PATH / f"test_{k}_{i}_out.jpg"), v)

                assert calculate_image_similarity(img, v)
                assert compare_image_size(img, v, cfg.scale)
