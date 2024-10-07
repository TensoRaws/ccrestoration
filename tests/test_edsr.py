import cv2

from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.model import SRBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, get_device, load_image


class Test_EDSR:
    def test_official_M(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.EDSR_Mx2_f64b16_DIV2K_official_2x,
            ConfigType.EDSR_Mx3_f64b16_DIV2K_official_3x,
            ConfigType.EDSR_Mx4_f64b16_DIV2K_official_4x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            img2 = model.inference_image(img1)
            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)
