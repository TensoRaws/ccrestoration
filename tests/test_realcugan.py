import cv2

from ccrestoration import AutoConfig, AutoModel, BaseConfig, ConfigType
from ccrestoration.config import RealCUGANConfig
from ccrestoration.model import SRBaseModel

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, get_device, load_image


class Test_RealCUGAN:
    def test_pro(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.RealCUGAN_Pro_Conservative_2x,
            ConfigType.RealCUGAN_Pro_Conservative_3x,
            ConfigType.RealCUGAN_Pro_Denoise3x_2x,
            ConfigType.RealCUGAN_Pro_Denoise3x_3x,
            ConfigType.RealCUGAN_Pro_No_Denoise_2x,
            ConfigType.RealCUGAN_Pro_No_Denoise_3x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            img2 = model.inference_image(img1)

            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)

    def test_non_pro(self) -> None:
        img1 = load_image()

        for k in [
            ConfigType.RealCUGAN_Conservative_2x,
            ConfigType.RealCUGAN_Denoise1x_2x,
            ConfigType.RealCUGAN_Denoise2x_2x,
            ConfigType.RealCUGAN_Denoise3x_2x,
            ConfigType.RealCUGAN_No_Denoise_2x,
            ConfigType.RealCUGAN_Conservative_3x,
            ConfigType.RealCUGAN_Denoise3x_3x,
            ConfigType.RealCUGAN_No_Denoise_3x,
            ConfigType.RealCUGAN_Conservative_4x,
            ConfigType.RealCUGAN_Denoise3x_4x,
            ConfigType.RealCUGAN_No_Denoise_4x,
        ]:
            print(f"Testing {k}")
            cfg: BaseConfig = AutoConfig.from_pretrained(k)
            model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            img2 = model.inference_image(img1)

            cv2.imwrite(str(ASSETS_PATH / f"test_{k}_out.jpg"), img2)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)

    def test_alpha(self) -> None:
        img1 = load_image()
        k = ConfigType.RealCUGAN_Conservative_2x

        for alpha in [0.4, 0.7, 0.9, 1.3, 1.9]:
            print(f"Testing alpha={alpha}")
            cfg: RealCUGANConfig = AutoConfig.from_pretrained(k)
            cfg.alpha = alpha
            model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            img2 = model.inference_image(img1)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)

    def test_cache_mode(self) -> None:
        img1 = load_image()
        k = ConfigType.RealCUGAN_Conservative_2x

        for c in [0, 1, 2, 3]:
            print(f"Testing cache_mode={c}")
            cfg: RealCUGANConfig = AutoConfig.from_pretrained(k)
            cfg.cache_mode = c
            model: SRBaseModel = AutoModel.from_config(config=cfg, fp16=False, device=get_device())
            print(model.device)

            img2 = model.inference_image(img1)

            assert calculate_image_similarity(img1, img2)
            assert compare_image_size(img1, img2, cfg.scale)
