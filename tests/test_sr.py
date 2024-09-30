import cv2
from torchvision import transforms

from ccrestoration.core.config import CONFIG_REGISTRY
from ccrestoration.core.model import MOEDL_REGISTRY
from ccrestoration.core.type import BaseConfig

from . import ASSETS_PATH, TEST_IMG_PATH


def test_sr() -> None:
    for k, _ in CONFIG_REGISTRY:
        cfg: BaseConfig = CONFIG_REGISTRY.get(k)
        model = MOEDL_REGISTRY.get(cfg.model)
        img = cv2.imread(TEST_IMG_PATH)
        img = transforms.ToTensor()(img).unsqueeze(0).to(model.device)
        img = model.inference(img)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")

        cv2.imwrite(ASSETS_PATH / f"test_{k}_out.jpg", img)
