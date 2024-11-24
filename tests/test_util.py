import cv2
import pytest
import torch
from torchvision import transforms

from ccrestoration.util.color import rgb_to_yuv, yuv_to_rgb
from ccrestoration.util.device import DEFAULT_DEVICE

from .util import calculate_image_similarity, load_image


def test_device() -> None:
    print(DEFAULT_DEVICE)


def test_color() -> None:
    with pytest.raises(TypeError):
        rgb_to_yuv(1)
    with pytest.raises(TypeError):
        yuv_to_rgb(1)

    with pytest.raises(ValueError):
        rgb_to_yuv(torch.zeros(1, 1))
    with pytest.raises(ValueError):
        yuv_to_rgb(torch.zeros(1, 1))

    img = load_image()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transforms.ToTensor()(img).unsqueeze(0).to("cpu")

    img = rgb_to_yuv(img)
    img = yuv_to_rgb(img)

    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    assert calculate_image_similarity(img, load_image())
