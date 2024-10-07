import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from ccrestoration.model import tile_sr

from .util import ASSETS_PATH, calculate_image_similarity, compare_image_size, get_device, load_image


def test_tile_sr() -> None:
    img0 = load_image()
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    img = transforms.ToTensor()(img).unsqueeze(0).to(get_device())

    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.interpolate(x, scale_factor=2, mode="bilinear")

    model = MyModel().to(get_device())

    img2 = tile_sr(model=model, scale=2, img=img)

    assert img2.shape[2] == img.shape[2] * 2 and img2.shape[3] == img.shape[3] * 2

    img2 = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2 = (img2 * 255).clip(0, 255).astype("uint8")

    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(ASSETS_PATH / "test_tile_sr_out.jpg"), img2)

    assert calculate_image_similarity(img0, img2)
    assert compare_image_size(img0, img2, 2)
