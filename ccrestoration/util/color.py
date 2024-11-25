import torch
from torch import Tensor


def rgb_to_yuv(image: Tensor) -> Tensor:
    r"""Convert an RGB image to YUV.

    .. image:: _static/img/rgb_to_yuv.png

    The image data is assumed to be in the range of :math:`(0, 1)`. The range of the output is of
    :math:`(0, 1)` to luma and the ranges of U and V are :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`,
    respectively.

    The YUV model adopted here follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]

    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    out: Tensor = torch.stack([y, u, v], -3)

    return out


def yuv_to_rgb(image: Tensor) -> Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of :math:`(0, 1)` for luma (Y). The ranges of U and V are
    :math:`(-0.436, 0.436)` and :math:`(-0.615, 0.615)`, respectively.

    YUV formula follows M/PAL values (see
    `BT.470-5 <https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-5-199802-S!!PDF-E.pdf>`_, Table 2,
    items 2.5 and 2.6).

    Args:
        image: YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if image.dim() < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: Tensor = image[..., 0, :, :]
    u: Tensor = image[..., 1, :, :]
    v: Tensor = image[..., 2, :, :]

    r: Tensor = y + 1.14 * v  # coefficient for g is 0
    g: Tensor = y + -0.396 * u - 0.581 * v
    b: Tensor = y + 2.029 * u  # coefficient for b is 0

    out: Tensor = torch.stack([r, g, b], -3)

    return out
