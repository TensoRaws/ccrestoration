import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def calculate_pad_img_size(width: int, height: int, tile: Tuple[int, int], tile_pad: int) -> Tuple[int, int]:
    """
    Calculate the size of the padded image.

    :param width: The width of the original image
    :param height: The height of the original image
    :param tile: The tile size, tile[0] is width, tile[1] is height
    :param tile_pad: The padding size for each tile
    :return: The size of the padded image as a tuple (padded_width, padded_height)
    """
    pad_w = math.ceil(min(tile[0] + 2 * tile_pad, width) / tile_pad) * tile_pad
    pad_h = math.ceil(min(tile[1] + 2 * tile_pad, height) / tile_pad) * tile_pad

    return pad_w, pad_h


def tile_sr(
    model: torch.nn.Module,
    scale: int,
    img: torch.Tensor,
    tile: Tuple[int, int] = (128, 128),
    tile_pad: int = 8,
    pad_img: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Process image by tiles

    :param model: The model to inference
    :param scale: The scale factor
    :param img: The input image
    :param tile: The tile size, tile[0] is width, tile[1] is height
    :param tile_pad: The padding size for each tile
    :param pad_img: The size for the padded image, pad[0] is width, pad[1] is height, None for auto calculate
    """
    batch, channel, height, width = img.shape

    if pad_img is None:
        pad_img = calculate_pad_img_size(width, height, tile, tile_pad)

    output_shape = (batch, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile[0])
    tiles_y = math.ceil(height / tile[1])

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile[0]
            ofs_y = y * tile[1]

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile[0], width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile[1], height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            h, w = input_tile.shape[2:]
            input_tile = F.pad(input_tile, (0, pad_img[0] - w, 0, pad_img[1] - h), "replicate")

            # process tile
            output_tile = model(input_tile)

            output_tile = output_tile[:, :, : h * scale, : w * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output


def tile_vsr(
    model: torch.nn.Module,
    scale: int,
    img: torch.Tensor,
    one_frame_out: bool = False,
    tile: Tuple[int, int] = (128, 128),
    tile_pad: int = 8,
    pad_img: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Process image group by tiles

    when one_frame_out is True:
        f-2, f-1, f0, f1, f2 -> f0'
    when one_frame_out is False:
        f1, f2, f3, f4 -> f1', f2', f3', f4'

    :param model: The model to inference
    :param scale: The scale factor
    :param one_frame_out: The output is one frame or not
    :param img: The input image
    :param tile: The tile size, tile[0] is width, tile[1] is height
    :param tile_pad: The padding size for each tile
    :param pad_img: The size for the padded image, pad[0] is width, pad[1] is height, None for auto calculate
    """
    batch, length, channel, height, width = img.shape

    if pad_img is None:
        pad_img = calculate_pad_img_size(width, height, tile, tile_pad)

    if one_frame_out:
        output_shape = (batch, 1, channel, height * scale, width * scale)
    else:
        output_shape = (batch, length, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile[0])
    tiles_y = math.ceil(height / tile[1])

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile[0]
            ofs_y = y * tile[1]

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile[0], width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile[1], height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            h, w = input_tile.shape[3:]

            input_tile = F.pad(input_tile, (0, pad_img[0] - w, 0, pad_img[1] - h, 0, 0), "replicate")

            # process tile
            output_tile = model(input_tile)

            output_tile = output_tile[:, :, :, : h * scale, : w * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output
