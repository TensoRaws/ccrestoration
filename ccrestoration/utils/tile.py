import math

import torch
import torch.nn.functional as F


def tile_process(
    img: torch.Tensor,
    scale: int,
    tile: list[int],
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    module: torch.nn.Module,
) -> torch.Tensor:
    batch, channel, height, width = img.shape
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
            input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), "replicate")

            output_tile = module(input_tile)

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
