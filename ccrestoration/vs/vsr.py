from typing import Any, Callable, Dict, Union

import torch
import vapoursynth as vs

from ccrestoration.vs.convert import frame_to_tensor, tensor_to_frame


def inference_vsr(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    length: int,
    device: torch.device,
    _frame_to_tensor: Callable[[vs.VideoFrame, torch.device], torch.Tensor] = frame_to_tensor,
    _tensor_to_frame: Callable[[torch.Tensor, vs.VideoFrame], vs.VideoFrame] = tensor_to_frame,
) -> vs.VideoNode:
    """
    Inference the video with the model, the clip should be a vapoursynth clip

    f1, f2, f3, f4 -> f1', f2', f3', f4'

    For the multi frame output model, the inference function should accept a tensor with shape (1, n, c, h, w)
    And return a tensor with shape (1, n, c, h, w)

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param length: The length of the input frames
    :param device: The device
    :param _frame_to_tensor: The function to convert the frame to tensor
    :param _tensor_to_frame: The function to convert the tensor to frame
    :return:
    """

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("Only vs.RGBH and vs.RGBS formats are supported")

    if length > clip.num_frames:
        raise ValueError("The length of the input frames should be less than the number of frames in the clip")

    if length < 2:
        raise ValueError("The length of the input frames should be greater than 1")

    cache: Dict[int, torch.Tensor] = {}

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        if n not in cache:
            cache.clear()

            img = []
            for i in range(length):
                if n + i >= clip.num_frames:
                    img.append(frame_to_tensor(clip.get_frame(clip.num_frames - 1), device=device).unsqueeze(0))
                img.append(frame_to_tensor(clip.get_frame(n + i), device=device).unsqueeze(0))

            img = torch.stack(img, dim=1)

            output = inference(img)

            for i in range(output.shape[0]):
                cache[n + i] = output[i, :, :, :]

        return _tensor_to_frame(cache[n], f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.ModifyFrame([clip, new_clip], _inference)


def inference_vsr_one_frame_out(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    length: int,
    device: torch.device,
    _frame_to_tensor: Callable[[vs.VideoFrame, torch.device], torch.Tensor] = frame_to_tensor,
    _tensor_to_frame: Callable[[torch.Tensor, vs.VideoFrame], vs.VideoFrame] = tensor_to_frame,
) -> vs.VideoNode:
    """
    Inference the video with the model, the clip should be a vapoursynth clip

    f-2, f-1, f0, f1, f2 -> f0'

    For these models, the inference function should accept a tensor with shape (1, n, c, h, w)
    And return a tensor with shape (1, 1, c, h, w)

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param length: The length of the input frames, should be odd
    :param device: The device
    :param _frame_to_tensor: The function to convert the frame to tensor
    :param _tensor_to_frame: The function to convert the tensor to frame
    :return:
    """

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("Only vs.RGBH and vs.RGBS formats are supported")

    if length > clip.num_frames:
        raise ValueError("The length of the input frames should be less than the number of frames in the clip")

    if length % 2 == 0:
        raise ValueError("The length of the input frames should be odd")

    if length < 2:
        raise ValueError("The length of the input frames should be greater than 1")

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        img = []
        for i in range(length):
            index = i - length // 2 + n
            if index < 0:
                img.append(frame_to_tensor(clip.get_frame(0), device=device).unsqueeze(0))

            elif index >= clip.num_frames:
                img.append(frame_to_tensor(clip.get_frame(clip.num_frames - 1), device=device).unsqueeze(0))

            else:
                img.append(frame_to_tensor(clip.get_frame(index), device=device).unsqueeze(0))

        img = torch.stack(img, dim=1)

        output = inference(img)

        return tensor_to_frame(output[0, 0, :, :, :], f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.ModifyFrame([clip, new_clip], _inference)
