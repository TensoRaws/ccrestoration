"""
Reference: https://github.com/HolyWu/vs-realesrgan

Thanks to HolyWu for his great work on super-resolution.
"""

from threading import Lock
from typing import Any, Callable, Union

import torch
import vapoursynth as vs

from ccrestoration.vs.convert import frame_to_tensor, tensor_to_frame


def inference_sr(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    device: torch.device,
) -> vs.VideoNode:
    """
    Inference the video with the model, the clip should be a vapoursynth clip

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param device: The device
    :return:
    """

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("Only vs.RGBH and vs.RGBS formats are supported")

    if device.type == torch.device("cuda").type:
        return inference_sr_cuda(inference, clip, scale, device)
    else:
        return inference_sr_general(inference, clip, scale, device)


def inference_sr_general(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    device: torch.device,
) -> vs.VideoNode:
    """
    Inference for General devices

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param device: The device
    :return:
    """

    f2t_stream_lock = Lock()
    inf_stream_lock = Lock()
    t2f_stream_lock = Lock()

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        with f2t_stream_lock:
            img = frame_to_tensor(f[0], device).unsqueeze(0)

        with inf_stream_lock:
            output = inference(img)

        with t2f_stream_lock:
            res = tensor_to_frame(output, f[1].copy())

        return res

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], _inference), clip_src=[clip, new_clip]
    )


def inference_sr_cuda(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    device: torch.device,
) -> vs.VideoNode:
    """
    Inference for CUDA devices

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param device: The device
    :return:
    """

    f2t_stream_lock = Lock()
    inf_stream_lock = Lock()
    t2f_stream_lock = Lock()

    f2t_stream = torch.cuda.Stream(device)
    inf_stream = torch.cuda.Stream(device)
    t2f_stream = torch.cuda.Stream(device)

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        with f2t_stream_lock, torch.cuda.stream(f2t_stream):
            img = frame_to_tensor(f[0], device).unsqueeze(0)
            f2t_stream.synchronize()

        with inf_stream_lock, torch.cuda.stream(inf_stream):
            output = inference(img)
            inf_stream.synchronize()

        with t2f_stream_lock, torch.cuda.stream(t2f_stream):
            res = tensor_to_frame(output, f[1].copy())
            t2f_stream.synchronize()

        return res

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], _inference), clip_src=[clip, new_clip]
    )
