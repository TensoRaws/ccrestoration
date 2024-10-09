from typing import Any, Callable, Union

import torch
import vapoursynth as vs

from ccrestoration.vs.convert import frame_to_tensor, tensor_to_frame


def inference_sr(
    inference: Callable[[torch.Tensor], torch.Tensor],
    clip: vs.VideoNode,
    scale: Union[float, int, Any],
    device: torch.device,
    _frame_to_tensor: Callable[[vs.VideoFrame, torch.device], torch.Tensor] = frame_to_tensor,
    _tensor_to_frame: Callable[[torch.Tensor, vs.VideoFrame], vs.VideoFrame] = tensor_to_frame,
) -> vs.VideoNode:
    """
    Inference the video with the model, the clip should be a vapoursynth clip

    :param inference: The inference function
    :param clip: vs.VideoNode
    :param scale: The scale factor
    :param device: The device
    :param _frame_to_tensor: The function to convert the frame to tensor
    :param _tensor_to_frame: The function to convert the tensor to frame
    :return:
    """

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("Only vs.RGBH and vs.RGBS formats are supported")

    def _inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        img = _frame_to_tensor(f[0], device).unsqueeze(0)

        output = inference(img)

        return _tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], _inference), clip_src=[clip, new_clip]
    )
