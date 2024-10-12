import sys

sys.path.append(".")
sys.path.append("..")

import vapoursynth as vs
from vapoursynth import core

from ccrestoration import AutoModel, BaseModelInterface, ConfigType

# --- sisr, use fp16 to inference (vs.RGBH)

model: BaseModelInterface = AutoModel.from_pretrained(
    pretrained_model_name=ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x
)

clip = core.bs.VideoSource(source="s.mp4")
clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
clip = model.inference_video(clip)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()

# ---  use fp32 to inference (vs.RGBS)

# model: BaseModelInterface = AutoModel.from_pretrained(
#     pretrained_model_name=ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x,
#     fp16=False,
# )
#
# clip = core.bs.VideoSource(source="s.mp4")
# clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBS)

# --- vsr

# model: BaseModelInterface = AutoModel.from_pretrained(
#     pretrained_model_name=ConfigType.AnimeSR_v2_4x
# )
#
# clip = core.bs.VideoSource(source="s.mp4")
# clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
# clip = model.inference_video(clip)
# clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
# clip.set_output()
