import sys

sys.path.append(".")

import vapoursynth as vs
from vapoursynth import core

from ccrestoration import AutoModel, BaseModelInterface, ConfigType

model: BaseModelInterface = AutoModel.from_pretrained(ConfigType.RealESRGAN_AnimeJaNai_HD_V3_Compact_2x)

clip = core.bs.VideoSource(source="s.mp4")
clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
clip = model.inference_video(clip)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()
