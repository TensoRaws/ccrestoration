import sys

sys.path.append(".")

import vapoursynth as vs
from vapoursynth import core

from ccrestoration.core.config import CONFIG_REGISTRY

for k, v in CONFIG_REGISTRY:
    print(k, v)

clip = core.bs.VideoSource(source="s.mp4")
clip = core.resize.Bicubic(clip=clip, format=vs.RGBH)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()
