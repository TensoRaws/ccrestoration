# ccrestoration

[![codecov](https://codecov.io/gh/TensoRaws/ccrestoration/graph/badge.svg?token=VK0BHDUXAI)](https://codecov.io/gh/TensoRaws/ccrestoration)
[![CI-test](https://github.com/TensoRaws/ccrestoration/actions/workflows/CI-test.yml/badge.svg)](https://github.com/TensoRaws/ccrestoration/actions/workflows/CI-test.yml)
[![Release-pypi](https://github.com/TensoRaws/ccrestoration/actions/workflows/Release-pypi.yml/badge.svg)](https://github.com/TensoRaws/ccrestoration/actions/workflows/Release-pypi.yml)
[![PyPI version](https://badge.fury.io/py/ccrestoration.svg)](https://badge.fury.io/py/ccrestoration)
![GitHub](https://img.shields.io/github/license/TensoRaws/ccrestoration)

an inference framework for image/video restoration with VapourSynth support, compatible with many community models

### Install

Make sure you have Python >= 3.9 and PyTorch installed

```bash
pip install ccrestoration
```

- Install VapourSynth (optional)

### Start

#### cv2

A simple example to use the sisr model (APISR) to process an image

```python
import cv2
import numpy as np

from ccrestoration import AutoModel, ConfigType, SRBaseModel

model: SRBaseModel = AutoModel.from_pretrained(ConfigType.RealESRGAN_APISR_RRDB_GAN_generator_2x)

img = cv2.imdecode(np.fromfile("test.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
img = model.inference_image(img)
cv2.imwrite("test_out.jpg", img)
```

#### VapourSynth

A simple example to use the vsr model (AnimeSR) to process a video

```python
import vapoursynth as vs
from vapoursynth import core

from ccrestoration import AutoModel, BaseModelInterface, ConfigType

model: BaseModelInterface = AutoModel.from_pretrained(
    pretrained_model_name=ConfigType.AnimeSR_v2_4x
)

clip = core.bs.VideoSource(source="s.mp4")
clip = core.resize.Bicubic(clip=clip, matrix_in_s="709", format=vs.RGBH)
clip = model.inference_video(clip)
clip = core.resize.Bicubic(clip=clip, matrix_s="709", format=vs.YUV420P16)
clip.set_output()
```

See more examples in the [example](./example) directory, ccrestoration can register custom configurations and models to extend the functionality

### Current Support

It still in development, the following models are supported:

- [Architecture](./ccrestoration/type/arch.py)

- [Model](./ccrestoration/type/model.py)

- [Weight(Config)](./ccrestoration/type/config.py)

### Reference

- [PyTorch](https://github.com/pytorch/pytorch)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [mmcv](https://github.com/open-mmlab/mmcv)
- [huggingface transformers](https://github.com/huggingface/transformers)
- [VapourSynth](https://www.vapoursynth.com/)
- [HolyWu's functions](https://github.com/HolyWu)

### License

This project is licensed under the MIT - see
the [LICENSE file](https://github.com/TensoRaws/ccrestoration/blob/main/LICENSE) for details.
