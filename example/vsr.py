import cv2
import numpy as np

from ccrestoration import AutoModel, ConfigType
from ccrestoration.model import VSRBaseModel

img = cv2.imdecode(np.fromfile("../assets/test.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
imgList = [img, img, img]

model: VSRBaseModel = AutoModel.from_pretrained(ConfigType.AnimeSR_v2_4x, fp16=False)

imgOutList = model.inference_image_list(imgList)

for i, v in enumerate(imgOutList):
    cv2.imwrite(f"../assets/test_vsr_{i}_out.jpg", v)
