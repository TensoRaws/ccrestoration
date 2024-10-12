"""
All the vs functions are referenced from HolyWu's repository: https://github.com/HolyWu
Thanks to HolyWu for his great work on super-resolution.
"""


from ccrestoration.vs.sr import inference_sr  # noqa
from ccrestoration.vs.convert import tensor_to_frame, frame_to_tensor  # noqa
from ccrestoration.vs.vsr import inference_vsr, inference_vsr_one_frame_out  # noqa
