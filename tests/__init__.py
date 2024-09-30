# from vapoursynth import core
#
# from ccrestoration import AutoModel
#
# # AutoModel.list_models()
# # AutoModel.register(model_name, model_class_object)
# clip = core.bs.VideoSource(source="s.mkv")
# clip = AutoModel.from_pretrained("REAL-ESRGAN/RealESRGANx4plus/x4").InferenceVideo(clip)  # download only once
# # clip = AutoModel.from_pretrained_url(url, key, model, **args).InferenceVideo(clip)
# # clip = AutoModel.from_pretrained_path(path, model, **args).InferenceVideo(clip)
# clip.set_output()


# 设计 autoconfig 模块，用于自动配置不同模型，如animejanai 对应 {arch: rrdb, scale: 4, noise: 0.1}
# config = AutoConfig.from_pretrained("animejanai")
# AutoModel.from_config(config).InferenceVideo(clip)
