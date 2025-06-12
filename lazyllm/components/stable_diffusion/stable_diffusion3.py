import os
import base64
import uuid
import importlib.util
from lazyllm.thirdparty import PIL, torch, diffusers
from lazyllm.thirdparty import numpy as np
from io import BytesIO

import lazyllm
from lazyllm import LOG
from lazyllm.components.formatter import encode_query_with_filepaths
from ..utils.downloader import ModelManager
from ..utils.file_operate import delete_old_files


class StableDiffusion3(object):

    _load_registry = {}
    _call_registry = {}

    @classmethod
    def register_loader(cls, model_type):
        def decorator(loader_func):
            cls._load_registry[model_type] = loader_func
            return loader_func
        return decorator

    @classmethod
    def register_caller(cls, model_type):
        def decorator(caller_func):
            cls._call_registry[model_type] = caller_func
            return caller_func
        return decorator

    def __init__(self, base_sd, source=None, embed_batch_size=30, trust_remote_code=True, save_path=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_sd = ModelManager(source).download(base_sd) or ''
        self.embed_batch_size = embed_batch_size
        self.trust_remote_code = trust_remote_code
        self.paintor = None
        self.init_flag = lazyllm.once_flag()
        self.save_path = save_path or os.path.join(lazyllm.config['temp_dir'], 'sd3')
        if init:
            lazyllm.call_once(self.init_flag, self.load_sd)

    def load_sd(self):
        if importlib.util.find_spec("torch_npu") is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401

        for model_type, loader in self._load_registry.items():
            if model_type in self.base_sd.lower():
                loader(self)
                return

        self.paintor = diffusers.StableDiffusion3Pipeline.from_pretrained(
            self.base_sd, torch_dtype=torch.float16).to("cuda")

    @staticmethod
    def image_to_base64(image):
        if isinstance(image, PIL.Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            raise ValueError("Unsupported image type")
        return img_str

    @staticmethod
    def images_to_base64(images):
        return [StableDiffusion3.image_to_base64(img) for img in images]

    @staticmethod
    def image_to_file(image, file_path):
        if isinstance(image, PIL.Image.Image):
            image.save(file_path, format="PNG")
        elif isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
            image.save(file_path, format="PNG")
        else:
            raise ValueError("Unsupported image type")

    @staticmethod
    def images_to_files(images, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        delete_old_files(directory)
        unique_id = uuid.uuid4()
        path_list = []
        for i, img in enumerate(images):
            file_path = os.path.join(directory, f'image_{unique_id}_{i}.png')
            StableDiffusion3.image_to_file(img, file_path)
            path_list.append(file_path)
        return path_list

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_sd)

        for model_type, caller in self._call_registry.items():
            if model_type in self.base_sd.lower():
                return caller(self, string)

        imgs = self.paintor(
            string,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
            max_sequence_length=512,
        ).images
        img_path_list = self.images_to_files(imgs, self.save_path)
        return encode_query_with_filepaths(files=img_path_list)

    @classmethod
    def rebuild(cls, base_sd, embed_batch_size, init, save_path):
        return cls(base_sd, embed_batch_size=embed_batch_size, init=init, save_path=save_path)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return StableDiffusion3.rebuild, (self.base_sd, self.embed_batch_size, init, self.save_path)

@StableDiffusion3.register_loader('flux')
def load_flux(model):
    import torch
    from diffusers import FluxPipeline
    model.paintor = FluxPipeline.from_pretrained(
        model.base_sd, torch_dtype=torch.bfloat16).to("cuda")

@StableDiffusion3.register_caller('flux')
def call_flux(model, prompt):
    imgs = model.paintor(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        max_sequence_length=512,
    ).images
    img_path_list = model.images_to_files(imgs, model.save_path)
    return encode_query_with_filepaths(files=img_path_list)

@StableDiffusion3.register_loader('cogview')
def load_cogview(model):
    import torch
    from diffusers import CogView4Pipeline
    model.paintor = CogView4Pipeline.from_pretrained(
        model.base_sd, torch_dtype=torch.bfloat16).to("cuda")

@StableDiffusion3.register_caller('cogview')
def call_cogview(model, prompt):
    imgs = model.paintor(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        num_images_per_prompt=1,
    ).images
    img_path_list = model.images_to_files(imgs, model.save_path)
    return encode_query_with_filepaths(files=img_path_list)

@StableDiffusion3.register_loader('wan')
def load_wan(model):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    vae = AutoencoderKLWan.from_pretrained(
        model.base_sd, subfolder="vae", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=3.0
    )
    model.paintor = WanPipeline.from_pretrained(
        model.base_sd, vae=vae, torch_dtype=torch.bfloat16)
    model.paintor.scheduler = scheduler
    model.paintor.to("cuda")

@StableDiffusion3.register_caller('wan')
def call_wan(model, prompt):
    from diffusers.utils import export_to_video
    videos = model.paintor(
        prompt,
        negative_prompt=(
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, three legs, "
            "many people in the background, walking backwards"),
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
    ).frames
    unique_id = uuid.uuid4()
    if not os.path.exists(model.save_path):
        os.makedirs(model.save_path)
    vid_path_list = []
    for i, vid in enumerate(videos):
        file_path = os.path.join(model.save_path, f'video_{unique_id}_{i}.mp4')
        export_to_video(vid, file_path, fps=16)
        vid_path_list.append(file_path)
    return encode_query_with_filepaths(files=vid_path_list)


class StableDiffusionDeploy(object):
    message_format = None
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=None, log_path=None):
        self._launcher = launcher
        self._log_path = log_path

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model) or \
            not any(file.endswith(('.bin', '.safetensors'))
                    for _, _, filenames in os.walk(finetuned_model) for file in filenames):
            LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                        f"base_model({base_model}) will be used")
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(func=StableDiffusion3(finetuned_model), launcher=self._launcher,
                                          log_path=self._log_path, cls='stable_diffusion')()
