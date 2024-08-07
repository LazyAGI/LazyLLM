import os
import json
import base64
from PIL import Image
import numpy as np
from io import BytesIO

import lazyllm
from lazyllm import LOG
from ..utils.downloader import ModelManager


class StableDiffusion3(object):
    def __init__(self, base_sd, source=None, embed_batch_size=30, trust_remote_code=True, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_sd = ModelManager(source).download(base_sd)
        self.embed_batch_size = embed_batch_size
        self.trust_remote_code = trust_remote_code
        self.sd = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_sd)

    def load_sd(self):
        import torch
        from diffusers import StableDiffusion3Pipeline
        self.sd = StableDiffusion3Pipeline.from_pretrained(self.base_sd, torch_dtype=torch.float16).to("cuda")

    @staticmethod
    def image_to_base64(image):
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            raise ValueError("Unsupported image type")
        return img_str

    @staticmethod
    def images_to_base64(images):
        return [StableDiffusion3.image_to_base64(img) for img in images]

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_sd)
        imgs = self.sd(
            string,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
            max_sequence_length=512,
        ).images
        img_base64_list = StableDiffusion3.images_to_base64(imgs)
        res = {"images_base64": img_base64_list}
        return json.dumps(res)

    @classmethod
    def rebuild(cls, base_sd, embed_batch_size, init):
        return cls(base_sd, embed_batch_size=embed_batch_size, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return StableDiffusion3.rebuild, (self.base_sd, self.embed_batch_size, init)

class StableDiffusionDeploy(object):
    message_format = None
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=None):
        self.launcher = launcher

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin', '.safetensors')
                    for _, _, filename in os.walk(finetuned_model) if filename):
            LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                        f"base_model({base_model}) will be used")
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(func=StableDiffusion3(finetuned_model), launcher=self.launcher)()
