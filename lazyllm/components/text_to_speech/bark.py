import os

import lazyllm
from lazyllm.thirdparty import torch
from lazyllm.thirdparty import transformers as tf
from lazyllm.components.formatter import encode_query_with_filepaths
from ..utils.downloader import ModelManager
import importlib.util
from .utils import sounds_to_files, TTSBase

class Bark(object):

    def __init__(self, base_path, source=None, trust_remote_code=True, save_path=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.trust_remote_code = trust_remote_code
        self.processor, self.bark = None, None
        self.init_flag = lazyllm.once_flag()
        self.device = 'cpu'
        self.save_path = save_path or os.path.join(lazyllm.config['temp_dir'], 'bark')
        if init:
            lazyllm.call_once(self.init_flag, self.load_bark)

    def load_bark(self):
        if importlib.util.find_spec("torch_npu") is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = tf.AutoProcessor.from_pretrained(self.base_path)
        self.processor.speaker_embeddings['repo_or_path'] = self.base_path
        self.bark = tf.BarkModel.from_pretrained(self.base_path, torch_dtype=torch.float16).to(self.device)

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_bark)
        if isinstance(string, str):
            query = string
            voice_preset = "v2/zh_speaker_9"
        elif isinstance(string, dict):
            query = string['inputs']
            voice_preset = string['voice_preset']
        else:
            raise TypeError(f"Not support input type:{type(string)}, requires str or dict.")
        inputs = self.processor(query, voice_preset=voice_preset).to(self.device)
        speech = self.bark.generate(**inputs).cpu().numpy().squeeze()
        file_path = sounds_to_files([speech], self.save_path, self.bark.generation_config.sample_rate)
        return encode_query_with_filepaths(files=file_path)

    @classmethod
    def rebuild(cls, base_path, init, save_path):
        return cls(base_path, init=init, save_path=save_path)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return Bark.rebuild, (self.base_path, init, self.save_path)

class BarkDeploy(TTSBase):
    keys_name_handle = {
        'inputs': 'inputs',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'voice_preset': None,
    }
    default_headers = {'Content-Type': 'application/json'}

    func = Bark
