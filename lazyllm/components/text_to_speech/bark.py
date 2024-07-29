import os
import json

import lazyllm
from lazyllm import LOG
from ..utils.downloader import ModelManager

class Bark(object):

    def __init__(self, base_sd, source=None, trust_remote_code=True, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_sd = ModelManager(source).download(base_sd)
        self.trust_remote_code = trust_remote_code
        self.processor, self.bark = None, None
        self.init_flag = lazyllm.once_flag()
        self.device = 'cpu'
        if init:
            lazyllm.call_once(self.init_flag, self.load_bark)

    def load_bark(self):
        import torch
        from transformers import AutoProcessor, BarkModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.base_sd)
        self.processor.speaker_embeddings['repo_or_path'] = self.base_sd
        self.bark = BarkModel.from_pretrained(self.base_sd,
                                              torch_dtype=torch.float16,
                                              attn_implementation="flash_attention_2").to(self.device)

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
        speech = self.bark.generate(**inputs) * 32767
        res = {'sounds': (
            self.bark.generation_config.sample_rate,
            speech.cpu().numpy().squeeze().tolist()
        )}
        return json.dumps(res)


class BarkDeploy(object):
    keys_name_handle = {
        'inputs': 'inputs',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'voice_preset': None,
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=None):
        self.launcher = launcher

    def __call__(self, finetuned_model=None, base_model=None):
        if not os.path.exists(finetuned_model) or \
            not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                    for _, _, filename in os.walk(finetuned_model) if filename):
            if not finetuned_model:
                LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                            f"base_model({base_model}) will be used")
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(func=Bark(finetuned_model), launcher=self.launcher)()
