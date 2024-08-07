import os
import json

import lazyllm
from lazyllm import LOG
from lazyllm.thirdparty import numpy as np
from ..utils.downloader import ModelManager

class MusicGen(object):

    def __init__(self, base_path, source=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path)
        self.model = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_tts)

    def load_tts(self):
        from transformers import pipeline
        self.model = pipeline("text-to-speech", self.base_path, device=0)

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_tts)
        speech = self.model(string, forward_params={"do_sample": True})
        speech['audio'] = (speech['audio'].flatten() * 32767).astype(np.int16).tolist()
        res = {'sounds': (speech['sampling_rate'], speech['audio'])}
        return json.dumps(res)

    @classmethod
    def rebuild(cls, base_path, init):
        return cls(base_path, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return MusicGen.rebuild, (self.base_path, init)

class MusicGenDeploy(object):
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
        return lazyllm.deploy.RelayServer(func=MusicGen(finetuned_model), launcher=self.launcher)()
