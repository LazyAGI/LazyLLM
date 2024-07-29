import os
import json
import numpy as np

import lazyllm
from lazyllm import LOG
from ..utils.downloader import ModelManager

class Bark(object):

    def __init__(self, base_sd, source=None, trust_remote_code=True, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_sd = ModelManager(source).download(base_sd)
        self.trust_remote_code = trust_remote_code
        self.bark = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_bark)

    def load_bark(self):
        from transformers import pipeline
        self.bark = pipeline("text-to-speech", "/home/mnt/lazyllm/models/bark", device=0, voice_preset="v2/en_speaker_3")

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_bark)
        # print("IIIII: ", string)
        speech = self.bark(string, forward_params={"do_sample": True})
        speech['audio'] = (speech['audio'].flatten() * 32767).astype(np.int16).tolist()
        res = {'sounds': (speech['sampling_rate'], speech['audio'])}
        # print("OOOO: ", res)
        return json.dumps(res)


class BarkDeploy(object):
    message_format = None
    keys_name_handle = None
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
