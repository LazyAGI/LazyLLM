import os
import importlib.util
from urllib.parse import urlparse

import lazyllm
from lazyllm import LOG
from ..utils.downloader import ModelManager
from lazyllm.thirdparty import funasr


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def is_valid_path(path):
    return os.path.isfile(path)

class SenseVoice(object):
    def __init__(self, base_path, source=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.model = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_stt)

    def load_stt(self):
        if importlib.util.find_spec("torch_npu") is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401

        self.model = funasr.AutoModel(
            model=self.base_path,
            trust_remote_code=False,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_stt)
        if isinstance(string, dict):
            if string['audio']:
                string = string['audio'][-1] if isinstance(string['audio'], list) else string['audio']
            else:
                string = string['inputs']
        assert isinstance(string, str)
        string = string.strip()
        if not string.endswith(('.mp3', '.wav')):
            return "Only '.mp3' and '.wav' formats in the form of file paths or URLs are supported."
        if not is_valid_path(string) and not is_valid_url(string):
            return f"This {string} is not a valid URL or file path. Please check."
        res = self.model.generate(
            input=string,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = funasr.utils.postprocess_utils.rich_transcription_postprocess(res[0]["text"])
        return text

    @classmethod
    def rebuild(cls, base_path, init):
        return cls(base_path, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return SenseVoice.rebuild, (self.base_path, init)

class SenseVoiceDeploy(object):
    keys_name_handle = {
        'inputs': 'inputs',
        'audio': 'audio',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'audio': None,
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=None, log_path=None):
        self._launcher = launcher
        self._log_path = log_path

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model) or \
            not any(file.endswith(('.pt', '.bin', '.safetensors'))
                    for _, _, filenames in os.walk(finetuned_model) for file in filenames):
            LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                        f"base_model({base_model}) will be used")
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(func=SenseVoice(finetuned_model), launcher=self._launcher,
                                          log_path=self._log_path, cls='sensevoice')()
