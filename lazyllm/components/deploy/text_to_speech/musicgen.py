import os

import lazyllm
from lazyllm.thirdparty import transformers
from lazyllm.components.formatter import encode_query_with_filepaths
from ...utils.downloader import ModelManager
from .utils import sounds_to_files, TTSBase

class MusicGen(object):

    def __init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.model = None
        self.init_flag = lazyllm.once_flag()
        self._trust_remote_code = trust_remote_code
        self.save_path = save_path or os.path.join(lazyllm.config['temp_dir'], 'musicgen')
        if init:
            lazyllm.call_once(self.init_flag, self.load_tts)

    def load_tts(self):
        self.model = transformers.pipeline("text-to-speech", self.base_path, device=0)

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_tts)
        speech = self.model(string, forward_params={"do_sample": True})
        file_path = sounds_to_files([speech['audio'].flatten()], self.save_path, speech['sampling_rate'])
        return encode_query_with_filepaths(files=file_path)

    @classmethod
    def rebuild(cls, base_path, init, save_path):
        return cls(base_path, init=init, save_path=save_path)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return MusicGen.rebuild, (self.base_path, init, self.save_path)

class MusicGenDeploy(TTSBase):
    message_format = None
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}
    func = MusicGen
