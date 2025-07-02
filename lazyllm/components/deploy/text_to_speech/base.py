from abc import abstractmethod
import os
import lazyllm
from lazyllm.components.deploy.text_to_speech.utils import sounds_to_base64_list
from lazyllm.components.formatter.formatterbase import encode_query_with_filepaths
from lazyllm.components.utils.downloader.model_downloader import ModelManager

class TTSInfer(object):
    def __init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True, model_name=None):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.model = None
        self.init_flag = lazyllm.once_flag()
        self._trust_remote_code = trust_remote_code
        self.model_name = model_name or self.__class__.__name__.lower()
        self.save_path = save_path or os.path.join(lazyllm.config['temp_dir'], self.model_name)
        self.sample_rate = 24000
        if init:
            lazyllm.call_once(self.init_flag, self.load_model)

    @abstractmethod
    def load_model(self):
        pass

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_model)
        speech, sample_rate = self._infer(string)
        base64_list = sounds_to_base64_list(speech, sample_rate=sample_rate)
        return encode_query_with_filepaths(files=base64_list)

    @abstractmethod
    def _infer(self, string):
        pass

    @classmethod
    def rebuild(cls, base_path, init, save_path):
        return cls(base_path, init=init, save_path=save_path)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return self.__class__.rebuild, (self.base_path, init, self.save_path)
