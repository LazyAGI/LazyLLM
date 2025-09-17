from abc import abstractmethod
import os
import base64
from io import BytesIO
from lazyllm.thirdparty import scipy, numpy as np
import lazyllm
from lazyllm.components.formatter.formatterbase import encode_query_with_filepaths
from lazyllm.components.utils.downloader.model_downloader import ModelManager


def _sound_to_base64(sound: 'np.array', mime_type: str = 'audio/wav', sample_rate: int = 24000) -> str:
    scaled_audio = np.int16(sound / np.max(np.abs(sound)) * 32767)
    buffer = BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, scaled_audio)
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:{mime_type};base64,{base64_str}'

def _sounds_to_base64_list(sounds: list, mime_type: str = 'audio/wav', sample_rate: int = 24000) -> list:
    base64_list = []
    for sound in sounds:
        base64_str = _sound_to_base64(sound, mime_type, sample_rate)
        base64_list.append(base64_str)
    return base64_list

class _TTSInfer(object):
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
            lazyllm.call_once(self.init_flag, self._load_model)

    @abstractmethod
    def _load_model(self):
        pass

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self._load_model)
        speech, sample_rate = self._infer(string)
        base64_list = _sounds_to_base64_list(speech, sample_rate=sample_rate)
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
