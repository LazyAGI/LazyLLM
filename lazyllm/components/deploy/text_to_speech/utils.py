import os
import uuid
from lazyllm.thirdparty import scipy, numpy as np
from ...utils.file_operate import _delete_old_files
import lazyllm
from lazyllm import LOG, LazyLLMLaunchersBase
from typing import Optional
from ..base import LazyLLMDeployBase


def _sound_to_file(sound: 'np.array', file_path: str, sample_rate: int = 24000) -> str:
    scaled_audio = np.int16(sound / np.max(np.abs(sound)) * 32767)
    scipy.io.wavfile.write(file_path, sample_rate, scaled_audio)
    return [file_path]

def _sounds_to_files(sounds: list, directory: str, sample_rate: int = 24000) -> list:
    if not os.path.exists(directory):
        os.makedirs(directory)
    _delete_old_files(directory)
    unique_id = uuid.uuid4()
    path_list = []
    for i, sound in enumerate(sounds):
        file_path = os.path.join(directory, f'sound_{unique_id}_{i}.wav')
        _sound_to_file(sound, file_path, sample_rate)
        path_list.append(file_path)
    return path_list

class TTSBase(LazyLLMDeployBase):
    func = None

    def __init__(self, launcher: LazyLLMLaunchersBase = None,
                 log_path: Optional[str] = None, port: Optional[int] = None, **kw):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._port = port

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model) or \
            not any(file.endswith(('.bin', '.safetensors'))
                    for _, _, filenames in os.walk(finetuned_model) for file in filenames):
            LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                        f'base_model({base_model}) will be used')
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(port=self._port, func=self.__class__.func(finetuned_model),
                                          launcher=self._launcher, log_path=self._log_path, cls='tts')()
