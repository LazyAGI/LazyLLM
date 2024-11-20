import os
import uuid
from lazyllm.thirdparty import scipy, numpy as np
from ..utils.file_operate import delete_old_files


def sound_to_file(sound: 'np.array', file_path: str, sample_rate: int = 24000) -> str:
    scaled_audio = np.int16(sound / np.max(np.abs(sound)) * 32767)
    scipy.io.wavfile.write(file_path, sample_rate, scaled_audio)
    return [file_path]

def sounds_to_files(sounds: list, directory: str, sample_rate: int = 24000) -> list:
    if not os.path.exists(directory):
        os.makedirs(directory)
    delete_old_files(directory)
    unique_id = uuid.uuid4()
    path_list = []
    for i, sound in enumerate(sounds):
        file_path = os.path.join(directory, f'sound_{unique_id}_{i}.wav')
        sound_to_file(sound, file_path, sample_rate)
        path_list.append(file_path)
    return path_list
