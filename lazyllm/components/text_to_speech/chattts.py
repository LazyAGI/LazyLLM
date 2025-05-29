import os

import lazyllm
from lazyllm.thirdparty import torch, ChatTTS
from lazyllm.components.formatter import encode_query_with_filepaths
from ..utils.downloader import ModelManager
from .utils import sounds_to_files, TTSBase


class ChatTTSModule(object):

    def __init__(self, base_path, source=None, save_path=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.model, self.spk = None, None
        self.init_flag = lazyllm.once_flag()
        self.device = 'cpu'
        self.seed = 1024
        self.save_path = save_path or os.path.join(lazyllm.config['temp_dir'], 'chattts')
        if init:
            lazyllm.call_once(self.init_flag, self.load_tts)

    def load_tts(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChatTTS.Chat()
        self.model.load(compile=False,
                        source="custom",
                        custom_path=self.base_path,
                        device=self.device)
        self.spk = self.set_spk(self.seed)

    def set_spk(self, seed):
        assert self.model
        torch.manual_seed(seed)
        rand_spk = self.model.sample_random_speaker()
        return rand_spk

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_tts)
        if isinstance(string, str):
            query = string
            params_refine_text = ChatTTS.Chat.RefineTextParams()
            params_infer_code = ChatTTS.Chat.InferCodeParams(spk_emb=self.spk)
        elif isinstance(string, dict):
            query = string['inputs']
            params_refine_text = ChatTTS.Chat.RefineTextParams(**string['refinetext'])
            spk_seed = string['infercode']['spk_emb']
            spk_seed = int(spk_seed) if spk_seed else spk_seed
            if isinstance(spk_seed, int) and self.seed != spk_seed:
                self.seed = spk_seed
                self.spk = self.set_spk(self.seed)
            string['infercode']['spk_emb'] = self.spk
            params_infer_code = ChatTTS.Chat.InferCodeParams(**string['infercode'])
        else:
            raise TypeError(f"Not support input type:{type(string)}, requires str or dict.")
        speech = self.model.infer(query,
                                  params_refine_text=params_refine_text,
                                  params_infer_code=params_infer_code,
                                )
        file_path = sounds_to_files(speech, self.save_path)
        return encode_query_with_filepaths(files=file_path)

    @classmethod
    def rebuild(cls, base_path, init, save_path):
        return cls(base_path, init=init, save_path=save_path)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return ChatTTSModule.rebuild, (self.base_path, init, self.save_path)

class ChatTTSDeploy(TTSBase):
    keys_name_handle = {
        'inputs': 'inputs',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'refinetext': {
            'prompt': "[oral_2][laugh_0][break_6]",
            'top_P': 0.7,
            'top_K': 20,
            'temperature': 0.7,
            'repetition_penalty': 1.0,
            'max_new_token': 384,
            'min_new_token': 0,
            'show_tqdm': True,
            'ensure_non_empty': True,
        },
        'infercode': {
            'prompt': "[speed_5]",
            'spk_emb': None,
            'temperature': 0.3,
            'repetition_penalty': 1.05,
            'max_new_token': 2048,
        }

    }
    default_headers = {'Content-Type': 'application/json'}
    func = ChatTTSModule
