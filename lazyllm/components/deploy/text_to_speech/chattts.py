from lazyllm.thirdparty import torch, ChatTTS
from .utils import TTSBase
from .base import _TTSInfer


class _ChatTTSModule(_TTSInfer):

    def __init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True):
        self.seed = 1024
        super().__init__(base_path, source, save_path, init, trust_remote_code, 'chattts')

    def _load_model(self):
        self.model = ChatTTS.Chat()
        self.model.load(compile=False,
                        source='custom',
                        custom_path=self.base_path)
        self.spk = self._set_spk(self.seed)

    def _set_spk(self, seed):
        assert self.model
        torch.manual_seed(seed)
        rand_spk = self.model.sample_random_speaker()
        return rand_spk

    def _infer(self, string):
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
                self.spk = self._set_spk(self.seed)
            string['infercode']['spk_emb'] = self.spk
            params_infer_code = ChatTTS.Chat.InferCodeParams(**string['infercode'])
        else:
            raise TypeError(f'Not support input type:{type(string)}, requires str or dict.')
        speech = self.model.infer(query,
                                  params_refine_text=params_refine_text,
                                  params_infer_code=params_infer_code,
                                )
        return speech, self.sample_rate

class ChatTTSDeploy(TTSBase):
    """ChatTTS Model Deployment Class.

Keyword Args: 
    keys_name_handle (dict): A key mapping dictionary used to handle parameter name conversion between 
                            internal and external API interfaces. Defaults to `{'inputs': 'inputs'}`.

    message_format (dict): The request payload structure containing three main sections: 

        - `inputs` (str): The raw text content to be synthesized into speech. 

        - `refinetext` (dict): Text refinement and stylization parameters controlling speech expression: 

            * `prompt` (str): Voice style control tags, e.g., "[oral_2][laugh_0][break_6]" 

            * `top_P` (float): Nucleus sampling parameter for decoding strategy (default: 0.7) 

            * `top_K` (int): Top-K sampling parameter (default: 20) 

            * `temperature` (float): Sampling temperature controlling randomness (default: 0.7) 

            * `repetition_penalty` (float): Repetition penalty to avoid redundant generation (default: 1.0) 

            * `max_new_token` (int): Maximum number of tokens to generate (default: 384) 

            * `min_new_token` (int): Minimum number of tokens to generate (default: 0) 

            * `show_tqdm` (bool): Whether to display progress bar during generation (default: True) 

            * `ensure_non_empty` (bool): Ensure non-empty generation result (default: True) 

        - `infercode` (dict): Inference and encoding parameters affecting audio quality: 

            * `prompt` (str): Voice speed control tags, e.g., "[speed_5]" 

            * `spk_emb` (Optional): Speaker embedding vector for specifying voice characteristics (default: None) 

            * `temperature` (float): Sampling temperature for audio generation (default: 0.3) 

            * `repetition_penalty` (float): Repetition penalty coefficient (default: 1.05) 

            * `max_new_token` (int): Maximum number of tokens for audio generation (default: 2048) 



Examples:
    >>> from lazyllm import launchers, UrlModule
    >>> from lazyllm.components import ChatTTSDeploy
    >>> deployer = ChatTTSDeploy(launchers.remote())
    >>> url = deployer(base_model='ChatTTS')
    >>> model = UrlModule(url=url)
    >>> res = model('Hello World!')
    >>> print(res)
    ... <lazyllm-query>{"query": "", "files": ["path/to/chattts/sound_xxx.wav"]}
    """
    keys_name_handle = {
        'inputs': 'inputs',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'refinetext': {
            'prompt': '[oral_2][laugh_0][break_6]',
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
            'prompt': '[speed_5]',
            'spk_emb': None,
            'temperature': 0.3,
            'repetition_penalty': 1.05,
            'max_new_token': 2048,
        }

    }
    default_headers = {'Content-Type': 'application/json'}
    func = _ChatTTSModule
