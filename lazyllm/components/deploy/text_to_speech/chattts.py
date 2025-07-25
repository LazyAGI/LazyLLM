from lazyllm.thirdparty import torch, ChatTTS
from .utils import TTSBase
from .base import TTSInfer


class ChatTTSModule(TTSInfer):

    def __init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True):
        self.seed = 1024
        super().__init__(base_path, source, save_path, init, trust_remote_code, 'chattts')

    def load_model(self):
        self.model = ChatTTS.Chat()
        self.model.load(compile=False,
                        source="custom",
                        custom_path=self.base_path)
        self.spk = self.set_spk(self.seed)

    def set_spk(self, seed):
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
                self.spk = self.set_spk(self.seed)
            string['infercode']['spk_emb'] = self.spk
            params_infer_code = ChatTTS.Chat.InferCodeParams(**string['infercode'])
        else:
            raise TypeError(f"Not support input type:{type(string)}, requires str or dict.")
        speech = self.model.infer(query,
                                  params_refine_text=params_refine_text,
                                  params_infer_code=params_infer_code,
                                )
        return speech, self.sample_rate

class ChatTTSDeploy(TTSBase):
    """ChatTTS 模型部署类。该类用于将ChatTTS模型部署到指定服务器上，以便可以通过网络进行调用。

`__init__(self, launcher=None)`
构造函数，初始化部署类。

Args:
    launcher(lazyllm.launcher): 用于启动远程服务的启动器实例。

`__call__(self, finetuned_model=None, base_model=None)`
部署模型，并返回远程服务地址。

Args: 
    finetuned_model (str): 如果提供，则使用该模型进行部署；如果未提供或路径无效，则使用 `base_model`。
    base_model (str): 默认模型，如果 `finetuned_model` 无效，则使用该模型进行部署。
    返回值 (str): 远程服务的URL地址。

Notes:
    - 推理的输入：字符串。待生成音频的对应文字。
    - 推理的返回值：从生成的文件路径编码的字符串， 编码标志以 "<lazyllm-query>"开头，后面跟序列化后的字典, 字典中 `files`键存放了一个列表，元素是生成的音频文件路径。
    - 支持的模型为：[ChatTTS](https://huggingface.co/2Noise/ChatTTS)


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
