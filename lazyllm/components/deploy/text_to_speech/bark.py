from lazyllm.thirdparty import torch
from lazyllm.thirdparty import transformers as tf
import importlib.util
from lazyllm.components.deploy.text_to_speech.utils import TTSBase
from lazyllm.components.deploy.text_to_speech.base import TTSInfer

class Bark(TTSInfer):

    def __init__(self, base_path, source=None, trust_remote_code=True, save_path=None, init=False):
        super().__init__(base_path, source, save_path, init, trust_remote_code, 'bark')

    def load_model(self):
        if importlib.util.find_spec("torch_npu") is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = tf.AutoProcessor.from_pretrained(self.base_path)
        self.processor.speaker_embeddings['repo_or_path'] = self.base_path
        self.model = tf.BarkModel.from_pretrained(self.base_path, torch_dtype=torch.float16).to(self.device)

    def _infer(self, string):
        if isinstance(string, str):
            query = string
            voice_preset = "v2/zh_speaker_9"
        elif isinstance(string, dict):
            query = string['inputs']
            voice_preset = string['voice_preset']
        else:
            raise TypeError(f"Not support input type:{type(string)}, requires str or dict.")
        inputs = self.processor(query, voice_preset=voice_preset).to(self.device)
        speech = self.model.generate(**inputs).cpu().numpy().squeeze()
        return [speech], self.model.generation_config.sample_rate

class BarkDeploy(TTSBase):
    """Bark 模型部署类。该类用于将Bark模型部署到指定服务器上，以便可以通过网络进行调用。

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
    - 支持的模型为：[bark](https://huggingface.co/suno/bark)


Examples:
    >>> from lazyllm import launchers, UrlModule
    >>> from lazyllm.components import BarkDeploy
    >>> deployer = BarkDeploy(launchers.remote())
    >>> url = deployer(base_model='bark')
    >>> model = UrlModule(url=url)
    >>> res = model('Hello World!')
    >>> print(res)
    ... <lazyllm-query>{"query": "", "files": ["path/to/bark/sound_xxx.wav"]}
    """
    keys_name_handle = {
        'inputs': 'inputs',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'voice_preset': None,
    }
    default_headers = {'Content-Type': 'application/json'}

    func = Bark
