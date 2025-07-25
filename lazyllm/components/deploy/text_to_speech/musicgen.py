from lazyllm.thirdparty import transformers
from .utils import TTSBase
from .base import TTSInfer

class MusicGen(TTSInfer):

    def __init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True):
        super().__init__(base_path, source, save_path, init, trust_remote_code, 'musicgen')

    def load_model(self):
        self.model = transformers.pipeline("text-to-speech", self.base_path, device=0)

    def _infer(self, string):
        speech = self.model(string, forward_params={"do_sample": True})
        return [speech['audio'].flatten()], speech['sampling_rate']

class MusicGenDeploy(TTSBase):
    """MusicGen 模型部署类。该类用于将MusicGen模型部署到指定服务器上，以便可以通过网络进行调用。

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
    - 支持的模型为：[musicgen-small](https://huggingface.co/facebook/musicgen-small)


Examples:
    >>> from lazyllm import launchers, UrlModule
    >>> from lazyllm.components import MusicGenDeploy
    >>> deployer = MusicGenDeploy(launchers.remote())
    >>> url = deployer(base_model='musicgen-small')
    >>> model = UrlModule(url=url)
    >>> model('Symphony with flute as the main melody')
    ... <lazyllm-query>{"query": "", "files": ["path/to/musicgen/sound_xxx.wav"]}
    """
    message_format = None
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}
    func = MusicGen
