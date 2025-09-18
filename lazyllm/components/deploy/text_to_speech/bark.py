from lazyllm.thirdparty import torch
from lazyllm.thirdparty import transformers as tf
import importlib.util
from lazyllm.components.deploy.text_to_speech.utils import TTSBase
from lazyllm.components.deploy.text_to_speech.base import _TTSInfer

class _Bark(_TTSInfer):

    def __init__(self, base_path, source=None, trust_remote_code=True, save_path=None, init=False):
        super().__init__(base_path, source, save_path, init, trust_remote_code, 'bark')

    def _load_model(self):
        if importlib.util.find_spec('torch_npu') is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = tf.AutoProcessor.from_pretrained(self.base_path)
        self.processor.speaker_embeddings['repo_or_path'] = self.base_path
        self.model = tf.BarkModel.from_pretrained(self.base_path, torch_dtype=torch.float16).to(self.device)

    def _infer(self, string):
        if isinstance(string, str):
            query = string
            voice_preset = 'v2/zh_speaker_9'
        elif isinstance(string, dict):
            query = string['inputs']
            voice_preset = string['voice_preset']
        else:
            raise TypeError(f'Not support input type:{type(string)}, requires str or dict.')
        inputs = self.processor(query, voice_preset=voice_preset).to(self.device)
        speech = self.model.generate(**inputs).cpu().numpy().squeeze()
        return [speech], self.model.generation_config.sample_rate

class BarkDeploy(TTSBase):
    """Bark Model Deployment Class. This class is used to deploy the Bark model to a specified server for network invocation.

`__init__(self, launcher=None)`
Constructor, initializes the deployment class.

Args:
    launcher (lazyllm.launcher): An instance of the launcher used to start the remote service.

`__call__(self, finetuned_model=None, base_model=None)`
Deploys the model and returns the remote service address.

Args:
    finetuned_model (str): If provided, this model will be used for deployment; if not provided or the path is invalid, `base_model` will be used.
    base_model (str): The default model, which will be used for deployment if `finetuned_model` is invalid.
    Return (str): The URL address of the remote service.

Notes:
    - Input for infer: `str`.  The text corresponding to the audio to be generated.
    - Return of infer: The string encoded from the generated file paths, starting with the encoding flag "<lazyllm-query>", followed by the serialized dictionary. The key `files` in the dictionary stores a list, with elements being the paths of the generated audio files.
    - Supported models: [bark](https://huggingface.co/suno/bark)


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

    func = _Bark
