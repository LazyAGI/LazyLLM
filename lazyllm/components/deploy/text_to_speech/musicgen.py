from lazyllm.thirdparty import transformers
from .utils import TTSBase
from .base import _TTSInfer

class _MusicGen(_TTSInfer):

    def __init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True):
        super().__init__(base_path, source, save_path, init, trust_remote_code, 'musicgen')

    def _load_model(self):
        self.model = transformers.pipeline('text-to-speech', self.base_path, device=0)

    def _infer(self, string):
        speech = self.model(string, forward_params={'do_sample': True})
        return [speech['audio'].flatten()], speech['sampling_rate']

class MusicGenDeploy(TTSBase):
    """MusicGen Model Deployment Class. This class is used to deploy the MusicGen model to a specified server for network invocation.

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
    - Supported models: [musicgen-small](https://huggingface.co/facebook/musicgen-small)


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
    func = _MusicGen
