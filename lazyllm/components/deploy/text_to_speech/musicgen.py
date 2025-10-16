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
    message_format = None
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}
    func = _MusicGen
