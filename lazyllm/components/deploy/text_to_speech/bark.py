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
    keys_name_handle = {
        'inputs': 'inputs',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'voice_preset': None,
    }
    default_headers = {'Content-Type': 'application/json'}

    func = _Bark
