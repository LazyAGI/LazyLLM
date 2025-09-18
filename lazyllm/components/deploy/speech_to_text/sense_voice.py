import os
import importlib.util

import lazyllm
from lazyllm.components.utils.file_operate import _base64_to_file, _is_base64_with_mime
from lazyllm import LOG, LazyLLMLaunchersBase, is_valid_url, is_valid_path
from ..base import LazyLLMDeployBase
from ...utils.downloader import ModelManager
from lazyllm.thirdparty import funasr
from typing import Optional

supported_formats = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma')

class SenseVoice(object):
    """The SenseVoice class encapsulates FunASR-based speech-to-text model loading and invocation.  
It supports lazy initialization, automatic model downloading, and accepts string paths, URLs, or dicts containing audio.  

Args:
    base_path (str): Model path or identifier, downloaded locally via ModelManager.  
    source (Optional[str]): Model source, defaults to ``lazyllm.config['model_source']`` if not specified.  
    init (bool): Whether to load the model immediately during initialization. Defaults to ``False``.  

Attributes:
    base_path (str): Resolved local path of the downloaded model.  
    model (Optional[funasr.AutoModel]): Instance of the FunASR speech recognition model, available after initialization.  
    init_flag: A flag used for lazy loading, ensuring the model is loaded only once.  
"""
    def __init__(self, base_path, source=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.model = None
        self.init_flag = lazyllm.once_flag()
        if init:
            lazyllm.call_once(self.init_flag, self.load_stt)

    def load_stt(self):
        """Initializes and loads the FunASR speech-to-text model. Supports Huawei NPU acceleration if `torch_npu` is available.

Uses `fsmn-vad` for voice activity detection (VAD), supporting long utterances.
Maximum single segment duration is set to 30 seconds.
Default inference device is `cuda:0` (GPU).

The loaded model is assigned to `self.model` for subsequent audio transcription.

Note:
- If the environment has `torch_npu` installed, the method will import it to enable Ascend NPU acceleration.
"""
        if importlib.util.find_spec('torch_npu') is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401

        self.model = funasr.AutoModel(
            model=self.base_path,
            trust_remote_code=False,
            vad_model='fsmn-vad',
            vad_kwargs={'max_single_segment_time': 30000},
            device='cuda:0',
        )

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_stt)
        if isinstance(string, dict):
            if string['audio']:
                string = string['audio'][-1] if isinstance(string['audio'], list) else string['audio']
            else:
                string = string['inputs']
        assert isinstance(string, str)
        string = string.strip()
        try:
            string = _base64_to_file(string) if _is_base64_with_mime(string) else string
        except Exception as e:
            LOG.error(f'Error processing base64 encoding: {e}')
            return f'Error processing base64 encoding {e}'
        if not string.endswith(supported_formats):
            return f'Only {", ".join(supported_formats)} formats in the form of file paths or URLs are supported.'
        if not is_valid_path(string) and not is_valid_url(string):
            return f'This {string} is not a valid URL or file path. Please check.'
        res = self.model.generate(
            input=string,
            cache={},
            language='auto',  # 'zn', 'en', 'yue', 'ja', 'ko', 'nospeech'
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = funasr.utils.postprocess_utils.rich_transcription_postprocess(res[0]['text'])
        return text

    @classmethod
    def rebuild(cls, base_path, init):
        """Class method to reconstruct a `SenseVoice` instance during deserialization (e.g., with `cloudpickle`).  

Args:
    base_path (str): Path to the speech-to-text model.  
    init (bool): Whether to initialize and load the model upon instantiation.

**Returns:**

- SenseVoice: A new `SenseVoice` instance, used for serialization/multiprocessing compatibility.
"""
        return cls(base_path, init=init)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return SenseVoice.rebuild, (self.base_path, init)

class SenseVoiceDeploy(LazyLLMDeployBase):
    """SenseVoice Model Deployment Class. This class is used to deploy the SenseVoice model to a specified server for network invocation.

`__init__(self, launcher=None)`
Constructor, initializes the deployment class.

Args:
    launcher (Optional[LazyLLMLaunchersBase]): Launcher instance, defaults to None.
    log_path (Optional[str]): Log file path, defaults to None.
    trust_remote_code (bool): Whether to trust remote code, defaults to True.
    port (Optional[int]): Service port number, defaults to None.

Notes:
    - Input for infer: `str`. The audio path or link.
    - Return of infer: `str`. The recognized content.
    - Supported models: [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)


Examples:
    >>> import os
    >>> import lazyllm
    >>> from lazyllm import launchers, UrlModule
    >>> from lazyllm.components import SenseVoiceDeploy
    >>> deployer = SenseVoiceDeploy(launchers.remote())
    >>> url = deployer(base_model='SenseVoiceSmall')
    >>> model = UrlModule(url=url)
    >>> model('path/to/audio') # support format: .mp3, .wav
    ... xxxxxxxxxxxxxxxx
    """
    keys_name_handle = {
        'inputs': 'inputs',
        'audio': 'audio',
    }
    message_format = {
        'inputs': 'Who are you ?',
        'audio': None,
    }
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher: Optional[LazyLLMLaunchersBase] = None,
                 log_path: Optional[str] = None, trust_remote_code: bool = True, port: Optional[int] = None):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model) or \
            not any(file.endswith(('.pt', '.bin', '.safetensors'))
                    for _, _, filenames in os.walk(finetuned_model) for file in filenames):
            LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                        f'base_model({base_model}) will be used')
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(port=self._port, func=SenseVoice(finetuned_model), launcher=self._launcher,
                                          log_path=self._log_path, cls='sensevoice')()
