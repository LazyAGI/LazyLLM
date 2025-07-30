from abc import abstractmethod
import os
import lazyllm
from lazyllm.components.deploy.text_to_speech.utils import sounds_to_base64_list
from lazyllm.components.formatter.formatterbase import encode_query_with_filepaths
from lazyllm.components.utils.downloader.model_downloader import ModelManager

class TTSInfer(object):
    """TTSInfer is an abstract base class for Text-to-Speech (TTS) inference models.

This class defines the common interface and logic for all TTS models, including model loading, lazy initialization, audio inference, and serialization support. Subclasses must implement the `load_model` and `_infer` methods to provide actual model behavior.

`__init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True, model_name=None)`
Constructor that prepares model path, lazy init flag, and output config.

Args:
    base_path: Path or model identifier to load the base TTS model.
    source: Optional model source. If not set, uses `lazyllm.config['model_source']`.
    save_path: Optional directory to store generated audio. Defaults to a temp path.
    init: Whether to load the model immediately upon construction. Defaults to False.
    trust_remote_code: Whether to trust and execute remote model code. Defaults to True.
    model_name: Optional model name to distinguish the output folder.

`load_model(self)`
Abstract method to load the actual TTS model. Must be implemented by subclasses.

`__call__(self, string)`
Convert input text to audio using the loaded model. Returns base64-encoded audio files.

Args:
    string: The input string to synthesize into speech.

Returns:
    A `lazyllm-query` string containing the base64-encoded audio file list.

`_infer(self, string)`
Abstract method to perform model inference on input text. Must return audio waveform and sample rate.

Args:
    string: Input text.

Returns:
    A tuple of (numpy.ndarray waveform, int sample_rate).

`rebuild(cls, base_path, init, save_path)`
Class method for rebuilding the object during unpickling or multiprocessing.

`__reduce__(self)`
Supports pickling and serialization for lazy loading scenarios.


Examples:
    >>> from lazyllm.components.deploy.text_to_speech.base import TTSInfer
    
    >>> class DummyTTSInfer(TTSInfer):
    ...     def load_model(self):
    ...         print("Loading dummy model...")
    ...     def _infer(self, string):
    ...         import numpy as np
    ...         return np.zeros(24000), 24000  # 1 second of silence
    
    >>> infer = DummyTTSInfer(base_path='dummy', init=True)
    >>> result = infer("Hello world!")
    >>> print(result)
    ... <lazyllm-query>{"query": "", "files": ["path/to/base64_audio.wav"]}
    """
    def __init__(self, base_path, source=None, save_path=None, init=False, trust_remote_code=True, model_name=None):
        source = lazyllm.config['model_source'] if not source else source
        self.base_path = ModelManager(source).download(base_path) or ''
        self.model = None
        self.init_flag = lazyllm.once_flag()
        self._trust_remote_code = trust_remote_code
        self.model_name = model_name or self.__class__.__name__.lower()
        self.save_path = save_path or os.path.join(lazyllm.config['temp_dir'], self.model_name)
        self.sample_rate = 24000
        if init:
            lazyllm.call_once(self.init_flag, self.load_model)

    @abstractmethod
    def load_model(self):
        pass

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_model)
        speech, sample_rate = self._infer(string)
        base64_list = sounds_to_base64_list(speech, sample_rate=sample_rate)
        return encode_query_with_filepaths(files=base64_list)

    @abstractmethod
    def _infer(self, string):
        pass

    @classmethod
    def rebuild(cls, base_path, init, save_path):
        return cls(base_path, init=init, save_path=save_path)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return self.__class__.rebuild, (self.base_path, init, self.save_path)
