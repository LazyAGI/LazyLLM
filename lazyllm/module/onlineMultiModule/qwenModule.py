import lazyllm
from http import HTTPStatus
import requests
from dashscope.audio.asr import Transcription
from dashscope.audio.tts_v2 import SpeechSynthesizer as SpeechSynthesizer_V2
from dashscope.audio.tts import SpeechSynthesizer
from dashscope.audio.qwen_tts import SpeechSynthesizer as SpeechSynthesizer_QwenTTS
from dashscope import ImageSynthesis
import dashscope
from typing import List
from lazyllm.components.utils.file_operate import bytes_to_file

from .onlineMultiModuleBase import OnlineMultiModuleBase

class QwenBaseModule(OnlineMultiModuleBase):
    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://dashscope.aliyuncs.com/api/v1',
                 base_websocket_url: str = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModuleBase.__init__(self, model_series="QWEN",
                                       model_name=model_name, return_trace=return_trace, **kwargs)
        dashscope.api_key = api_key or lazyllm.config['qwen_api_key']
        dashscope.base_http_api_url = base_url
        dashscope.base_websocket_api_url = base_websocket_url

class QwenSTTModule(QwenBaseModule):
    MODEL_NAME = "paraformer-v2"

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        QwenBaseModule.__init__(self, api_key=api_key,
                                model_name=model or lazyllm.config['qwen_stt_model_name'] or QwenSTTModule.MODEL_NAME,
                                return_trace=return_trace, **kwargs)

    def _forward(self, files: List[str] = [], **kwargs):
        assert any(file.startswith('http') for file in files), "QwenSTTModule only supports http file urls"
        call_params = {
            'model': self._model_name,
            'file_urls': files,
        }
        task_response = Transcription.async_call(**call_params)
        transcribe_response = Transcription.wait(task=task_response.output.task_id)
        if transcribe_response.status_code == HTTPStatus.OK:
            output = transcribe_response.output
            return output, None
        else:
            lazyllm.LOG.error(f"failed to transcribe: {transcribe_response.output}")
            raise Exception(f"failed to transcribe: {transcribe_response.output.message}")

class QwenTextToImageModule(QwenBaseModule):
    MODEL_NAME = "wanx2.1-t2i-turbo"

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        QwenBaseModule.__init__(self, api_key=api_key,
                                model_name=model or lazyllm.config['qwen_text2image_model_name']
                                or QwenTextToImageModule.MODEL_NAME, return_trace=return_trace, **kwargs)
        self._format_output_files = bytes_to_file

    def _forward(self, input: str = None, negative_prompt: str = None, n: int = 1, prompt_extend: bool = True,
                 size: str = '1024*1024', seed: int = None, **kwargs):
        call_params = {
            'model': self._model_name,
            'prompt': input,
            'negative_prompt': negative_prompt,
            'n': n,
            'prompt_extend': prompt_extend,
            'size': size
        }
        if seed:
            call_params['seed'] = seed
        task_response = ImageSynthesis.async_call(**call_params)
        response = ImageSynthesis.wait(task=task_response.output.task_id)
        if response.status_code == HTTPStatus.OK:
            return None, [requests.get(result.url).content for result in response.output.results]
        else:
            lazyllm.LOG.error(f"failed to generate image: {response.output}")
            raise Exception(f"failed to generate image: {response.output.message}")

def synthesize_qwentts(input: str, model_name: str = None, voice: str = None):
    response = SpeechSynthesizer_QwenTTS.call(
        model=model_name,
        text=input,
        voice=voice,
    )
    if response.status_code == HTTPStatus.OK:
        return requests.get(response.output['audio']['url']).content
    else:
        lazyllm.LOG.error(f"failed to synthesize: {response}")
        raise Exception(f"failed to synthesize: {response.message}")

def synthesize(input: str, model_name: str = None, voice: str = None):
    model_name = model_name + '-' + voice
    response = SpeechSynthesizer.call(model=model_name, text=input)
    if response.get_response().status_code == HTTPStatus.OK:
        return response.get_audio_data()
    else:
        lazyllm.LOG.error(f"failed to synthesize: {response.get_response()}")
        raise Exception(f"failed to synthesize: {response.get_response().message}")

def synthesize_v2(input: str, model_name: str = None, voice: str = None):
    synthesizer = SpeechSynthesizer_V2(model=model_name, voice=voice)
    audio = synthesizer.call(input)
    if synthesizer.last_response['header']['event'] == 'task-finished':
        return audio
    else:
        lazyllm.LOG.error(f"failed to synthesize: {synthesizer.last_response}")
        raise Exception(f"failed to synthesize: {synthesizer.last_response['header']['error_message']}")

class QwenTTSModule(QwenBaseModule):
    MODEL_NAME = "cosyvoice-v2"
    SYNTHESIZERS = {
        "cosyvoice-v2": (synthesize_v2, 'longxiaochun_v2'),
        "cosyvoice-v1": (synthesize_v2, 'longxiaochun'),
        "sambert": (synthesize, 'zhinan-v1'),
        "qwen-tts": (synthesize_qwentts, 'Cherry'),
        "qwen-tts-latest": (synthesize_qwentts, 'Cherry')
    }

    def __init__(self, model: str = None, voice: str = None, api_key: str = None,
                 return_trace: bool = False, **kwargs):
        QwenBaseModule.__init__(self, api_key=api_key,
                                model_name=model or lazyllm.config['qwen_tts_model_name'] or QwenTTSModule.MODEL_NAME,
                                return_trace=return_trace, **kwargs)
        if self._model_name not in self.SYNTHESIZERS:
            raise ValueError(f"unsupported model: {self._model_name}. "
                             f"supported models: {QwenTTSModule.SYNTHESIZERS.keys()}")
        synthesizer_func, default_voice = QwenTTSModule.SYNTHESIZERS[self._model_name]
        self._synthesizer_func = synthesizer_func
        self._voice = voice or default_voice
        self._format_output_files = bytes_to_file

    def _forward(self, input: str = None, **kwargs):
        return None, self._synthesizer_func(input, self._model_name, self._voice)
