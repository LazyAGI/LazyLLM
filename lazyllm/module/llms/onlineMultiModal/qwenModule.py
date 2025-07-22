import lazyllm
from http import HTTPStatus
import requests
from lazyllm.thirdparty import dashscope
from typing import List
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.components.formatter import encode_query_with_filepaths
from .onlineMultiModalBase import OnlineMultiModalBase
import json
import re


class QwenModule(OnlineMultiModalBase):
    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://dashscope.aliyuncs.com/api/v1',
                 base_websocket_url: str = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series="QWEN",
                                      model_name=model_name, return_trace=return_trace, **kwargs)
        dashscope.api_key = lazyllm.config['qwen_api_key']
        dashscope.base_http_api_url = base_url
        dashscope.base_websocket_api_url = base_websocket_url
        self._api_key = api_key


class QwenSTTModule(QwenModule):
    MODEL_NAME = "paraformer-v2"

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        QwenModule.__init__(self, api_key=api_key,
                            model_name=model or lazyllm.config['qwen_stt_model_name'] or QwenSTTModule.MODEL_NAME,
                            return_trace=return_trace, **kwargs)

    def _forward(self, files: List[str] = [], **kwargs):
        assert any(file.startswith('http') for file in files), "QwenSTTModule only supports http file urls"
        call_params = {'model': self._model_name, 'file_urls': files, **kwargs}
        if self._api_key: call_params['api_key'] = self._api_key
        task_response = dashscope.audio.asr.Transcription.async_call(**call_params)
        transcribe_response = dashscope.audio.asr.Transcription.wait(task=task_response.output.task_id)
        if transcribe_response.status_code == HTTPStatus.OK:
            result_text = ""
            for task in transcribe_response.output.results:
                assert task['subtask_status'] == "SUCCEEDED", "subtask_status is not SUCCEEDED"
                response = json.loads(requests.get(task['transcription_url']).text)
                for transcript in response['transcripts']:
                    result_text = re.sub(r"<[^>]+>", "", transcript['text'])
            return result_text
        else:
            lazyllm.LOG.error(f"failed to transcribe: {transcribe_response.output}")
            raise Exception(f"failed to transcribe: {transcribe_response.output.message}")


class QwenTextToImageModule(QwenModule):
    MODEL_NAME = "wanx2.1-t2i-turbo"

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        QwenModule.__init__(self, api_key=api_key,
                            model_name=model or lazyllm.config['qwen_text2image_model_name']
                            or QwenTextToImageModule.MODEL_NAME, return_trace=return_trace, **kwargs)

    def _forward(self, input: str = None, negative_prompt: str = None, n: int = 1, prompt_extend: bool = True,
                 size: str = '1024*1024', seed: int = None, **kwargs):
        call_params = {
            'model': self._model_name,
            'prompt': input,
            'negative_prompt': negative_prompt,
            'n': n,
            'prompt_extend': prompt_extend,
            'size': size,
            **kwargs
        }
        if self._api_key: call_params['api_key'] = self._api_key
        if seed: call_params['seed'] = seed
        task_response = dashscope.ImageSynthesis.async_call(**call_params)
        response = dashscope.ImageSynthesis.wait(task=task_response.output.task_id)
        if response.status_code == HTTPStatus.OK:
            return encode_query_with_filepaths(None, bytes_to_file([requests.get(result.url).content
                                                                    for result in response.output.results]))
        else:
            lazyllm.LOG.error(f"failed to generate image: {response.output}")
            raise Exception(f"failed to generate image: {response.output.message}")


def synthesize_qwentts(input: str, model_name: str, voice: str, speech_rate: float, volume: int, pitch: float,
                       api_key: str = None, **kwargs):
    call_params = {
        'model': model_name,
        'text': input,
        'voice': voice,
        **kwargs
    }
    if api_key: call_params['api_key'] = api_key
    response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(**call_params)
    if response.status_code == HTTPStatus.OK:
        return requests.get(response.output['audio']['url']).content
    else:
        lazyllm.LOG.error(f"failed to synthesize: {response}")
        raise Exception(f"failed to synthesize: {response.message}")

def synthesize(input: str, model_name: str, voice: str, speech_rate: float, volume: int, pitch: float,
               api_key: str = None, **kwargs):
    assert api_key is None, f"{model_name} does not support multi user, don't set api_key"
    model_name = model_name + '-' + voice
    response = dashscope.audio.tts.SpeechSynthesizer.call(model=model_name, text=input, volume=volume,
                                                          pitch=pitch, rate=speech_rate, **kwargs)
    if response.get_response().status_code == HTTPStatus.OK:
        return response.get_audio_data()
    else:
        lazyllm.LOG.error(f"failed to synthesize: {response.get_response()}")
        raise Exception(f"failed to synthesize: {response.get_response().message}")

def synthesize_v2(input: str, model_name: str, voice: str, speech_rate: float, volume: int, pitch: float,
                  api_key: str = None, **kwargs):
    assert api_key is None, f"{model_name} does not support multi user, don't set api_key"
    synthesizer = dashscope.audio.tts_v2.SpeechSynthesizer(model=model_name, voice=voice, volume=volume,
                                                           pitch_rate=pitch, speech_rate=speech_rate, **kwargs)
    audio = synthesizer.call(input)
    if synthesizer.last_response['header']['event'] == 'task-finished':
        return audio
    else:
        lazyllm.LOG.error(f"failed to synthesize: {synthesizer.last_response}")
        raise Exception(f"failed to synthesize: {synthesizer.last_response['header']['error_message']}")


class QwenTTSModule(QwenModule):
    MODEL_NAME = "qwen-tts"
    SYNTHESIZERS = {
        "cosyvoice-v2": (synthesize_v2, 'longxiaochun_v2'),
        "cosyvoice-v1": (synthesize_v2, 'longxiaochun'),
        "sambert": (synthesize, 'zhinan-v1'),
        "qwen-tts": (synthesize_qwentts, 'Cherry'),
        "qwen-tts-latest": (synthesize_qwentts, 'Cherry')
    }

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        QwenModule.__init__(self, api_key=api_key,
                            model_name=model or lazyllm.config['qwen_tts_model_name'] or QwenTTSModule.MODEL_NAME,
                            return_trace=return_trace, **kwargs)
        if self._model_name not in self.SYNTHESIZERS:
            raise ValueError(f"unsupported model: {self._model_name}. "
                             f"supported models: {QwenTTSModule.SYNTHESIZERS.keys()}")
        self._synthesizer_func, self._voice = QwenTTSModule.SYNTHESIZERS[self._model_name]

    def _forward(self, input: str = None, voice: str = None, speech_rate: float = 1.0, volume: int = 50,
                 pitch: float = 1.0, **kwargs):
        call_params = {
            "input": input,
            "model_name": self._model_name,
            "voice": voice or self._voice,
            "speech_rate": speech_rate,
            "volume": volume,
            "pitch": pitch,
            **kwargs
        }
        if self._api_key: call_params['api_key'] = self._api_key
        return encode_query_with_filepaths(None, bytes_to_file(self._synthesizer_func(**call_params)))
