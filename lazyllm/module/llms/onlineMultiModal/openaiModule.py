from openai import OpenAI
from typing import List
import lazyllm
from .onlineMultiModalBase import OnlineMultiModalBase
from lazyllm.components.utils.file_operate import base64_to_file
import os
from pathlib import Path

class OpenAIModule(OnlineMultiModalBase):
    def __init__(self, model_series: str, model_name: str = None, api_key: str = None,
                 base_url: str = 'https://api.openai.com/v1', base_websocket_url: str = None,
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series=model_series, model_name=model_name,
                                      return_trace=return_trace, **kwargs)
        self._client = OpenAI(api_key=api_key or lazyllm.config['openai_api_key'], base_url=base_url,
                              websocket_base_url=base_websocket_url)

class OpenAITTSModule(OpenAIModule):
    MODEL_NAME = "gpt-4o-mini-tts"

    def __init__(self, model: str = None, api_key: str = None, voice: str = None, return_trace: bool = False, **kwargs):
        OpenAIModule.__init__(self, model_series="OPENAI", api_key=api_key or lazyllm.config['openai_api_key'],
                              model_name=model or OpenAITTSModule.MODEL_NAME or lazyllm.config['openai_tts_model_name'],
                              return_trace=return_trace, **kwargs)
        self._voice = voice

    def _forward(self, input: str = None, **kwargs):
        speech_file_path = Path(__file__).parent / "speech.mp3"

        with self._client.audio.speech.with_streaming_response.create(
            model=self._model_name,
            voice=self._voice,
            input=input,
        ) as response:
            response.stream_to_file(speech_file_path)
        return None, [speech_file_path]

class OpenAISTTModule(OpenAIModule):
    MODEL_NAME = "gpt-4o-transcribe"

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        OpenAIModule.__init__(self, model_series="OPENAI", api_key=api_key or lazyllm.config['openai_api_key'],
                              model_name=model or OpenAISTTModule.MODEL_NAME or lazyllm.config['openai_stt_model_name'],
                              return_trace=return_trace, **kwargs)

    def _forward(self, files: List[str] = [], **kwargs):
        assert len(files) == 1, "OpenAISTTModule only supports one file"
        assert os.path.exists(files[0]), f"File {files[0]} not found"
        audio_file = open(files[0], "rb")
        transcription = self._client.audio.transcriptions.create(
            model=self._model_name,
            file=audio_file
        )
        return transcription.text, None

class OpenAITextToImageModule(OpenAIModule):
    MODEL_NAME = "gpt-image-1"

    def __init__(self, model: str = None, api_key: str = None, return_trace: bool = False, **kwargs):
        OpenAIModule.__init__(self, model_series="OPENAI", api_key=api_key or lazyllm.config['openai_api_key'],
                              model_name=model or OpenAITextToImageModule.MODEL_NAME
                              or lazyllm.config['openai_text2image_model_name'],
                              return_trace=return_trace, **kwargs)
        self._format_output = base64_to_file

    def _forward(self, input: str = None, **kwargs):
        result = self._client.images.generate(
            model=self._model_name,
            prompt=input
        )
        image_base64 = result.data[0].b64_json
        return None, [image_base64]
