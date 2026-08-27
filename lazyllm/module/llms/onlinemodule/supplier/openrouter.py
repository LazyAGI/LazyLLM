import base64
import mimetypes
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests

from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file

from ..base import (
    LazyLLMOnlineSTTModuleBase,
    LazyLLMOnlineTTSModuleBase,
    LazyLLMOnlineText2ImageModuleBase,
    LazyLLMOnlineText2VideoModuleBase,
)
from .openai import OpenAIChat, OpenAIEmbed


class OpenRouterChat(OpenAIChat):
    PROVIDER_NAME = 'openrouter'
    TRAINABLE_MODEL_LIST = []

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            base_url=base_url or 'https://openrouter.ai/api/v1/',
            model=model or 'openrouter/free',
            api_key=api_key,
            stream=stream,
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    def _get_system_prompt(self):
        return 'You are a helpful assistant accessed through OpenRouter.'


class OpenRouterEmbed(OpenAIEmbed):
    def __init__(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None,
                 api_key: str = None, **kwargs):
        super().__init__(
            embed_url=embed_url or 'https://openrouter.ai/api/v1/',
            embed_model_name=embed_model_name or 'liquid/lfm-2.5-embedding-350m:free',
            api_key=api_key,
            **kwargs,
        )


class OpenRouterText2Image(LazyLLMOnlineText2ImageModuleBase):
    MODEL_NAME = 'bytedance-seed/seedream-4.5'

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: Optional[str] = None, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            api_key=api_key or self._default_api_key(),
            model=model or self.MODEL_NAME,
            url=base_url or 'https://openrouter.ai/api/v1/',
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    def _forward(self, input: str = None, files: List[str] = None, url: str = None,
                 model: str = None, n: int = 1, **kwargs):
        if files:
            raise ValueError('OpenRouter text-to-image does not accept reference images in this mode.')
        payload = {'model': model or self._model_name, 'prompt': input, 'n': n, **kwargs}
        response = requests.post(
            urljoin(url or self._base_url, 'images'),
            headers=self._header,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()

        image_bytes = []
        for item in response.json().get('data', []):
            if item.get('b64_json'):
                encoded = item['b64_json'].split(',', 1)[-1]
                image_bytes.append(base64.b64decode(encoded, validate=True))
            elif item.get('url'):
                image_bytes.append(self._load_images(item['url'])[0][1])
        if not image_bytes:
            raise RuntimeError('OpenRouter image API returned no images.')
        return encode_query_with_filepaths(None, bytes_to_file(image_bytes))


class OpenRouterText2Video(LazyLLMOnlineText2VideoModuleBase):
    MODEL_NAME = 'bytedance/seedance-2.0-mini'

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: Optional[str] = None, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            api_key=api_key or self._default_api_key(),
            model=model or self.MODEL_NAME,
            url=base_url or 'https://openrouter.ai/api/v1/',
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    def _build_input_references(self, files: List[str] = None):
        references = []
        for file in files or []:
            if file.startswith(('http://', 'https://', 'data:')):
                image_url = file
            else:
                encoded, _ = self._load_images(file)[0]
                content_type = mimetypes.guess_type(file)[0] or 'image/png'
                image_url = f'data:{content_type};base64,{encoded}'
            references.append({'type': 'image_url', 'image_url': {'url': image_url}})
        return references

    def _forward(self, input: str = None, files: List[str] = None, url: str = None,
                 model: str = None, poll_interval: float = 3.0, poll_timeout: float = 900.0,
                 **kwargs):
        api_root = url or self._base_url
        payload = {'model': model or self._model_name, 'prompt': input, **kwargs}
        if references := self._build_input_references(files):
            payload['input_references'] = references

        response = requests.post(
            urljoin(api_root, 'videos'), headers=self._header, json=payload, timeout=180)
        response.raise_for_status()
        job = response.json()
        job_id = job.get('id')
        if not job_id:
            raise RuntimeError('OpenRouter video API returned no job ID.')

        deadline = time.monotonic() + poll_timeout
        while True:
            status = job.get('status')
            if status == 'completed':
                content_response = requests.get(
                    urljoin(api_root, f'videos/{job_id}/content'),
                    headers=self._get_empty_header(),
                    timeout=180,
                )
                content_response.raise_for_status()
                return encode_query_with_filepaths(None, bytes_to_file([content_response.content]))
            if status in {'failed', 'cancelled', 'expired'}:
                raise RuntimeError(f'OpenRouter video generation {status}: {job.get("error") or job_id}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'OpenRouter video generation timed out for job {job_id}.')
            time.sleep(poll_interval)
            status_response = requests.get(
                urljoin(api_root, f'videos/{job_id}'), headers=self._header, timeout=180)
            status_response.raise_for_status()
            job = status_response.json()


class OpenRouterTTS(LazyLLMOnlineTTSModuleBase):
    MODEL_NAME = 'deepgram/flux-tts:free'

    def __init__(self, api_key: str = None, model: str = None, model_name: str = None,
                 base_url: Optional[str] = None, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            api_key=api_key or self._default_api_key(),
            model=model or model_name or self.MODEL_NAME,
            url=base_url or 'https://openrouter.ai/api/v1/',
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    def _forward(self, input: str = None, response_format: str = 'mp3', speed: float = 1.0,
                 voice: str = None, out_path: str = None, url: str = None,
                 model: str = None, **kwargs):
        payload = {
            'model': model or self._model_name,
            'input': input,
            'response_format': response_format,
            'speed': speed,
            **kwargs,
        }
        if voice:
            payload['voice'] = voice
        response = requests.post(
            urljoin(url or self._base_url, 'audio/speech'),
            headers=self._header,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        file_path = bytes_to_file([response.content])[0]
        if out_path:
            shutil.copyfile(file_path, out_path)
            file_path = out_path
        return encode_query_with_filepaths(None, [file_path])


class OpenRouterSTT(LazyLLMOnlineSTTModuleBase):
    MODEL_NAME = 'openai/whisper-large-v3-turbo'

    def __init__(self, api_key: str = None, model: str = None, model_name: str = None,
                 base_url: Optional[str] = None, return_trace: bool = False,
                 skip_auth: bool = False, **kwargs):
        super().__init__(
            api_key=api_key or self._default_api_key(),
            model=model or model_name or self.MODEL_NAME,
            url=base_url or 'https://openrouter.ai/api/v1/',
            return_trace=return_trace,
            skip_auth=skip_auth,
            **kwargs,
        )

    @staticmethod
    def _resolve_audio_path(input: str = None, files: List[str] = None) -> str:
        if files and len(files) > 1:
            raise ValueError('OpenRouter STT only supports one audio file at a time.')
        file_path = files[0] if files else input
        if not file_path or not os.path.isfile(file_path):
            raise ValueError('OpenRouter STT requires a local audio file path.')
        return file_path

    def _forward(self, input: str = None, files: List[str] = None, language: str = None,
                 response_format: str = 'json', url: str = None, model: str = None,
                 **kwargs):
        file_path = self._resolve_audio_path(input=input, files=files)
        audio_format = Path(file_path).suffix.lstrip('.').lower() or 'wav'
        with open(file_path, 'rb') as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('ascii')
        payload = {
            'model': model or self._model_name,
            'input_audio': {'data': encoded_audio, 'format': audio_format},
            'response_format': response_format,
            **kwargs,
        }
        if language:
            payload['language'] = language
        response = requests.post(
            urljoin(url or self._base_url, 'audio/transcriptions'),
            headers=self._header,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get('text', '')
