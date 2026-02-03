import mimetypes
import requests
import lazyllm
import base64
import os
from typing import Any, Dict, List, Union, Optional
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.components.utils.downloader.model_downloader import LLMType
from lazyllm.components.prompter import LazyLLMPrompterBase
from ..base import OnlineChatModuleBase, LazyLLMOnlineText2ImageModuleBase

class GeminiChat(OnlineChatModuleBase):
    MODEL_NAME = 'gemini-2.5-flash'
    NO_PROXY = True

    def __init__(self, base_url: str = 'https://generativelanguage.googleapis.com/v1beta', model: str = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['gemini_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['gemini_model_name'] or self.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)
        self._prompt = GeminiPrompter(show=False, enable_system=False)

    @staticmethod
    def _get_header(api_key: str) -> dict:
        return {'Content-Type': 'application/json'}

    def _get_system_prompt(self):
        return 'You are an intelligent assistant developed by Google. You are a helpful assistant.'

    def _get_chat_url(self, url):
        return url

    def _extract_text(self, obj: Any) -> str:
        items = obj if isinstance(obj, list) else [obj]
        out: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            candidates = item.get('candidates') or []
            if not (isinstance(candidates, list) and candidates and isinstance(candidates[0], dict)):
                continue
            content = candidates[0].get('content')
            if not isinstance(content, dict):
                continue
            parts = content.get('parts') or []
            if isinstance(parts, list) and parts:
                part_texts = [p.get('text') for p in parts
                              if isinstance(p, dict) and isinstance(p.get('text'), str) and p.get('text')
                            ]
                if part_texts:
                    out.append(''.join(part_texts))
                    continue
            text = content.get('text')
            if isinstance(text, str) and text:
                out.append(text)
        return ''.join(out)

    def _format_vl_chat_image_url(self, image_url: str, mime: str) -> List[Dict[str, Any]]:
        return [{'type': 'image_url', 'image_url': {'url': image_url, 'mime': mime}}]

    def _process_data_for_inline(self, data: Any, default_mime: str = 'application/octet-stream') -> Union[Dict, None]:
        if isinstance(data, dict):
            mime = data.get('mime') or data.get('mime_type') or default_mime
            payload = data.get('data') or data.get('url') or ''
            if not payload:
                return None
            return self._process_data_for_inline(payload, mime)

        if isinstance(data, (bytes, bytearray)):
            return {'mime_type': default_mime, 'data': base64.b64encode(data).decode('utf-8')}

        if not isinstance(data, str):
            return None

        if data.startswith('data:'):
            header, payload = data.split(',', 1) if ',' in data else (data, '')
            mime = header[5:].split(';')[0] if header.startswith('data:') else ''
            return {'mime_type': mime or default_mime, 'data': payload}

        if os.path.exists(data):
            mime = mimetypes.guess_type(data)[0] or default_mime
            with open(data, 'rb') as f:
                file_data = f.read()
            return {'mime_type': mime, 'data': base64.b64encode(file_data).decode('utf-8')}

        mime = mimetypes.guess_type(data)[0] or default_mime
        return {'mime_type': mime, 'data': data}

    def _format_input_with_files(self, query: str, query_files: List[Any]) -> List[Dict[str, Any]]:
        if not query_files:
            return [{'text': query}] if query is not None else []

        parts: List[Dict[str, Any]] = []
        if query:
            parts.append({'text': query})

        for file_item in query_files:
            file_data = self._process_data_for_inline(file_item)
            if file_data:
                parts.append({'inline_data': file_data})
        return parts

    def forward(self, __input: Union[Dict, str] = None, *, llm_chat_history: List[List[str]] = None,
                tools: List[Dict[str, Any]] = None, stream_output: bool = False, lazyllm_files=None,
                url: str = None, model: str = None, **kw,):
        stream_output = stream_output or self._stream
        __input, files = self._get_files(__input, lazyllm_files)
        runtime_base_url = (url or kw.pop('base_url', None) or self._base_url).rstrip('/')
        runtime_model = model or kw.pop('model_name', None) or self._model_name

        params = {'input': __input, 'history': llm_chat_history, 'return_dict': True}
        if tools:
            params['tools'] = tools
        gemini_dict = self._prompt.generate_prompt(**params)
        contents = gemini_dict.get('contents', [])
        system_instruction = gemini_dict.get('system_instruction')

        if self.type == LLMType.VLM and files:
            vlm_parts = self._format_input_with_files(contents[-1]['parts'][0].get('text') if contents else __input, files)
            if contents:
                last_content = contents[-1]
                if last_content.get('role') == 'user':
                    last_content['parts'] = vlm_parts
                else:
                    contents.append({'role': 'user', 'parts': vlm_parts})
            else:
                contents.append({'role': 'user', 'parts': vlm_parts})

        stop = kw.pop('stop', None)
        generation_config = {
            'temperature': self._static_params.get('temperature'),
            'topP': self._static_params.get('top_p'),
            'topK': self._static_params.get('top_k'),
            'maxOutputTokens': self._static_params.get('max_tokens'),
            'stopSequences': stop,
        }
        req_body = {
            'contents': contents,
            'generationConfig': {
                k: v for k, v in generation_config.items() if v is not None
            },
        }
        if system_instruction:
            req_body['system_instruction'] = system_instruction
        if kw:
            req_body.update(kw)

        action = 'streamGenerateContent' if stream_output else 'generateContent'
        request_url = (f'{runtime_base_url}/models/{runtime_model}:{action}?key={self._api_key}')
        if stream_output:
            request_url = f'{request_url}&alt=sse'

        proxies = {'http': None, 'https': None} if self.NO_PROXY else None
        with requests.post(request_url, json=req_body, headers=self._header, stream=stream_output,
                           proxies=proxies) as r:
            if r.status_code != 200:
                raise requests.RequestException(f'{r.status_code}: {r.text}')

            if stream_output:
                color = (stream_output.get('color') if isinstance(stream_output, dict) else None)
                full_text: List[str] = []
                with self.stream_output(stream_output):
                    for chunk in self._iter_sse_json(r):
                        text = self._extract_text(chunk)
                        if text:
                            self._stream_output(text, color)
                            full_text.append(text)
                result = ''.join(full_text)
                return (self._formatter({'role': 'assistant', 'content': result}) if result else '')
            text = self._extract_text(r.json())
            return (self._formatter({'role': 'assistant', 'content': text}) if text else '')


class GeminiText2Image(LazyLLMOnlineText2ImageModuleBase):
    MODEL_NAME = 'gemini-2.5-flash-image'
    IMAGE_EDITING_MODEL_NAME = 'nano-banana-pro-preview'

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: str = 'https://generativelanguage.googleapis.com/v1beta',
                 return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['gemini_api_key'],
                         model=model or self.MODEL_NAME, url=base_url, return_trace=return_trace, **kwargs)

    @staticmethod
    def _get_header(api_key: str) -> dict:
        return {'Content-Type': 'application/json'}

    def _build_request_url(self, base_url: str, model: str) -> str:
        runtime_base_url = (base_url or self._base_url).rstrip('/')
        runtime_model = model or self._model_name
        if not runtime_model:
            raise ValueError('Gemini image model is required. Please set `model` explicitly.')
        request_url = f'{runtime_base_url}/models/{runtime_model}:generateContent'
        if self._api_key:
            request_url = f'{request_url}?key={self._api_key}'
        return request_url

    def _extract_image_bytes(self, response_json: Dict[str, Any]) -> List[bytes]:
        image_bytes: List[bytes] = []
        candidates = response_json.get('candidates') or []

        for cand in candidates:
            content = cand.get('content') or {}
            parts = content.get('parts') or []
            for part in parts:
                inline = part.get('inline_data') or part.get('inlineData')
                if inline and isinstance(inline, dict):
                    b64 = inline.get('data')
                    if b64:
                        try:
                            image_bytes.append(base64.b64decode(b64))
                        except Exception:
                            continue

        if image_bytes:
            return image_bytes

        if isinstance(response_json.get('images'), list):
            for item in response_json['images']:
                b64 = item.get('imageBytes') or item.get('bytesBase64Encoded')
                if b64:
                    try:
                        image_bytes.append(base64.b64decode(b64))
                    except Exception:
                        continue

        return image_bytes

    def _build_parts(self, prompt: Optional[str], files: Optional[List[str]]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        if prompt:
            parts.append({'text': prompt})
        if files:
            image_results = self._load_images(files)
            for base64_str, _ in image_results:
                parts.append({
                    'inline_data': {
                        'mime_type': 'image/png',
                        'data': base64_str
                    }
                })
        return parts

    def _forward(self, input: str = None, files: List[str] = None,
                 url: str = None, model: str = None, timeout: int = 180,
                 **kwargs):
        has_ref_image = files is not None and len(files) > 0
        if self._type == LLMType.IMAGE_EDITING and not has_ref_image:
            raise ValueError(
                f'Image editing is enabled for model {self._model_name}, but no image file was provided. '
                f'Please provide an image file via the "files" parameter.'
            )
        if self._type != LLMType.IMAGE_EDITING and has_ref_image:
            raise ValueError(
                f'Image file was provided, but image editing is not enabled for model {self._model_name}. '
                f'Please use default image-editing model {self.IMAGE_EDITING_MODEL_NAME} or other image-editing model'
            )

        request_url = self._build_request_url(url or self._base_url, model or self._model_name)
        payload = kwargs.pop('payload', None)
        if payload is None:
            payload = {
                'contents': [
                    {'role': 'user', 'parts': self._build_parts(input, files)}
                ],
                'generationConfig': {
                    'responseModalities': ['IMAGE']
                }
            }
        else:
            payload = dict(payload)

        payload.update(kwargs)

        response = requests.post(request_url, headers=self._header, json=payload, timeout=timeout)
        if response.status_code != 200:
            raise requests.RequestException(f'{response.status_code}: {response.text}')
        response_json = response.json()
        image_bytes = self._extract_image_bytes(response_json)
        if not image_bytes:
            raise ValueError(f'No images returned from API: {response_json}')
        file_paths = bytes_to_file(image_bytes)
        return encode_query_with_filepaths(None, file_paths)


class GeminiPrompter(LazyLLMPrompterBase):
    def __init__(self, show=False, tools=None, history=None, *, enable_system: bool = False):
        super().__init__(show=show, tools=tools, history=history, enable_system=enable_system)
        self._init_prompt('{sos}{system}{instruction}{tools}{eos}', '')

    def generate_prompt(self, input=None, history=None, tools=None, label=None, show=False, return_dict=True):
        base = super().generate_prompt(
            input=input, history=history, tools=tools, label=label, show=show, return_dict=True
        )
        messages = base.get('messages', [])
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                if system_instruction is None and isinstance(content, str):
                    system_instruction = {'parts': [{'text': content}]}
                continue

            parts = []
            if isinstance(content, str):
                parts = [{'text': content}]
            elif isinstance(content, list):
                text_parts = [item.get('text', '') for item in content if item.get('type') == 'text']
                parts = [{'text': ''.join(text_parts)}] if text_parts else []

            if parts:
                contents.append({'role': 'user' if role == 'user' else 'model', 'parts': parts})

        result = {'contents': contents}
        if system_instruction:
            result['system_instruction'] = system_instruction
        if tools:
            result['tools'] = tools

        if self._show or show:
            lazyllm.LOG.info(result)
        return result
