import os
import requests
from typing import Any, Dict, List, Union
from urllib.parse import urljoin

import lazyllm
from lazyllm.components.utils.file_operate import _image_to_base64
from lazyllm.components.prompter import LazyLLMPrompterBase
from ..base import OnlineChatModuleBase


class ClaudeChat(OnlineChatModuleBase):
    MODEL_NAME = 'claude-4-5-sonnet-latest'
    NO_PROXY = True
    ANTHROPIC_VERSION = '2023-06-01'

    def __init__(self, base_url: str = 'https://api.anthropic.com/v1', model: str = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['claude_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['claude_model_name'] or self.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)
        self._prompt = ClaudePrompter(show=False, enable_system=True)

    @staticmethod
    def _get_header(api_key: str) -> dict:
        return {
            'Content-Type': 'application/json',
            'x-api-key': api_key or '',
            'anthropic-version': ClaudeChat.ANTHROPIC_VERSION,
        }

    def _get_system_prompt(self):
        return 'You are an intelligent assistant developed by Anthropic. You are a helpful assistant.'

    def _get_chat_url(self, url):
        if url.rstrip('/').endswith('messages'):
            return url
        return urljoin(url.rstrip('/') + '/', 'messages')

    def _extract_text(self, obj: Any) -> str:
        if not isinstance(obj, dict):
            return ''
        content = obj.get('content') or []
        texts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
        return ''.join(filter(None, texts))

    def _format_input_with_files(self, query: str, query_files: List[Any]) -> List[Dict[str, Any]]:
        if not query_files:
            return [{'type': 'text', 'text': query}] if query is not None else []

        parts: List[Dict[str, Any]] = []
        if query:
            parts.append({'type': 'text', 'text': query})

        for file_item in query_files:
            if isinstance(file_item, str) and file_item.startswith('data:'):
                header, payload = file_item.split(',', 1) if ',' in file_item else (file_item, '')
                mime = header[5:].split(';')[0] if header.startswith('data:') else 'image/png'
                parts.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': mime or 'image/png',
                        'data': payload,
                    }
                })
                continue

            if isinstance(file_item, str) and os.path.exists(file_item):
                base64_str, mime = _image_to_base64(file_item)
                if base64_str:
                    parts.append({
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': mime or 'image/png',
                            'data': base64_str,
                        }
                    })
                continue

        return parts

    def forward(self, __input: Union[Dict, str] = None, *, llm_chat_history: List[List[str]] = None,
                tools: List[Dict[str, Any]] = None, stream_output: bool = False,
                lazyllm_files=None, url: str = None, model: str = None, **kw):
        stream_output = stream_output or self._stream
        __input, files = self._get_files(__input, lazyllm_files)
        runtime_base_url = (url or kw.pop('base_url', None) or self._base_url).rstrip('/')
        runtime_url = self._get_chat_url(runtime_base_url)
        runtime_model = model or kw.pop('model_name', None) or self._model_name

        params = {'input': __input, 'history': llm_chat_history, 'return_dict': True}
        if tools:
            params['tools'] = tools
        claude_dict = self._prompt.generate_prompt(**params)
        messages = claude_dict.get('messages', [])
        system = claude_dict.get('system')

        if self.type == 'VLM' and files:
            if messages and messages[-1].get('role') == 'user':
                content = messages[-1].get('content')
                query_text = ''
                if isinstance(content, list):
                    query_text = ''.join([
                        item.get('text', '') for item in content
                        if isinstance(item, dict) and item.get('type') == 'text'
                    ])
                elif isinstance(content, str):
                    query_text = content
                messages[-1]['content'] = self._format_input_with_files(query_text or __input, files)
            else:
                messages.append({'role': 'user', 'content': self._format_input_with_files(__input, files)})

        max_tokens = kw.pop('max_tokens', None) or self._static_params.get('max_tokens') or 1024
        payload = {
            'model': runtime_model,
            'messages': messages,
            'max_tokens': max_tokens,
            'stream': bool(stream_output),
        }
        if system:
            payload['system'] = system
        if self._static_params.get('temperature') is not None:
            payload['temperature'] = self._static_params.get('temperature')
        if self._static_params.get('top_p') is not None:
            payload['top_p'] = self._static_params.get('top_p')
        if self._static_params.get('top_k') is not None:
            payload['top_k'] = self._static_params.get('top_k')
        if tools:
            payload['tools'] = tools
        if kw:
            payload.update(kw)

        proxies = {'http': None, 'https': None} if self.NO_PROXY else None
        with requests.post(runtime_url, json=payload, headers=self._header, stream=stream_output,
                           proxies=proxies) as r:
            if r.status_code != 200:
                msg = r.text if not stream_output else '\n'.join([c.decode('utf-8') for c in r.iter_content(None)])
                raise requests.RequestException(f'{r.status_code}: {msg}')

            if stream_output:
                color = stream_output.get('color') if isinstance(stream_output, dict) else None
                full_text: List[str] = []
                with self.stream_output(stream_output):
                    for chunk in self._iter_sse_json(r):
                        chunk_type = chunk.get('type')
                        if chunk_type == 'content_block_delta':
                            delta = chunk.get('delta') or {}
                            text = delta.get('text')
                        elif chunk_type == 'message_delta':
                            text = ''
                        elif chunk_type == 'content_block_start':
                            content = chunk.get('content_block') or {}
                            text = content.get('text') if content.get('type') == 'text' else ''
                        else:
                            text = ''
                        if text:
                            self._stream_output(text, color)
                            full_text.append(text)
                result = ''.join(full_text)
                return self._formatter({'role': 'assistant', 'content': result}) if result else ''

            text = self._extract_text(r.json())
            return self._formatter({'role': 'assistant', 'content': text}) if text else ''


class ClaudePrompter(LazyLLMPrompterBase):
    def __init__(self, show=False, tools=None, history=None, *, enable_system: bool = True):
        super().__init__(show=show, tools=tools, history=history, enable_system=enable_system)
        self._init_prompt('{sos}{system}{instruction}{tools}{eos}', '')

    def _convert_content(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{'type': 'text', 'text': content}]
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get('type') == 'text' and item.get('text'):
                    parts.append({'type': 'text', 'text': item.get('text')})
                elif item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url')
                    if isinstance(image_url, str) and image_url.startswith('data:'):
                        header, payload = image_url.split(',', 1) if ',' in image_url else (image_url, '')
                        mime = header[5:].split(';')[0] if header.startswith('data:') else 'image/png'
                        parts.append({
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': mime or 'image/png',
                                'data': payload,
                            }
                        })
            return parts
        return []

    def generate_prompt(self, input=None, history=None, tools=None, label=None, show=False, return_dict=True):
        base = super().generate_prompt(
            input=input, history=history, tools=tools, label=label, show=show, return_dict=True
        )
        messages = base.get('messages', [])
        system = None
        out_messages = []

        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'system':
                if system is None and isinstance(content, str):
                    system = content
                continue
            out_messages.append({'role': role, 'content': self._convert_content(content)})

        result = {'messages': out_messages}
        if system:
            result['system'] = system
        if tools:
            result['tools'] = tools

        if self._show or show:
            lazyllm.LOG.info(result)
        return result
