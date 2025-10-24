import requests
import lazyllm
from typing import Dict, List, Union
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase

class SiliconFlowModule(OnlineChatModuleBase, FileHandlerBase):
    VLM_MODEL_PREFIX = ['Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen3-VL-30B-A3B-Instruct', 'deepseek-ai/deepseek-vl2',
                        'Qwen/Qwen3-VL-30B-A3B-Thinking', 'THUDM/GLM-4.1V-9B-Thinking']

    def __init__(self, base_url: str = 'https://api.siliconflow.cn/v1/', model: str = 'Qwen/QwQ-32B',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        OnlineChatModuleBase.__init__(self, model_series='SILICONFLOW',
                                      api_key=api_key or lazyllm.config['siliconflow_api_key'],
                                      base_url=base_url, model_name=model, stream=stream,
                                      return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        if stream:
            self._model_optional_params['stream'] = True

    def _get_system_prompt(self):
        return 'You are an intelligent assistant provided by SiliconFlow. You are a helpful assistant.'

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')

class SiliconFlowEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self, embed_url: str = 'https://api.siliconflow.cn/v1/embeddings',
                 embed_model_name: str = 'BAAI/bge-large-zh-v1.5', api_key: str = None,
                 batch_size: int = 16, **kw):
        super().__init__('SILICONFLOW', embed_url, api_key or lazyllm.config['siliconflow_api_key'],
                         embed_model_name, batch_size=batch_size, **kw)


class SiliconFlowReranking(OnlineEmbeddingModuleBase):
    def __init__(self, rerank_url: str = 'https://api.siliconflow.cn/v1/rerank',
                 rerank_model_name: str = 'BAAI/bge-reranker-v2-m3', api_key: str = None, **kw):
        super().__init__('SILICONFLOW', rerank_url, api_key or lazyllm.config['siliconflow_api_key'],
                         rerank_model_name, **kw)
        self._rerank_model_name = rerank_model_name

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict:
        if isinstance(input, str):
            raise ValueError('Rerank requires both query and documents')

        if isinstance(input, list) and len(input) == 2:
            query = input[0]
            documents = input[1]
        elif isinstance(input, dict):
            query = input.get('query', '')
            documents = input.get('documents', [])
        else:
            raise ValueError("Input must be a list [query, documents] or dict with 'query' and 'documents' keys")

        json_data = {
            'model': self._rerank_model_name,
            'query': query,
            'documents': documents
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[Dict]:
        return response.get('results', [])

class SiliconFlowTextToImageModule(OnlineMultiModalBase):
    MODEL_NAME = 'Qwen/Qwen-Image'

    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series='SiliconFlow',
                                      model_name=model_name or SiliconFlowTextToImageModule.MODEL_NAME,
                                      return_trace=return_trace, **kwargs)
        self._endpoint = 'images/generations'
        self._base_url = base_url
        self._api_key = api_key or lazyllm.config['siliconflow_api_key']

    def _make_request(self, endpoint, payload, timeout=60):

        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json'
        }

        url = f'{self._base_url}{endpoint}'

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            lazyllm.LOG.error(f'API request failed: {str(e)}')
            raise

    def _forward(self, input: str = None, size: str = '1024x1024', **kwargs):
        payload = {
            'model': self._model_name,
            'prompt': input
        }
        payload.update(kwargs)

        result = self._make_request(self._endpoint, payload)

        image_urls = [item['url'] for item in result['data']]

        image_files = []
        for url in image_urls:
            img_response = requests.get(url, timeout=60)
            if img_response.status_code == 200:
                image_files.append(img_response.content)
            else:
                raise Exception(f'Failed to download image from {url}')

        file_paths = bytes_to_file(image_files)

        if self._return_trace:
            return {
                'response': encode_query_with_filepaths(None, file_paths),
                'trace_info': {
                    'model': self._model_name,
                    'full_response': result
                }
            }
        return encode_query_with_filepaths(None, file_paths)

class SiliconFlowTTS(OnlineMultiModalBase):
    MODEL_NAME = 'fnlp/MOSS-TTSD-v0.5'

    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series='SiliconFlow',
                                      model_name=model_name or SiliconFlowTTS.MODEL_NAME,
                                      return_trace=return_trace, **kwargs)
        self._endpoint = 'audio/speech'
        self._base_url = base_url
        self._api_key = api_key or lazyllm.config['siliconflow_api_key']

    def _make_binary_request(self, endpoint, payload, timeout=180):

        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json'
        }

        url = f'{self._base_url}{endpoint}'

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            lazyllm.LOG.error(f'API request failed: {str(e)}')
            raise

    def _forward(self, input: str = None, response_format: str = 'mp3',
                 sample_rate: int = 44100, speed: float = 1.0,
                 voice: str = None, references=None, out_path: str = None, **kwargs):

        payload = {
            'model': self._model_name,
            'input': input,
            'response_format': response_format,
            'sample_rate': sample_rate,
            'speed': speed
        }

        if voice:
            payload['voice'] = voice
        if references:
            payload['references'] = references

        payload.update(kwargs)
        audio_content = self._make_binary_request(self._endpoint, payload, timeout=180)
        file_path = bytes_to_file([audio_content])[0]

        if out_path:
            with open(file_path, 'rb') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
            file_path = out_path

        result = encode_query_with_filepaths(None, [file_path])

        if self._return_trace:
            return {
                'response': result,
                'trace_info': {
                    'model': self._model_name,
                    'full_response': f'Audio generated successfully, length: {len(audio_content)} bytes'
                }
            }
        return result
