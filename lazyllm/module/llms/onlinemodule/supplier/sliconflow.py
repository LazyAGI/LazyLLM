import lazyllm
from typing import Dict, List, Union
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
import os, tempfile


class SiliconFlowModule(OnlineChatModuleBase):
    def __init__(self, base_url: str = 'https://api.siliconflow.cn/v1', model: str = 'Qwen/QwQ-32B',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(model_series='SILICONFLOW', api_key=api_key or lazyllm.config['siliconflow_api_key'],
                         base_url=base_url, model_name=model, stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return 'You are an intelligent assistant provided by SiliconFlow. You are a helpful assistant.'

    def _set_chat_url(self):
        self._url = urljoin(self._base_url, 'chat/completions')

class SiliconFlowEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = 'https://api.siliconflow.cn/v1/embeddings',
                 embed_model_name: str = 'BAAI/bge-large-zh-v1.5',
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        super().__init__('SILICONFLOW', embed_url, api_key or lazyllm.config['siliconflow_api_key'], 
                         embed_model_name, batch_size=batch_size, **kw)


class SiliconFlowRerankModule(OnlineEmbeddingModuleBase):
    def __init__(self,
                 rerank_url: str = 'https://api.siliconflow.cn/v1/rerank',
                 rerank_model_name: str = 'BAAI/bge-reranker-v2-m3',
                 api_key: str = None,
                 **kw):
        super().__init__('SILICONFLOW', rerank_url, api_key or lazyllm.config['siliconflow_api_key'], 
                         rerank_model_name, **kw)
        self._rerank_model_name = rerank_model_name

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict:
        if isinstance(input, str):
            raise ValueError("Rerank requires both query and documents")
        
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

class SiliconFlowBase(OnlineMultiModalBase):
    
    def __init__(self, model_series='SILICONFLOW', model_name=None, 
                 base_url='https://api.siliconflow.cn/v1/', return_trace=False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series=model_series, 
                                     model_name=model_name, return_trace=return_trace, **kwargs)
        self._base_url = base_url
        self._api_key = kwargs.get('api_key') or lazyllm.config['siliconflow_api_key']
        self._headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json'
        }

    def _make_request(self, endpoint, payload, timeout=60):
        url = urljoin(self._base_url, endpoint)
        response = requests.post(url, headers=self._headers, json=payload, timeout=timeout)
        print(url, response.status_code)
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        
        return response.json()


class SiliconFlowMultiModalChat(SiliconFlowBase):
    def __init__(self, api_key: str = None, model_name: str = 'Qwen/Qwen2.5-VL-72B-Instruct', 
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        SiliconFlowBase.__init__(self, model_name=model_name, base_url=base_url,
                                return_trace=return_trace, api_key=api_key, **kwargs)
        self._endpoint = 'chat/completions'

    def _forward(self, input: Union[Dict, str] = None, files: List[str] = None, **kwargs):
        messages = [input] if isinstance(input, dict) else [{'role': 'user', 'content': input}]
        
        payload = {
            'model': self._model_name,
            'messages': messages,
            'stream': False
        }
        payload.update(kwargs)
        
        result = self._make_request(self._endpoint, payload)
        reply = result['choices'][0]['message']['content']
        
        if self._return_trace:
            return {
                'response': reply,
                'trace_info': {
                    'model': self._model_name,
                    'usage': result.get('usage', {}),
                    'full_response': result
                }
            }
        return reply


class SiliconFlowTextToImageModule(SiliconFlowBase):
    MODEL_NAME = 'Qwen/Qwen-Image-Edit-2509'

    def __init__(self, api_key: str = None, model_name: str = None, 
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        SiliconFlowBase.__init__(self, 
                                model_name=model_name or SiliconFlowTextToImageModule.MODEL_NAME,
                                base_url=base_url, return_trace=return_trace, 
                                api_key=api_key, **kwargs)
        self._endpoint = 'images/generations'

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
                raise Exception(f"Failed to download image from {url}")
        
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

class SiliconFlowTTS:
    def __init__(self, api_key: str=None, model: str="fnlp/MOSS-TTSD-v0.5",
                 base_url: str="https://api.siliconflow.cn/v1/"):
        self.api_key = api_key or lazyllm.config['siliconflow_api_key']
        self.url = base_url.rstrip('/') + '/audio/speech'
        self.model = model

    def __call__(self, text: str,
                 response_format: str="mp3", sample_rate: int=44100, speed: float=1.0,
                 voice: str=None, references=None,
                 out_path: str=None) -> str:
        payload = {
            "model": self.model,
            "input": text,                 
            "response_format": response_format,
            "sample_rate": sample_rate,
            "speed": speed
        }
        if voice:
            payload["voice"] = voice
        if references:
            payload["references"] = references
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/octet-stream"
        }
        r = requests.post(self.url, headers=headers, json=payload, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"TTS failed: {r.text}")

        if out_path is None:
            suffix = { "mp3": ".mp3", "wav": ".wav", "opus": ".opus", "pcm": ".pcm" }.get(response_format, ".bin")
            f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            f.write(r.content); f.close()
            out_path = f.name
        else:
            with open(out_path, "wb") as f: f.write(r.content)
        return out_path
