import requests
import lazyllm
from typing import Tuple, List, Dict, Union
from urllib.parse import urljoin
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase
from lazyllm import LOG


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

    def _validate_api_key(self):
        '''Validate API Key by sending a minimal request'''
        try:
            # SiliconFlow validates API key using a minimal chat request
            models_url = urljoin(self._base_url, 'models')
            response = requests.get(models_url, headers=self._header, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


class SiliconFlowEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self, embed_url: str = 'https://api.siliconflow.cn/v1/embeddings',
                 embed_model_name: str = 'BAAI/bge-large-zh-v1.5', api_key: str = None,
                 batch_size: int = 16, **kw):
        super().__init__('SILICONFLOW', embed_url, api_key or lazyllm.config['siliconflow_api_key'],
                         embed_model_name, batch_size=batch_size, **kw)


class SiliconFlowReranking(OnlineEmbeddingModuleBase):
    def __init__(self, embed_url: str = 'https://api.siliconflow.cn/v1/rerank',
                 embed_model_name: str = 'BAAI/bge-reranker-v2-m3', api_key: str = None, **kw):
        super().__init__('SILICONFLOW', embed_url, api_key or lazyllm.config['siliconflow_api_key'],
                         embed_model_name, **kw)

    def _encapsulated_data(self, query: str, documents: List[str], top_n: int, **kwargs) -> Dict:
        json_data = {
            'model': self._embed_model_name,
            'query': query,
            'documents': documents,
            'top_n': top_n
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)
        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[Tuple]:
        results = response.get('results', [])
        return [(result['index'], result['relevance_score']) for result in results]


class SiliconFlowTextToImageModule(OnlineMultiModalBase):
    MODEL_NAME = 'Qwen/Qwen-Image'
    IMAGE_EDITING_MODEL_NAME = 'Qwen/Qwen-Image-Edit-2509'

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series='SiliconFlow', api_key=api_key or lazyllm.config['siliconflow_api_key'],
                                      model=model or SiliconFlowTextToImageModule.MODEL_NAME,
                                      base_url=base_url, return_trace=return_trace, **kwargs)
        self._endpoint = 'images/generations'

    def _make_request(self, endpoint, payload, base_url=None, timeout=180):
        url = f'{(base_url or self._base_url)}{endpoint}'
        try:
            response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            LOG.error(f'API request failed: {str(e)}')
            raise

    def _forward(self, input: str = None, files: List[str] = None, size: str = '1024x1024', url: str = None,
                 model: str = None, **kwargs):
        has_ref_image = files is not None and len(files) > 0
        reference_image_data = None
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

        payload = {
            'model': model,
            'prompt': input,
            **kwargs
        }
        if has_ref_image:
            if len(files) > 3:
                raise ValueError(
                    f'Too many reference images provided ({len(files)}). '
                    f'Image editing supports at most 3 reference images.'
                )
            for i, file in enumerate(files):
                reference_image_base64, _ = self._load_image(file)
                reference_image_data = f"data:image/png;base64,{reference_image_base64}"
                if i == 0:
                    payload['image'] = reference_image_data
                elif i > 0:
                    payload[f'image{i+1}'] = reference_image_data
        try:
            result = self._make_request(self._endpoint, payload)
            image_urls = [item['url'] for item in result.get('data', [])]
            if not image_urls:
                raise Exception('No images returned from API')            
            image_bytes = []
            for image_url in image_urls:
                try:
                    _, image_byte = self._load_image(image_url)
                    image_bytes.append(image_byte)
                except Exception as e:
                    LOG.warning(f'Failed to download image from {image_url}: {str(e)}')
            if not image_bytes:
                raise Exception('Failed to download any images')
            file_paths = bytes_to_file(image_bytes)
            return encode_query_with_filepaths(None, file_paths)
        except Exception as e:
            LOG.error(f'Error in SiliconFlowTextToImageModule._forward: {str(e)}')
            raise

class SiliconFlowTTSModule(OnlineMultiModalBase):
    MODEL_NAME = 'fnlp/MOSS-TTSD-v0.5'

    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series='SiliconFlow',
                                      api_key=api_key or lazyllm.config['siliconflow_api_key'],
                                      model_name=model_name or SiliconFlowTTSModule.MODEL_NAME,
                                      return_trace=return_trace, base_url=base_url, **kwargs)
        self._endpoint = 'audio/speech'

    def _make_binary_request(self, endpoint, payload, base_url=None, timeout=180):
        url = f'{(base_url or self._base_url)}{endpoint}'
        try:
            response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            LOG.error(f'API request failed: {str(e)}')
            raise

    def _forward(self, input: str = None, response_format: str = 'mp3',
                 sample_rate: int = 44100, speed: float = 1.0,
                 voice: str = None, references=None, out_path: str = None,
                 url: str = None, model: str = None, **kwargs):

        if not voice:
            active_model = model
            if active_model == 'fnlp/MOSS-TTSD-v0.5':
                voice = 'fnlp/MOSS-TTSD-v0.5:alex'
            elif active_model == 'FunAudioLLM/CosyVoice2-0.5B':
                voice = 'FunAudioLLM/CosyVoice2-0.5B:alex'
            else:
                raise ValueError(
                    f'Default voice is only supported for models "fnlp/MOSS-TTSD-v0.5" and '
                    f'"FunAudioLLM/CosyVoice2-0.5B". For model "{active_model}", '
                    f'please provide a valid voice parameter.')
        payload = {
            'model': model,
            'input': input,
            'response_format': response_format,
            'sample_rate': sample_rate,
            'speed': speed,
            'voice': voice
        }

        if references:
            payload['references'] = references

        payload.update(kwargs)
        audio_content = self._make_binary_request(self._endpoint, payload, base_url=url, timeout=180)
        file_path = bytes_to_file([audio_content])[0]

        if out_path:
            with open(file_path, 'rb') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
            file_path = out_path

        result = encode_query_with_filepaths(None, [file_path])

        return result
