import os
import requests
from typing import Tuple, List, Dict, Union, Optional
from urllib.parse import urljoin
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase, LazyLLMOnlineMultimodalEmbedModuleBase,
    LazyLLMOnlineRerankModuleBase, LazyLLMOnlineSTTModuleBase, LazyLLMOnlineText2ImageModuleBase,
    LazyLLMOnlineTTSModuleBase,
)
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file, _image_to_base64
from ..fileHandler import FileHandlerBase
from lazyllm import LOG, config

class SiliconFlowChat(OnlineChatModuleBase, FileHandlerBase):
    VLM_MODEL_PREFIX = ['Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen3-VL-30B-A3B-Instruct', 'deepseek-ai/deepseek-vl2',
                        'Qwen/Qwen3-VL-30B-A3B-Thinking', 'THUDM/GLM-4.1V-9B-Thinking']

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        base_url = base_url or 'https://api.siliconflow.cn/v1/'
        model = model or 'Qwen/Qwen3-8B'
        super().__init__(api_key=api_key or self._default_api_key(), base_url=base_url, model_name=model,
                         stream=stream, return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)

    def _get_system_prompt(self):
        return 'You are an intelligent assistant provided by SiliconFlow. You are a helpful assistant.'


class SiliconFlowEmbed(LazyLLMOnlineEmbedModuleBase):
    def __init__(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None,
                 api_key: str = None,
                 batch_size: int = 16, **kw):
        embed_url = embed_url or 'https://api.siliconflow.cn/v1/embeddings'
        embed_model_name = embed_model_name or 'BAAI/bge-large-zh-v1.5'
        super().__init__(embed_url, api_key or self._default_api_key(),
                         embed_model_name, batch_size=batch_size, **kw)


class SiliconFlowMultimodalEmbed(LazyLLMOnlineMultimodalEmbedModuleBase):
    MODEL_NAME = 'Qwen/Qwen3-VL-Embedding-8B'

    def __init__(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None,
                 api_key: str = None, batch_size: int = 1, **kw):
        kw.pop('type', None)
        if batch_size != 1:
            LOG.warning('SiliconFlowMultimodalEmbed does not support batch_size > 1; resetting batch_size to 1.')
            batch_size = 1
        embed_url = embed_url or 'https://api.siliconflow.cn/v1/embeddings'
        embed_model_name = embed_model_name or SiliconFlowMultimodalEmbed.MODEL_NAME
        super().__init__(embed_url, api_key or self._default_api_key(),
                         embed_model_name, batch_size=batch_size, **kw)

    @staticmethod
    def _format_image(image: str) -> str:
        if image.startswith(('http://', 'https://', 'data:')):
            return image
        if not os.path.exists(image):
            return image
        image_base64, mime = _image_to_base64(image)
        if not image_base64 or not mime:
            raise ValueError(f'Unsupported image file: {image}')
        return f'data:{mime};base64,{image_base64}'

    @classmethod
    def _format_input_item(cls, item: Union[str, Dict]) -> Union[str, Dict]:
        if isinstance(item, dict) and 'image' in item:
            item = item.copy()
            item['image'] = cls._format_image(item['image'])
        return item

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict:
        if isinstance(input, str):
            input = [input]
        elif isinstance(input, list):
            if len(input) == 0:
                raise ValueError('Input list cannot be empty')
            if any(isinstance(item, list) for item in input):
                raise ValueError('SiliconFlowMultimodalEmbed expects a 1D input list')
            if not all(isinstance(item, (str, dict)) for item in input):
                raise ValueError('Input list must contain strings or dictionaries')
        else:
            raise ValueError('Input must be either a string or a list')
        input = [self._format_input_item(item) for item in input]

        json_data = {
            'input': input,
            'model': self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)
        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> Union[List[List[float]], List[float]]:
        data = response.get('data', [])
        if not data:
            raise ValueError('No data found in response')
        embeddings = [res.get('embedding', []) for res in data]
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings


class SiliconFlowRerank(LazyLLMOnlineRerankModuleBase):
    def __init__(self, embed_url: Optional[str] = None, embed_model_name: Optional[str] = None,
                 api_key: str = None, **kw):
        embed_url = embed_url or 'https://api.siliconflow.cn/v1/rerank'
        embed_model_name = embed_model_name or 'BAAI/bge-reranker-v2-m3'
        super().__init__(embed_url, api_key or self._default_api_key(), embed_model_name, **kw)

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


class SiliconFlowText2Image(LazyLLMOnlineText2ImageModuleBase):
    MODEL_NAME = 'Qwen/Qwen-Image'
    IMAGE_EDITING_MODEL_NAME = 'Qwen/Qwen-Image-Edit-2509'

    def __init__(self, api_key: str = None, model: str = None,
                 url: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        url = url or 'https://api.siliconflow.cn/v1/'
        super().__init__(api_key=api_key or self._default_api_key(),
                         model=model or SiliconFlowText2Image.MODEL_NAME, url=url, return_trace=return_trace, **kwargs)
        self._endpoint = 'images/generations'

    def _get_image_data_from_url(self, url: str, timeout: int = 30) -> bytes:
        '''
        Override parent implementation because SiliconFlow S3 temporary URLs
        may return application/octet-stream instead of image/* content type.
        '''
        self._validate_url_security(url)

        resp = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True
        )
        resp.raise_for_status()
        data = resp.content
        self._validate_image_data(data, url)
        return data

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
            for i, file in enumerate(files):
                reference_image_base64, _ = self._load_images(file)[0]
                reference_image_data = f'data:image/png;base64,{reference_image_base64}'
                if i == 0:
                    payload['image'] = reference_image_data
                elif i > 0:
                    payload[f'image{i + 1}'] = reference_image_data
        result = self._make_request(self._endpoint, payload)
        image_urls = [item['url'] for item in result.get('data', [])]
        if not image_urls:
            raise Exception('No images returned from API')
        image_results = self._load_images(image_urls)
        image_bytes = [data for _, data in image_results]
        if not image_bytes:
            raise Exception('Failed to download any images')
        ai_img_path = os.path.join(config['temp_dir'], 'ai_img')
        file_paths = bytes_to_file(image_bytes, target_dir=ai_img_path)
        return encode_query_with_filepaths(None, file_paths)


class SiliconFlowSTT(LazyLLMOnlineSTTModuleBase):
    MODEL_NAME = 'FunAudioLLM/SenseVoiceSmall'

    def __init__(self, api_key: str = None, model: str = None, model_name: str = None,
                 base_url: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        base_url = base_url or 'https://api.siliconflow.cn/v1/'
        resolved_model = model or model_name or SiliconFlowSTT.MODEL_NAME
        super().__init__(api_key=api_key or self._default_api_key(),
                         model=resolved_model, return_trace=return_trace, url=base_url, **kwargs)
        self._endpoint = 'audio/transcriptions'

    def _resolve_audio_path(self, input: str = None, files: List[str] = None) -> str:
        if files and len(files) > 1:
            raise ValueError('SiliconFlowSTT only supports one audio file at a time')
        if files and len(files) == 1:
            return files[0]
        if input and os.path.isfile(input):
            return input
        raise ValueError('SiliconFlowSTT requires a local audio file path')

    def _transcribe(self, file_path: str, model: str, base_url: Optional[str] = None) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found')
        url = urljoin(base_url or self._base_url, self._endpoint)
        filename = os.path.basename(file_path)
        with open(file_path, 'rb') as audio_file:
            multipart = {
                'file': (filename, audio_file),
                'model': (None, model),
            }
            response = requests.post(url, headers=self._get_empty_header(), files=multipart, timeout=180)
        response.raise_for_status()
        return response.json().get('text', '')

    def _forward(self, input: str = None, files: List[str] = None, url: str = None, model: str = None, **kwargs):
        file_path = self._resolve_audio_path(input=input, files=files)
        return self._transcribe(file_path, model or self._model_name, base_url=url)


class SiliconFlowTTS(LazyLLMOnlineTTSModuleBase):
    MODEL_NAME = 'fnlp/MOSS-TTSD-v0.5'

    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        base_url = base_url or 'https://api.siliconflow.cn/v1/'
        super().__init__(api_key=api_key or self._default_api_key(),
                         model_name=model_name or SiliconFlowTTS.MODEL_NAME,
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
