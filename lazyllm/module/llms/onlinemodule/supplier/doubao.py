import lazyllm
from typing import Dict, List, Union
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase,
    LazyLLMOnlineMultimodalEmbedModuleBase, LazyLLMOnlineText2ImageModuleBase
)
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime
from lazyllm import LOG


class DoubaoChat(OnlineChatModuleBase):
    MODEL_NAME = 'doubao-1-5-pro-32k-250115'
    VLM_MODEL_PREFIX = ['doubao-seed-1-6-vision', 'doubao-1-5-ui-tars']

    def __init__(self, model: str = None, base_url: str = 'https://ark.cn-beijing.volces.com/api/v3/',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(api_key=api_key or lazyllm.config['doubao_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoChat.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ('You are Doubao, an AI assistant. Your task is to provide appropriate responses '
                'and support to user\'s questions and requests.')

    def _validate_api_key(self):
        '''Validate API Key by sending a minimal request'''
        try:
            # Doubao (Volcano Engine) validates API key using a minimal chat request
            data = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1  # Only generate 1 token for validation
            }
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


class DoubaoEmbed(LazyLLMOnlineEmbedModuleBase):
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings',
                 embed_model_name: str = 'doubao-embedding-text-240715',
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        super().__init__(embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name,
                         batch_size=batch_size, **kw)


class DoubaoMultimodalEmbed(LazyLLMOnlineMultimodalEmbedModuleBase):
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal',
                 embed_model_name: str = 'doubao-embedding-vision-241215',
                 api_key: str = None):
        super().__init__(embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name)

    def _encapsulated_data(self, input: Union[List, str], **kwargs) -> Dict[str, str]:
        if isinstance(input, str):
            input = [{'text': input}]
        elif isinstance(input, list):
            # Validate input format, at most 1 text segment + 1 image
            if len(input) == 0:
                raise ValueError('Input list cannot be empty')
            if len(input) > 2:
                raise ValueError('Input list must contain at most 2 items (1 text and/or 1 image)')
        else:
            raise ValueError('Input must be either a string or a list of dictionaries')

        json_data = {
            'input': input,
            'model': self._embed_model_name
        }
        if len(kwargs) > 0:
            json_data.update(kwargs)

        return json_data

    def _parse_response(self, response: Dict, input: Union[List, str]) -> List[float]:
        # Doubao multimodal embedding returns a single fused embedding
        return response['data']['embedding']


class DoubaoMultiModal():
    def __init__(self, api_key: str = None, url: str = ''):
        api_key = api_key or lazyllm.config['doubao_api_key']
        self._client = volcenginesdkarkruntime.Ark(base_url=url, api_key=api_key)


class DoubaoText2Image(LazyLLMOnlineText2ImageModuleBase, DoubaoMultiModal):
    MODEL_NAME = 'doubao-seedream-4-0-250828'
    IMAGE_EDITING_MODEL_NAME = 'doubao-seedream-4-0-250828'

    def __init__(self, api_key: str = None, model: str = None, url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        super().__init__(model=model, api_key=api_key,
                         return_trace=return_trace, url=url, **kwargs)
        DoubaoMultiModal.__init__(self, api_key=api_key, url=url)

    def _forward(self, input: str = None, files: List[str] = None, n: int = 1, size: str = '1024x1024', seed: int = -1,
                 guidance_scale: float = 2.5, watermark: bool = True, model: str = None, url: str = None, **kwargs):
        has_ref_image = files is not None and len(files) > 0
        if self._type == LLMType.IMAGE_EDITING and not has_ref_image:
            LOG.warning(
                f'Image editing is enabled for model {self._model_name}, but no image file was provided. '
                f'Please provide an image file via the "files" parameter.'
            )
        if self._type != LLMType.IMAGE_EDITING and has_ref_image:
            msg = str(f'Image file was provided, but image editing is not enabled for model {self._model_name}. Please '
                      f'use default image-editing model {self.IMAGE_EDITING_MODEL_NAME} or other image-editing model.')
            raise ValueError(msg)

        if has_ref_image:
            image_results = self._load_images(files)
            contents = [f'data:image/png;base64,{base64_str}' for base64_str, _ in image_results]
        api_params = {
            'model': model,
            'prompt': input,
            'size': size,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'watermark': watermark,
            **kwargs
        }
        if has_ref_image:
            api_params['image'] = contents
            if n > 1:
                api_params['sequential_image_generation'] = 'auto'
                max_images = min(n, 15)
                sigo = volcenginesdkarkruntime.types.images.SequentialImageGenerationOptions
                api_params['sequential_image_generation_options'] = sigo(max_images=max_images)
        imagesResponse = self._client.images.generate(**api_params)
        image_contents = [requests.get(result.url).content for result in imagesResponse.data]
        return encode_query_with_filepaths(None, bytes_to_file(image_contents))
