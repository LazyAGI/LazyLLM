import lazyllm
from typing import Dict, List, Union, Optional
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime
from lazyllm import LOG


class DoubaoModule(OnlineChatModuleBase):
    MODEL_NAME = 'doubao-1-5-pro-32k-250115'
    VLM_MODEL_PREFIX = ['doubao-seed-1-6-vision', 'doubao-1-5-ui-tars']

    def __init__(self, model: str = None, base_url: str = 'https://ark.cn-beijing.volces.com/api/v3/',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        super().__init__(model_series='DOUBAO', api_key=api_key or lazyllm.config['doubao_api_key'], base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoModule.MODEL_NAME,
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

class DoubaoEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings',
                 embed_model_name: str = 'doubao-embedding-text-240715',
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        super().__init__('DOUBAO', embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name,
                         batch_size=batch_size, **kw)


class DoubaoMultimodalEmbedding(OnlineEmbeddingModuleBase):
    def __init__(self,
                 embed_url: str = 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal',
                 embed_model_name: str = 'doubao-embedding-vision-241215',
                 api_key: str = None):
        super().__init__('DOUBAO', embed_url, api_key or lazyllm.config['doubao_api_key'], embed_model_name)

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


class DoubaoMultiModal(OnlineMultiModalBase):
    def __init__(self, api_key: str = None, model: str = None, base_url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        api_key = api_key or lazyllm.config['doubao_api_key']
        OnlineMultiModalBase.__init__(self, model_series='DOUBAO', model=model, api_key=api_key,
                                      return_trace=return_trace, base_url=base_url, **kwargs)
        self._client = volcenginesdkarkruntime.Ark(base_url=base_url, api_key=api_key)

class DoubaoTextToImageModule(DoubaoMultiModal):
    MODEL_NAME = 'doubao-seedream-3-0-t2i-250415'
    IMAGE_EDITING_MODEL_NAME = 'doubao-seedream-4-0-250828'    
    def __init__(self, api_key: str = None, model: str = None, return_trace: bool = False, **kwargs):
        DoubaoMultiModal.__init__(self, api_key=api_key, model=model
                            or DoubaoTextToImageModule.MODEL_NAME or DoubaoTextToImageModule.IMAGE_EDITING_MODEL_NAME
                            or lazyllm.config['doubao_text2image_model_name'],
                            return_trace=return_trace, **kwargs)

    def _forward(self, input: str = None, files: List[str] = None, size: str = '1024x1024', seed: int = -1, 
                 guidance_scale: float = 2.5, watermark: bool = True, model: str = None, url: str = None, **kwargs):
        has_ref_image = files is not None and len(files) > 0
        reference_image_data_url=None
        if self._type=='image_editing' and not has_ref_image:
            LOG.warning(
                f'Image editing is enabled for model {self._model_name}, but no image file was provided. '
                f'Please provide an image file via the "files" parameter.'
            )
        if not self._type=='image_editing' and has_ref_image:
            LOG.error(
                f'Image file was provided, but image editing is not enabled for model {self._model_name}. '
                f'Please use default image editing model {self.IMAGE_EDITING_MODEL_NAME}'
            )
            raise ValueError()        
        if has_ref_image:
            reference_image_base64, _ = self._load_image(files[0])
            reference_image_data_url = f"data:image/png;base64,{reference_image_base64}"

        try:
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
                api_params['image'] = reference_image_data_url
            imagesResponse = self._client.images.generate(**api_params)
            image_contents = [requests.get(result.url).content for result in imagesResponse.data]
            return encode_query_with_filepaths(None, bytes_to_file(image_contents))
        except Exception as e:
            lazyllm.LOG.error(f'Image generation/editing request failed: {str(e)}')
            raise
