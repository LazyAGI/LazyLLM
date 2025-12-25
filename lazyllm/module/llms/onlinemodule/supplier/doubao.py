import lazyllm
from typing import Dict, List, Union
from ..base import OnlineChatModuleBase, OnlineEmbeddingModuleBase, OnlineMultiModalBase
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime

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
    def __init__(self, api_key: str = None, model_name: str = None, base_url='https://ark.cn-beijing.volces.com/api/v3',
                 return_trace: bool = False, **kwargs):
        api_key = api_key or lazyllm.config['doubao_api_key']
        OnlineMultiModalBase.__init__(self, model_series='DOUBAO', model_name=model_name, api_key=api_key,
                                      return_trace=return_trace, **kwargs)
        self._client = volcenginesdkarkruntime.Ark(base_url=base_url, api_key=api_key)


class DoubaoTextToImageModule(DoubaoMultiModal):
    MODEL_NAME = 'doubao-seedream-3-0-t2i-250415'

    def __init__(self, api_key: str = None, model_name: str = None, return_trace: bool = False, **kwargs):
        DoubaoMultiModal.__init__(self, api_key=api_key, model_name=model_name
                                  or DoubaoTextToImageModule.MODEL_NAME
                                  or lazyllm.config['doubao_text2image_model_name'],
                                  return_trace=return_trace, **kwargs)

    def _forward(self, input: str = None, size: str = '1024x1024', seed: int = -1, guidance_scale: float = 2.5,
                 watermark: bool = True, **kwargs):
        imagesResponse = self._client.images.generate(
            model=self._model_name,
            prompt=input,
            size=size,
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=watermark,
            **kwargs
        )
        return encode_query_with_filepaths(None, bytes_to_file([requests.get(result.url).content
                                                                for result in imagesResponse.data]))


class DoubaoTextToImageEditModule(DoubaoMultiModal):
    """ByteDance Doubao Text-to-Image Edit module supporting image editing with reference images.

        Based on ByteDance Doubao multimodal model's text-to-image editing functionality, 
        inherits from DoubaoMultiModal, providing high-quality image editing capability with reference images.

        Args:
            api_key (str, optional): Doubao API key, defaults to None.
            model_name (str, optional): Model name, defaults to "doubao-seedream-4-5-251128".
            return_trace (bool, optional): Whether to return trace information, defaults to False.
            **kwargs: Other parameters passed to parent class.
    """
    MODEL_NAME = 'doubao-seedream-4-5-251128'

    def __init__(self, api_key: str = None, model_name: str = None, return_trace: bool = False, **kwargs):
        DoubaoMultiModal.__init__(self, api_key=api_key, model_name=model_name
                                  or DoubaoTextToImageEditModule.MODEL_NAME
                                  or lazyllm.config.get('doubao_text2image_model_name'),
                                  return_trace=return_trace, **kwargs)

    def _load_image_as_base64(self, image_path: str) -> str:
        """
        Load image from file path or URL and convert to base64 string.
        
        Args:
            image_path: Local file path or HTTP(S) URL of the image
            
        Returns:
            Base64 encoded image string
        """
        import base64
        from pathlib import Path
        
        try:
            if image_path.startswith('http://') or image_path.startswith('https://'):
                # Download image from URL
                response = requests.get(image_path, timeout=30)
                response.raise_for_status()
                image_data = response.content
            else:
                # Load from local file
                if not Path(image_path).exists():
                    raise FileNotFoundError(f'Image file not found: {image_path}')
                with open(image_path, 'rb') as f:
                    image_data = f.read()
            
            # Convert to base64
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            lazyllm.LOG.error(f'Failed to load image from {image_path}: {str(e)}')
            raise

    def _forward(self, input: str = None, files: List[str] = None, size: str = '2K',
                 seed: int = -1, guidance_scale: float = 2.5, watermark: bool = True, **kwargs):
        """
        Forward method for image editing with reference images.
        
        Args:
            input (str): Text prompt for editing instructions.
            files (List[str]): List of reference image file paths or URLs (must provide at least one).
            size (str): Output image size (e.g., '4096x2160', '2048x2048'). Defaults to '4096x2160'.
            seed (int): Random seed for generation reproducibility. Defaults to -1 (random).
            guidance_scale (float): Guidance scale for generation quality. Defaults to 2.5.
            watermark (bool): Whether to add watermark to generated images. Defaults to True.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            str: Encoded file paths containing the edited images.
            
        Raises:
            ValueError: If no reference images are provided.
        """
        if not files or len(files) == 0:
            raise ValueError('DoubaoTextToImageEditModule requires at least one reference image in files parameter')
        
        # Load the first image as reference
        reference_image_base64 = self._load_image_as_base64(files[0])
        
        # Build request parameters with reference image
        # Note: Doubao API expects base64 image data in the format: data:image/png;base64,{base64_data}
        reference_image_data_url = f"data:image/png;base64,{reference_image_base64}"
        
        try:
            # Make API request using the client
            imagesResponse = self._client.images.generate(
                model=self._model_name,
                prompt=input,
                # image参数格式：base64
                image=reference_image_data_url,  # Reference image for editing
                size=size,
                seed=seed,
                guidance_scale=guidance_scale,
                watermark=watermark,
                
                **kwargs
            )
            
            # Download images and convert to files
            image_contents = [requests.get(result.url).content for result in imagesResponse.data]
            
            return encode_query_with_filepaths(None, bytes_to_file(image_contents))
        except Exception as e:
            lazyllm.LOG.error(f'Image editing request failed: {str(e)}')
            raise