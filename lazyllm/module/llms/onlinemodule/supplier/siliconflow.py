import requests
import lazyllm
from typing import Tuple, List, Dict, Union
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
    """
    SiliconFlow Text-to-Image module supporting both text-to-image and image-to-image generation.
    
    This class supports two modes:
    1. Text-to-Image: Generate images from text prompts only
    2. Image-to-Image: Edit/generate images based on reference images and text prompts
    
    Args:
        api_key (str, optional): API key, defaults to configured siliconflow_api_key
        model (str, optional): Model name (compatible with OnlineMultiModalModule)
        model_name (str, optional): Model name (alternative parameter name)
        base_url (str, optional): Base API URL, defaults to "https://api.siliconflow.cn/v1/"
        image_edit (bool, optional): Whether to enable image editing mode. If None, auto-detect based on model name.
        return_trace (bool, optional): Whether to return trace information, defaults to False
        **kwargs: Other model parameters
    """
    # Models that only support text-to-image generation
    IMAGE_MODEL = [
        'Qwen/Qwen-Image',
        # Add other text-to-image only models here
    ]
    
    # Models that support both text-to-image and image-to-image editing
    IMAGE_MODEL_EDIT = [
        'Qwen/Qwen-Image-Edit-2509',
        'Qwen/Qwen-Image-Edit'
        # Add other image editing capable models here
    ]
    
    MODEL_NAME = 'Qwen/Qwen-Image'

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 image_edit: bool = None, return_trace: bool = False, **kwargs):
        """
        Initialize SiliconFlowTextToImageModule.
        
        Args:
            api_key: API key for SiliconFlow service
            model: Model name to use (preferred, compatible with OnlineMultiModalModule)
            model_name: Model name to use (alternative, for backward compatibility)
            base_url: Base API URL
            image_edit: Whether to enable image editing mode. 
                       If None, auto-detect based on model name.
            return_trace: Whether to return trace information
            **kwargs: Additional parameters
        """
        # Determine model name (优先使用 model，兼容 OnlineMultiModalModule)
        final_model_name = (model 
                            or SiliconFlowTextToImageModule.MODEL_NAME
                            or lazyllm.config.get('siliconflow_text2image_model_name'))
        
        # Auto-detect image_edit capability if not explicitly specified
        if image_edit is None:
            image_edit = final_model_name in SiliconFlowTextToImageModule.IMAGE_MODEL_EDIT
        
        # Store image_edit capability
        self._supports_image_edit = image_edit
        
        # Validate model capability if image_edit is requested
        if image_edit and final_model_name in SiliconFlowTextToImageModule.IMAGE_MODEL:
            lazyllm.LOG.warning(
                f'Model {final_model_name} does not support image editing. '
                f'Use models from IMAGE_MODEL_EDIT list: {SiliconFlowTextToImageModule.IMAGE_MODEL_EDIT}'
            )
        
        OnlineMultiModalBase.__init__(
            self, 
            model_series='SiliconFlow',
            api_key=api_key or lazyllm.config['siliconflow_api_key'],
            model_name=final_model_name,
            return_trace=return_trace, 
            **kwargs
        )
        self._endpoint = 'images/generations'
        self._base_url = base_url

    def _make_request(self, endpoint, payload, timeout=180):
        """Make HTTP request to SiliconFlow API"""
        url = f'{self._base_url}{endpoint}'
        try:
            response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            lazyllm.LOG.error(f'API request failed: {str(e)}')
            raise

    
    def _forward(self, input: str = None, files: List[str] = None, size: str = '1024x1024',
                 num_inference_steps: int = 20, guidance_scale: float = 7.5, **kwargs):
        """
        Forward method for image generation/editing.
        
        Supports two modes:
        1. Text-to-Image: When files is None or empty, generates images from text only
        2. Image-to-Image: When files is provided, edits/generates images based on reference image
        
        Args:
            input (str): Text prompt for image generation/editing instructions.
            files (List[str], optional): List of reference image file paths or URLs for image editing.
            size (str): Output image size. Defaults to '1024x1024' for text-to-image, '1328x1328' for image editing.
            num_inference_steps (int): Number of inference steps (for image editing). Defaults to 20.
            guidance_scale (float): Guidance scale (for image editing). Defaults to 7.5.
            **kwargs: Additional parameters passed to the API.
            
        Returns:
            str: Encoded file paths containing the generated/edited images.
            
        Raises:
            ValueError: If files is provided but model doesn't support image editing.
            ValueError: If files is provided but is empty.
        """
        # Determine if image editing mode is requested
        use_image_edit = files is not None and len(files) > 0
        
        # Validate image editing capability
        if use_image_edit:
            if not self._supports_image_edit:
                raise ValueError(
                    f'Model {self._model_name} does not support image editing. '
                    f'Please use a model from IMAGE_MODEL_EDIT list: {self.IMAGE_MODEL_EDIT}'
                )
            
            # Load reference image and convert to base64
            reference_image_base64 = self._load_image_as_base64(files[0])
            # SiliconFlow API expects base64 image data in the format: data:image/png;base64,{base64_data}
            reference_image_data_url = f"data:image/png;base64,{reference_image_base64}"
            
            # Use '1328x1328' as default size for image editing if not explicitly set
            if size == '1024x1024':  # Default text-to-image size
                size = '1328x1328'
        else:
            reference_image_data_url = None
        
        # Build request payload
        payload = {
            'model': self._model_name,
            'prompt': input,
            **kwargs
        }
        
        # Add image parameter only for image editing mode
        if use_image_edit:
            payload['image'] = reference_image_data_url
            # Note: num_inference_steps and guidance_scale are typically not used by SiliconFlow
            # but can be passed via kwargs if the API supports them
        
        try:
            # Make API request
            result = self._make_request(self._endpoint, payload)
            
            # Extract image URLs from response
            image_urls = [item['url'] for item in result.get('data', [])]
            
            if not image_urls:
                raise Exception('No images returned from API')
            
            # Download and convert images to files
            image_files = []
            for url in image_urls:
                img_response = requests.get(url, timeout=180)
                if img_response.status_code == 200:
                    image_files.append(img_response.content)
                else:
                    lazyllm.LOG.warning(
                        f'Failed to download image from {url}, status code: {img_response.status_code}'
                    )
            
            if not image_files:
                raise Exception('Failed to download any images')
            
            # Convert to file paths and return encoded result
            file_paths = bytes_to_file(image_files)
            return encode_query_with_filepaths(None, file_paths)
            
        except Exception as e:
            lazyllm.LOG.error(f'Error in SiliconFlowTextToImageModule._forward: {str(e)}')
            raise

class SiliconFlowTextToImageEditModule(OnlineMultiModalBase):
    """SiliconFlow Text-to-Image Edit module, inherits from OnlineMultiModalBase.

    Provides text-to-image editing functionality based on SiliconFlow, supports editing images using text prompts and reference images.

    Args:
        api_key (str, optional): API key, defaults to configured siliconflow_api_key
        model_name (str, optional): Model name, defaults to "Qwen/Qwen-image-edit"
        base_url (str, optional): Base API URL, defaults to "https://api.siliconflow.cn/v1/"
        return_trace (bool, optional): Whether to return trace information, defaults to False
        **kwargs: Other model parameters

    Main Parameters for _forward():
        input (str): Text prompt describing the desired edits.
        files (List[str]): Reference image file paths or URLs to be edited.
        size (str): Output image size, defaults to '1024x1024'.
        num_inference_steps (int): Number of inference steps, defaults to 20.
        guidance_scale (float): Guidance scale for generation quality, defaults to 7.5.
        **kwargs: Additional parameters passed to the API.
    """
    MODEL_NAME = 'Qwen/Qwen-Image-Edit-2509'



    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series='SiliconFlow',
                                      api_key=api_key or lazyllm.config['siliconflow_api_key'],  
                                      model_name=model_name or SiliconFlowTextToImageEditModule.MODEL_NAME,
                                      return_trace=return_trace, **kwargs)
        self._endpoint = 'images/generations' # API端点的相对路径。用于指定调用 SiliconFlow API 的具体功能模块。
        self._base_url = base_url
        # self._api_key = api_key or lazyllm.config['siliconflow_api_key']

    def _make_request(self, endpoint, payload, timeout=180):
        """Make HTTP request to SiliconFlow API"""
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

    def _forward(self, input: str = None, files: List[str] = None, size: str = '1328x1328',
                 num_inference_steps: int = 20, guidance_scale: float = 7.5, **kwargs):
        """
        Forward method for image editing.
        
        Args:
            input (str): Text prompt for editing instructions.
            files (List[str]): List of reference image file paths or URLs (must provide at least one).
            size (str): Output image size (e.g., '1024x1024', '512x512').
            num_inference_steps (int): Number of inference steps for better quality.
            guidance_scale (float): Guidance scale (7.5 is typical for good quality).
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            str: Encoded file paths containing the edited images.
            
        Raises:
            ValueError: If no reference images are provided.
        """
        if not files or len(files) == 0:
            raise ValueError('SiliconFlowTextToImageEditModule requires at least one reference image in files parameter')
        
        # Load the first image as reference
        reference_image_base64 = self._load_image_as_base64(files[0])
        # siliconflow API 调用格式
        reference_image_data_url = f"data:image/png;base64,{reference_image_base64}" 
        
        # Build request payload
        payload = {
            'model': self._model_name,
            'prompt': input,
            'image': reference_image_data_url  # Base64 encoded reference image
            
            # 'num_inference_steps': num_inference_steps,
            # 'guidance_scale': guidance_scale
        }
        
        # Add any additional parameters
        payload.update(kwargs)
        
        try:
            # Make API request
            result = self._make_request(self._endpoint, payload)
            
            # Extract image URLs from response
            image_urls = [item['url'] for item in result.get('data', [])]
            
            if not image_urls:
                raise Exception('No images returned from API')
            
            # Download and convert images to files
            image_files = []
            for url in image_urls:
                img_response = requests.get(url, timeout=180)
                # 报错：status_code = 500.
                if img_response.status_code == 200: # 
                    image_files.append(img_response.content)
                else:
                    lazyllm.LOG.warning(f'Failed to download image from {url}, status code: {img_response.status_code}')
            
            if not image_files:
                raise Exception('Failed to download any images')
            
            # Convert to file paths and return encoded result
            file_paths = bytes_to_file(image_files)
            return encode_query_with_filepaths(None, file_paths)
            
        except Exception as e:
            lazyllm.LOG.error(f'Error in SiliconFlowTextToImageEditModule._forward: {str(e)}')
            raise

class SiliconFlowTTSModule(OnlineMultiModalBase):
    MODEL_NAME = 'fnlp/MOSS-TTSD-v0.5'

    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://api.siliconflow.cn/v1/',
                 return_trace: bool = False, **kwargs):
        OnlineMultiModalBase.__init__(self, model_series='SiliconFlow',
                                      api_key=api_key or lazyllm.config['siliconflow_api_key'],
                                      model_name=model_name or SiliconFlowTTSModule.MODEL_NAME,
                                      return_trace=return_trace, **kwargs)
        self._endpoint = 'audio/speech'
        self._base_url = base_url

    def _make_binary_request(self, endpoint, payload, timeout=180):
        url = f'{self._base_url}{endpoint}'
        try:
            response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            lazyllm.LOG.error(f'API request failed: {str(e)}')
            raise

    def _forward(self, input: str = None, response_format: str = 'mp3',
                 sample_rate: int = 44100, speed: float = 1.0,
                 voice: str = None, references=None, out_path: str = None, **kwargs):

        if not voice:
            if self._model_name == 'fnlp/MOSS-TTSD-v0.5':
                voice = 'fnlp/MOSS-TTSD-v0.5:alex'
            elif self._model_name == 'FunAudioLLM/CosyVoice2-0.5B':
                voice = 'FunAudioLLM/CosyVoice2-0.5B:alex'
            else:
                raise ValueError(
                    f'Default voice is only supported for models "fnlp/MOSS-TTSD-v0.5" and '
                    f'"FunAudioLLM/CosyVoice2-0.5B". For model "{self._model_name}", '
                    f'please provide a valid voice parameter.')
        payload = {
            'model': self._model_name,
            'input': input,
            'response_format': response_format,
            'sample_rate': sample_rate,
            'speed': speed,
            'voice': voice
        }

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

        return result
