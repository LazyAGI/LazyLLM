import time
import lazyllm
from typing import Dict, List, Union, Optional
from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    OnlineChatModuleBase, LazyLLMOnlineEmbedModuleBase,
    LazyLLMOnlineMultimodalEmbedModuleBase, LazyLLMOnlineText2ImageModuleBase,
    LazyLLMOnlineText2VideoModuleBase,
)
import requests
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from lazyllm.thirdparty import volcenginesdkarkruntime
from lazyllm import LOG


class DoubaoChat(OnlineChatModuleBase):
    MODEL_NAME = 'doubao-1-5-pro-32k-250115'
    VLM_MODEL_PREFIX = ['doubao-seed-1-6-vision', 'doubao-1-5-ui-tars']

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        base_url = base_url or 'https://ark.cn-beijing.volces.com/api/v3/'
        super().__init__(api_key=api_key or self._default_api_key(), base_url=base_url,
                         model_name=model or lazyllm.config['doubao_model_name'] or DoubaoChat.MODEL_NAME,
                         stream=stream, return_trace=return_trace, **kwargs)

    def _get_system_prompt(self):
        return ('You are Doubao, an AI assistant. Your task is to provide appropriate responses '
                'and support to user\'s questions and requests.')


class DoubaoEmbed(LazyLLMOnlineEmbedModuleBase):
    def __init__(self,
                 embed_url: Optional[str] = None,
                 embed_model_name: Optional[str] = None,
                 api_key: str = None,
                 batch_size: int = 16,
                 **kw):
        embed_url = embed_url or 'https://ark.cn-beijing.volces.com/api/v3/embeddings'
        embed_model_name = embed_model_name or 'doubao-embedding-text-240715'
        super().__init__(embed_url, api_key or self._default_api_key(), embed_model_name,
                         batch_size=batch_size, **kw)


class DoubaoMultimodalEmbed(LazyLLMOnlineMultimodalEmbedModuleBase):
    def __init__(self,
                 embed_url: Optional[str] = None,
                 embed_model_name: str = None,
                 api_key: str = None):
        embed_url = embed_url or 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal'
        embed_model_name = (embed_model_name or lazyllm.config['doubao_multimodal_embed_model_name']
                            or 'doubao-embedding-vision-241215')
        super().__init__(embed_url, api_key or self._default_api_key(), embed_model_name)

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


class DoubaoText2Image(LazyLLMOnlineText2ImageModuleBase):
    # Default to Seedream 4.0 (stable production); override via model= or config.
    MODEL_NAME = 'doubao-seedream-4-0-250828'
    IMAGE_EDITING_MODEL_NAME = 'doubao-seedream-4-0-250828'
    MODEL_NAMES = (
        'doubao-seedream-5-0-pro-260628',  # Seedream 5.0 Pro
        'doubao-seedream-5-0-260128',      # Seedream 5.0 Lite
        'doubao-seedream-4-5-251128',      # Seedream 4.5
        'doubao-seedream-4-0-250828',      # Seedream 4.0
        'doubao-seedream-3-0-t2i-250415',  # Seedream 3.0
    )

    def __init__(self, api_key: str = None, model: Optional[str] = None, url: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        url = url or 'https://ark.cn-beijing.volces.com/api/v3'
        resolved_model = model or lazyllm.config['doubao_text2image_model_name'] or DoubaoText2Image.MODEL_NAME
        super().__init__(model=resolved_model, api_key=api_key or self._default_api_key(),
                         return_trace=return_trace, url=url, **kwargs)

    def _ark_client(self, base_url=None):
        return volcenginesdkarkruntime.Ark(base_url=(base_url or self._base_url), api_key=self._api_key)

    def _forward(self, input: str = None, files: List[str] = None, n: int = 1, size: str = '1024x1024', seed: int = -1,
                 guidance_scale: float = 2.5, watermark: bool = True, model: str = None, url: str = None, **kwargs):
        # kwargs may include LazyLLM framework fields (stream_output / priority); ignore them.
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

        # Only pass fields required/supported by Ark images.generate.
        api_params = {
            'model': model or self._model_name,
            'prompt': input,
            'size': size,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'watermark': watermark,
            'response_format': 'url',
            'stream': False,
        }
        if has_ref_image:
            image_results = self._load_images(files)
            contents = [f'data:image/png;base64,{base64_str}' for base64_str, _ in image_results]
            api_params['image'] = contents
            if n > 1:
                api_params['sequential_image_generation'] = 'auto'
                max_images = min(n, 15)
                sigo = volcenginesdkarkruntime.types.images.SequentialImageGenerationOptions
                api_params['sequential_image_generation_options'] = sigo(max_images=max_images)
            else:
                api_params['sequential_image_generation'] = 'disabled'
        imagesResponse = self._ark_client(base_url=url).images.generate(**api_params)
        image_contents = [requests.get(result.url).content for result in imagesResponse.data]
        return encode_query_with_filepaths(None, bytes_to_file(image_contents))


class DoubaoText2Video(LazyLLMOnlineText2VideoModuleBase):
    # Default to the cheapest free model for traffic-saving usage.
    MODEL_NAME = 'doubao-seedance-1-0-pro-fast-251015'
    MODEL_NAMES = (
        'doubao-seedance-2-0',
        'doubao-seedance-2-0-fast',
        'doubao-seedance-1-5-pro-251215',
        'doubao-seedance-1-0-pro-250528',
        'doubao-seedance-1-0-pro-fast-251015',
        'doubao-seedance-2-0-mini',
    )
    # 2,000,000 tokens
    FREE_MODEL_NAMES = (
        'doubao-seedance-1-5-pro-251215',
        'doubao-seedance-1-0-pro-250528',
        'doubao-seedance-1-0-pro-fast-251015',
    )

    def __init__(self, api_key: str = None, model: Optional[str] = None, url: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        url = url or 'https://ark.cn-beijing.volces.com/api/v3'
        resolved_model = model or lazyllm.config['doubao_text2video_model_name'] or DoubaoText2Video.MODEL_NAME
        super().__init__(model=resolved_model, api_key=api_key or self._default_api_key(),
                         return_trace=return_trace, url=url, **kwargs)

    def _ark_client(self, base_url=None):
        return volcenginesdkarkruntime.Ark(base_url=(base_url or self._base_url), api_key=self._api_key)

    def _build_content(self, input: str, files: List[str] = None, resolution: str = '480p',
                       duration: int = 2, ratio: str = '16:9', watermark: bool = True,
                       camerafixed: bool = False) -> List[Dict]:
        text = (f'{input} --resolution {resolution} --duration {duration} '
                f'--ratio {ratio} --camerafixed {str(camerafixed).lower()} '
                f'--watermark {str(watermark).lower()}')
        content = [{'type': 'text', 'text': text}]
        if files:
            for file in files:
                if file.startswith(('http://', 'https://', 'data:')):
                    image_url = file
                else:
                    b64, _ = self._load_images(file)[0]
                    image_url = f'data:image/png;base64,{b64}'
                content.append({'type': 'image_url', 'image_url': {'url': image_url}})
        return content

    def _forward(self, input: str = None, files: List[str] = None, resolution: str = '480p',
                 duration: int = 2, ratio: str = '16:9', watermark: bool = True,
                 camerafixed: bool = False, poll_interval: float = 3.0,
                 model: str = None, url: str = None, **kwargs):
        # kwargs may include LazyLLM framework fields (stream_output / priority); ignore them.
        content = self._build_content(
            input=input, files=files, resolution=resolution, duration=duration,
            ratio=ratio, watermark=watermark, camerafixed=camerafixed)
        client = self._ark_client(base_url=url)
        # Only pass fields required by Ark content_generation.tasks.create.
        create_result = client.content_generation.tasks.create(
            model=model or self._model_name,
            content=content,
        )
        task_id = create_result.id
        while True:
            get_result = client.content_generation.tasks.get(task_id=task_id)
            status = get_result.status
            if status == 'succeeded':
                video_url = get_result.content.video_url
                video_bytes = requests.get(video_url, timeout=180).content
                return encode_query_with_filepaths(None, bytes_to_file([video_bytes]))
            if status == 'failed':
                raise Exception(f'Doubao text2video failed: {getattr(get_result, "error", None)}')
            time.sleep(poll_interval)
