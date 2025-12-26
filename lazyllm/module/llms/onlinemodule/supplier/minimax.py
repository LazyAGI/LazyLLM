import base64
import requests
import lazyllm
from typing import Any, Dict, List
from urllib.parse import urljoin
from ..base import OnlineChatModuleBase, OnlineMultiModalBase
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase


class MinimaxModule(OnlineChatModuleBase, FileHandlerBase):

    def __init__(self, base_url: str = 'https://api.minimaxi.com/v1/', model: str = 'MiniMax-M2',
                 api_key: str = None, stream: bool = True, return_trace: bool = False, **kwargs):
        OnlineChatModuleBase.__init__(self, model_series='MINIMAX',
                                      api_key=api_key or lazyllm.config['minimax_api_key'],
                                      base_url=base_url, model_name=model, stream=stream,
                                      return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)
        if stream:
            self._model_optional_params['stream'] = True

    def _get_system_prompt(self):
        return 'You are an intelligent assistant provided by Minimax. You are a helpful assistant.'

    def _convert_msg_format(self, msg: Dict[str, Any]):
        '''Convert the reasoning_details in output to reasoning_content field in message'''
        choices = msg.get('choices')
        if not isinstance(choices, list):
            return msg

        for choice in choices:
            message = choice.get('message') or choice.get('delta') or {}
            details = message.get('reasoning_details')
            if not details:
                continue

            text = ''
            if isinstance(details, dict):
                text = (details.get('text') or '').strip()
            elif isinstance(details, list):
                texts = [
                    (item.get('text') or '').strip()
                    for item in details
                    if isinstance(item, dict) and item.get('text')
                ]
                text = '\n'.join(filter(None, texts)).strip()

            if text:
                message['reasoning_content'] = text

        return msg

    def _validate_api_key(self):
        '''Validate API Key by sending a minimal chat request'''
        try:
            data = {
                'model': self._model_name,
                'messages': [{'role': 'user', 'content': 'test'}],
                'max_tokens': 1
            }
            response = requests.post(self._chat_url, headers=self._header, json=data, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

class MinimaxTextToImageModule(OnlineMultiModalBase):
    """Minimax text-to-image module, supporting both text-to-image and image-to-image (edit) generation.

    This class supports two modes:
    1. Text-to-Image: Generate images from text prompts only
    2. Image-to-Image: Edit/generate images based on reference images and text prompts

    Args:
        api_key (str, optional): API key, defaults to lazyllm.config['minimax_api_key']
        model (str, optional): Model name to use (preferred, compatible with OnlineMultiModalModule)
        model_name (str, optional): Model name (alternative, for backward compatibility)
        base_url (str, optional): Base API URL, defaults to "https://api.minimaxi.com/v1/"
        image_edit (bool, optional): Whether to enable image editing mode. If None, auto-detect based on model name.
        return_trace (bool, optional): Whether to return trace information, defaults to False
        **kwargs: Additional optional parameters passed to the parent classes
    """
    
    IMAGE_MODEL = [
        'image-01',
        # 在这里补充其他仅支持文生图的 minimax 模型
    ]

    # 支持图生图 / 图文生图的模型
    IMAGE_MODEL_EDIT = [
        'image-01',
        # 在这里补充支持图生图的 minimax 模型
    ]

    MODEL_NAME = 'image-01'

    def __init__(self, api_key: str = None, model: str = None,
                 base_url: str = 'https://api.minimaxi.com/v1/',
                 image_edit: bool = None, return_trace: bool = False, **kwargs):
        """
        Initialize MinimaxTextToImageModule.
        """
        # 统一确定最终模型名：优先用 model，其次 model_name，最后默认/配置
        final_model_name = (
            model
            or MinimaxTextToImageModule.MODEL_NAME
            or lazyllm.config.get('minimax_text2image_model_name')
        )

        # 如果没显式指定 image_edit，则根据模型名自动判断是否支持图生图
        if image_edit is None:
            image_edit = final_model_name in MinimaxTextToImageModule.IMAGE_MODEL_EDIT

        # 记录能力
        self._supports_image_edit = image_edit

        # 如果强行开启 image_edit 但模型在纯文生图列表里，打 warning
        if image_edit and final_model_name in MinimaxTextToImageModule.IMAGE_MODEL:
            lazyllm.LOG.warning(
                f'Model {final_model_name} may not support image editing. '
                f'Please use models from IMAGE_MODEL_EDIT list: {MinimaxTextToImageModule.IMAGE_MODEL_EDIT}'
            )

        OnlineMultiModalBase.__init__(
            self,
            model_series='MINIMAX',
            api_key=api_key or lazyllm.config['minimax_api_key'],
            model_name=final_model_name,
            return_trace=return_trace,
            **kwargs,
        )
        self._base_url = base_url
        self._endpoint = 'image_generation'

    def _make_request(self, endpoint: str, payload: dict, timeout: int = 180) -> dict:
        url = urljoin(self._base_url, endpoint)
        response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        base_resp = result.get('base_resp')
        if base_resp and base_resp.get('status_code') not in (None, 0):
            raise Exception(f"Minimax API error {base_resp.get('status_code')}: {base_resp.get('status_msg')}")
        return result


    def _forward(
        self,
        input: str = None,
        files: list[str] | None = None,
        style: dict | None = None,
        aspect_ratio: str | None = None,
        width: int | None = None,
        height: int | None = None,
        response_format: str = 'url',
        seed: int | None = None,
        n: int = 1,
        prompt_optimizer: bool = False,
        aigc_watermark: bool = False,
        **kwargs,
    ):
        
        use_image_edit = bool(files)

        if use_image_edit:
            if not self._supports_image_edit:
                raise ValueError(
                    f'Model {self._model_name} does not support image editing. '
                    f'Please use a model from IMAGE_MODEL_EDIT list: {self.IMAGE_MODEL_EDIT}'
                )
            
            ref_b64 = self._load_image_as_base64(files[0])

        payload: dict = {
            'model': self._model_name,
            'prompt': input,
            'response_format': response_format or 'url',
            'n': n,
            'prompt_optimizer': prompt_optimizer,
            'aigc_watermark': aigc_watermark,
        }

        if style is not None:
            payload['style'] = style
        if aspect_ratio is not None:
            payload['aspect_ratio'] = aspect_ratio
        if width is not None and height is not None:
            payload['width'] = width
            payload['height'] = height
        if seed is not None:
            payload['seed'] = seed

        # 图生图模式下，追加参考图字段（字段名需按 minimax 文档调整）
        if use_image_edit:
            # 示例：假设 minimax 使用 'image' 字段 + data URL
            payload['image'] = f"data:image/png;base64,{ref_b64}"
            # 或者，如果文档要求另一个字段，可以改成：
            # payload['ref_image'] = ref_b64

        payload.update(kwargs)

        result = self._make_request(self._endpoint, payload)
        data = result.get('data') or {}

        image_bytes: list[bytes] = []

        if payload['response_format'] == 'base64':
            images_base64 = data.get('image_base64') or []
            if not images_base64:
                raise Exception('Minimax API did not return any base64 images.')
            for image_b64 in images_base64:
                image_bytes.append(base64.b64decode(image_b64))
        elif payload['response_format'] == 'url':
            image_urls = data.get('image_urls') or []
            if not image_urls:
                raise Exception('Minimax API did not return any image URLs.')
            for image_url in image_urls:
                img_response = requests.get(image_url, timeout=180)
                if img_response.status_code != 200:
                    raise Exception(f'Failed to download image from {image_url}')
                image_bytes.append(img_response.content)
        else:
            raise ValueError(f"Unsupported response format: {payload['response_format']}")

        file_paths = bytes_to_file(image_bytes)
        response = encode_query_with_filepaths(None, file_paths)

        return response


class MinimaxTTSModule(OnlineMultiModalBase):
    MODEL_NAME = 'speech-2.6-hd'

    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: str = 'https://api.minimaxi.com/v1/',
                 return_trace: bool = False, **kwargs):
        if kwargs.pop('stream', False):
            raise ValueError('MinimaxTTSModule does not support streaming output, please set stream to False')
        OnlineMultiModalBase.__init__(self, model_series='MINIMAX',
                                      api_key=api_key or lazyllm.config['minimax_api_key'],
                                      model_name=model_name or MinimaxTTSModule.MODEL_NAME,
                                      return_trace=return_trace, **kwargs)
        self._endpoint = 't2a_v2'
        self._base_url = base_url

    def _make_request(self, endpoint: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
        url = urljoin(self._base_url, endpoint)
        try:
            response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            base_resp = result.get('base_resp')
            if base_resp and base_resp.get('status_code') not in (None, 0):
                raise Exception(f"Minimax API error {base_resp.get('status_code')}: {base_resp.get('status_msg')}")
            return result
        except Exception as e:
            lazyllm.LOG.error(f'API request failed: {str(e)}')
            raise

    def _forward(self, input: str = None, stream: bool = False, output_format: str = 'hex',
                 voice_setting: Dict[str, Any] = None, audio_setting: Dict[str, Any] = None,
                 pronunciation_dict: Dict[str, Any] = None, timbre_weights: List[Dict[str, Any]] = None,
                 language_boost: str = None, voice_modify: Dict[str, Any] = None,
                 subtitle_enable: bool = False, aigc_watermark: bool = False,
                 stream_options: Dict[str, Any] = None, out_path: str = None, **kwargs):
        if stream:
            raise ValueError('MinimaxTTSModule does not support streaming output, please set stream to False')
        voice_setting = voice_setting or {}
        voice_setting.setdefault('voice_id', 'male-qn-qingse')
        payload: Dict[str, Any] = {
            'model': self._model_name,
            'text': input,
            'stream': stream,
            'output_format': output_format,
            'voice_setting': voice_setting,
        }
        optional_params = {
            'audio_setting': audio_setting,
            'pronunciation_dict': pronunciation_dict,
            'timbre_weights': timbre_weights,
            'language_boost': language_boost,
            'voice_modify': voice_modify,
            'stream_options': stream_options,
            'subtitle_enable': subtitle_enable,
            'aigc_watermark': aigc_watermark,
        }
        payload.update({k: v for k, v in optional_params.items() if v is not None})
        payload.update(kwargs)
        result = self._make_request(self._endpoint, payload, timeout=180)
        data = result.get('data') or {}
        audio_data = data.get('audio')
        if not audio_data:
            raise Exception('Minimax API did not return any audio data.')

        if output_format == 'url':
            audio_response = requests.get(audio_data, timeout=180)
            audio_response.raise_for_status()
            audio_content = audio_response.content
        elif output_format == 'hex':
            audio_content = bytes.fromhex(audio_data)
        else:
            raise ValueError(f'Unsupported output_format: {output_format}. Supported formats are "url" and "hex".')

        file_path = bytes_to_file([audio_content])[0]
        if out_path:
            with open(file_path, 'rb') as src, open(out_path, 'wb') as dst:
                dst.write(src.read())
            file_path = out_path
        result_encoded = encode_query_with_filepaths(None, [file_path])
        return result_encoded
