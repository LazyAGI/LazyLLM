import base64
import requests
import lazyllm
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit

from lazyllm.components.utils.downloader.model_downloader import LLMType
from ..base import (
    ModelFailureCode,
    OnlineChatModuleBase,
    LazyLLMOnlineText2ImageModuleBase,
    LazyLLMOnlineTTSModuleBase,
)
from ..base.provider_response import _OpenAICompatibleResponseParser
from lazyllm.components.formatter import encode_query_with_filepaths
from lazyllm.components.utils.file_operate import bytes_to_file
from ..fileHandler import FileHandlerBase


class _MinimaxResponseParser(_OpenAICompatibleResponseParser):
    def _extract_provider_error(self, message: Dict[str, Any]):
        base_resp = message.get('base_resp')
        if isinstance(base_resp, dict):
            status_code = base_resp.get('status_code')
            if status_code not in (None, 0, '0'):
                return str(status_code), None
        return super()._extract_provider_error(message)


class MinimaxChat(OnlineChatModuleBase, FileHandlerBase):

    MODEL_NAME = 'MiniMax-M3'
    PROVIDER_NAME = 'minimax'
    RESPONSE_PROFILE = OnlineChatModuleBase.RESPONSE_PROFILE.extend(code_map={
        '1001': ModelFailureCode.REQUEST_TIMEOUT,
        '1002': ModelFailureCode.RATE_LIMITED,
        '1004': ModelFailureCode.AUTHENTICATION_FAILED,
        '1008': ModelFailureCode.QUOTA_EXHAUSTED,
        '1013': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
        '1024': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
        '1026': ModelFailureCode.INPUT_FILTERED,
        '1027': ModelFailureCode.OUTPUT_FILTERED,
        '1033': ModelFailureCode.PROVIDER_INTERNAL_ERROR,
        '1039': ModelFailureCode.TOKEN_LIMIT,
        '1041': ModelFailureCode.RATE_LIMITED,
        '1042': ModelFailureCode.INVALID_REQUEST,
        '2013': ModelFailureCode.INVALID_REQUEST,
        '2045': ModelFailureCode.RATE_LIMITED,
        '2049': ModelFailureCode.AUTHENTICATION_FAILED,
        '2056': ModelFailureCode.QUOTA_EXHAUSTED,
    })
    RESPONSE_PARSER_CLASS = _MinimaxResponseParser
    BASE_URLS = {
        'global': 'https://api.minimax.io/v1/',
        'cn': 'https://api.minimaxi.com/v1/',
    }

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: str = None, stream: bool = True, return_trace: bool = False,
                 region: str = 'global', **kwargs):
        if region not in self.BASE_URLS:
            raise ValueError(f'Unsupported MiniMax region: {region!r}. Expected one of {tuple(self.BASE_URLS)}')
        base_url = base_url or self.BASE_URLS[region]
        model = model or self.MODEL_NAME
        super().__init__(api_key=api_key or self._default_api_key(), base_url=base_url, model_name=model,
                         stream=stream, return_trace=return_trace, **kwargs)
        FileHandlerBase.__init__(self)

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


class MinimaxText2Image(LazyLLMOnlineText2ImageModuleBase):
    MODEL_NAME = 'image-01'
    H3_MODELS = ('MiniMax-H3-Image', 'MiniMax-H3-Image-Highspeed')

    def __init__(self, api_key: str = None, model: str = None,
                 url: Optional[str] = None, return_trace: bool = False, **kwargs):
        url = url or kwargs.pop('base_url', None)
        super().__init__(api_key=api_key or self._default_api_key(),
                         model_name=model or MinimaxText2Image.MODEL_NAME, url=url, return_trace=return_trace, **kwargs)
        if self._type == LLMType.IMAGE_EDITING:
            raise ValueError('MINIMAX series models do not support image editing now.')
        self._endpoint = 'image_generation'

    def _make_request(self, endpoint: str, payload: Dict[str, Any], base_url: Optional[str] = None,
                      timeout: int = 180) -> Dict[str, Any]:
        url = urljoin(base_url or self._base_url or 'https://api.minimaxi.com/v1/', endpoint)
        response = requests.post(url, headers=self._header, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        base_resp = result.get('base_resp')
        if base_resp and base_resp.get('status_code') not in (None, 0):
            raise Exception(f"Minimax API error {base_resp.get('status_code')}: {base_resp.get('status_msg')}")
        return result

    def _forward(self, input: str = None, url: str = None, model: str = None, **kwargs):
        if model in self.H3_MODELS:
            return self._generate_h3(input, model, url, **kwargs)
        return self._generate_legacy(input, model=model, url=url, **kwargs)

    def _generate_h3(self, prompt, model, url, n=1, response_format='url', size=None, aspect_ratio=None,
                     area=None, output_format=None, output_compression=None, disable_prompt_enhanced=None,
                     aigc_watermark=None, stream=False, files=None, **kwargs):
        unsupported = kwargs.keys() - {'priority', 'stream_output'}
        if unsupported:
            raise ValueError(f'Unsupported MiniMax H3 image options: {sorted(unsupported)}')
        if stream or files:
            raise ValueError('MiniMax H3 text-to-image accepts non-streaming text prompts only.')
        if response_format == 'base64':
            response_format = 'b64_json'
        if response_format not in ('url', 'b64_json'):
            raise ValueError('MiniMax H3 response_format must be url or b64_json.')
        endpoint = urlsplit(url or 'https://api.minimax.cn/v1/')
        path = endpoint.path.rstrip('/')
        if not path.endswith('/images/generations'):
            path += '/images/generations'
        if aigc_watermark is not None and endpoint.hostname != 'api.minimax.cn':
            raise ValueError('aigc_watermark is available only on the MiniMax China endpoint.')
        options = {'size': size, 'aspect_ratio': aspect_ratio, 'area': area, 'output_format': output_format,
                   'output_compression': output_compression, 'disable_prompt_enhanced': disable_prompt_enhanced,
                   'aigc_watermark': aigc_watermark}
        payload = {'prompt': prompt, 'model': model, 'n': n, 'response_format': response_format}
        payload.update({key: value for key, value in options.items() if value is not None})
        # A redirect could replay a billable generation.
        response = requests.post(endpoint._replace(path=path).geturl(), json=payload, headers=self._header,
                                 timeout=600, allow_redirects=False)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                message = response.json()['error']['message']
            except (ValueError, KeyError, TypeError):
                raise exc
            raise requests.HTTPError(f'{exc}: {message}', response=response) from exc
        if response.is_redirect:
            raise RuntimeError('MiniMax H3 image generation returned a redirect.')
        images = self._read_h3_images(response.json().get('data'), response_format)
        return encode_query_with_filepaths(None, bytes_to_file(images))

    def _read_h3_images(self, data, response_format):
        if not isinstance(data, list) or not data:
            raise RuntimeError('MiniMax H3 returned no images.')
        images = []
        for item in data:
            if error := item.get('error'):
                lazyllm.LOG.warning(f'MiniMax H3 image generation failed: {error.get("message")}')
                continue
            if not item.get(response_format):
                raise RuntimeError(f'MiniMax H3 returned an image without {response_format}.')
            if response_format == 'url':
                image = self._get_image_data_from_url(item['url'], timeout=180)
            else:
                image = base64.b64decode(item['b64_json'], validate=True)
                self._validate_image_data(image, 'MiniMax H3 image response')
            images.append(image)
        if not images:
            raise RuntimeError('MiniMax H3 returned no successful images.')
        return images

    def _generate_legacy(self, input: str = None, style: Dict[str, Any] = None,
                         aspect_ratio: str = None, width: int = None, height: int = None,
                         response_format: str = 'url', seed: int = None, n: int = 1,
                         prompt_optimizer: bool = False, aigc_watermark: bool = False,
                         url: str = None, model: str = None, **kwargs):
        payload: Dict[str, Any] = {
            'model': model,
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
        payload.update(kwargs)

        result = self._make_request(self._endpoint, payload, base_url=url)
        data = result.get('data') or {}

        image_bytes: List[bytes] = []
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


class MinimaxTTS(LazyLLMOnlineTTSModuleBase):
    MODEL_NAME = 'speech-2.6-hd'

    def __init__(self, api_key: str = None, model_name: str = None,
                 base_url: Optional[str] = None,
                 return_trace: bool = False, **kwargs):
        if kwargs.pop('stream', False):
            raise ValueError('MinimaxTTS does not support streaming output, please set stream to False')
        base_url = base_url or 'https://api.minimaxi.com/v1/'
        super().__init__(api_key=api_key or self._default_api_key(),
                         model_name=model_name or MinimaxTTS.MODEL_NAME,
                         return_trace=return_trace, base_url=base_url, **kwargs)
        self._endpoint = 't2a_v2'

    def _make_request(self, endpoint: str, payload: Dict[str, Any], base_url: Optional[str] = None,
                      timeout: int = 180) -> Dict[str, Any]:
        url = urljoin(base_url or self._base_url, endpoint)
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
                 stream_options: Dict[str, Any] = None, out_path: str = None,
                 url: str = None, model: str = None, **kwargs):
        if stream:
            raise ValueError('MinimaxTTS does not support streaming output, please set stream to False')
        voice_setting = voice_setting or {}
        voice_setting.setdefault('voice_id', 'male-qn-qingse')
        payload: Dict[str, Any] = {
            'model': model,
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
        result = self._make_request(self._endpoint, payload, base_url=url, timeout=180)
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
