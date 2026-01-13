from typing import List, Dict, Union, Optional
import lazyllm
from ....servermodule import LLMBase
from .utils import OnlineModuleBase
import base64
from pathlib import Path
import requests
from urllib.parse import urlparse
import ipaddress
import socket
from io import BytesIO
from lazyllm.thirdparty import PIL

class OnlineMultiModalBase(OnlineModuleBase, LLMBase):
    def __init__(self, model_series: str, model: str = None, return_trace: bool = False, skip_auth: bool = False,
                 api_key: Optional[Union[str, List[str]]] = None, url: str = None, type: Optional[str] = None, **kwargs):
        OnlineModuleBase.__init__(self, api_key=api_key, skip_auth=skip_auth, return_trace=return_trace)
        LLMBase.__init__(self, stream=False, init_prompt=False, type=type)
        self._model_series = model_series
        self._model_name = model if model is not None else kwargs.get('model_name')
        self._base_url = url if url is not None else kwargs.get('base_url')
        self._validate_model_config()

    def _validate_model_config(self):
        '''Validate model configuration'''
        if not self._model_series:
            raise ValueError('model_series cannot be empty')
        if not self._model_name:
            lazyllm.LOG.warning(f'model_name not specified for {self._model_series}')

    @property
    def series(self):
        return self._model_series

    @property
    def type(self):
        return 'MultiModal'

    def _forward(self, input: Union[Dict, str] = None, files: List[str] = None, **kwargs):
        '''Forward method to be implemented by subclasses'''
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method')

    def forward(self, input: Union[Dict, str] = None, *, lazyllm_files=None,
                url: str = None, model: str = None, **kwargs):
        '''Main forward method with file handling'''
        try:
            input, files = self._get_files(input, lazyllm_files)
            runtime_url = url or kwargs.pop('base_url', None) or self._base_url
            runtime_model = model or kwargs.pop('model_name', None) or self._model_name
            call_params = {'input': input, **kwargs}
            if files: call_params['files'] = files
            return self._forward(**call_params, model=runtime_model, url=runtime_url)

        except Exception as e:
            lazyllm.LOG.error(f'Error in {self.__class__.__name__}.forward: {str(e)}')
            raise

    def __repr__(self):
        return lazyllm.make_repr('Module', 'OnlineMultiModalModule',
                                 series=self._model_series,
                                 name=self._model_name,
                                 return_trace=self._return_trace)

    def _is_internal_address(self, hostname: str) -> bool:
        try:
            ip_addresses = socket.gethostbyname_ex(hostname)[2]
            for ip_str in ip_addresses:
                ip = ipaddress.ip_address(ip_str)
                if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                    return True
            return False
        except Exception as e:
            lazyllm.LOG.warning(f'Failed to parse hostname={hostname}: {e}')
            return True

    def _validate_url_security(self, url: str) -> None:
        if not lazyllm.config['allow_internal_network']:
            parse = urlparse(url)
            hostname = parse.hostname
            if hostname and self._is_internal_address(hostname):
                raise ValueError(
                    f'Access to internal network address is not allowed: {hostname}. '
                    f'Set LAZYLLM_ALLOW_INTERNAL_NETWORK=True to enable internal network access.'
                )

    def _validate_image_content_type(self, content_type: str, source: str) -> None:
        if not content_type.startswith('image/'):
            raise ValueError(
                f'Invalid content type for image: {content_type} from {source}. '
                f'Expected content type starting with "image/".'
            )

    def _validate_image_data(self, data: bytes, source: str) -> None:
        try:
            with PIL.Image.open(BytesIO(data)) as img:
                img.verify()
        except Exception:
            raise ValueError(
                f'Invalid image data from {source}. '
                f'The file does not appear to be a valid image.'
            )

    def _get_image_data_from_url(self, url: str, timeout: int = 30) -> bytes:
        self._validate_url_security(url)
        resp = requests.get(url, timeout=timeout, allow_redirects=False)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '')
        self._validate_image_content_type(content_type, url)
        data = resp.content
        self._validate_image_data(data, url)
        return data

    def _load_images(self, image_paths: Union[str, List[str]]) -> List[tuple]:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        results = []
        for image_path in image_paths:
            try:
                if image_path.startswith('http://') or image_path.startswith('https://'):
                    data = self._get_image_data_from_url(image_path)
                else:
                    p = Path(image_path)
                    if not p.exists():
                        raise FileNotFoundError(f'Image file not found: {image_path}')
                    data = p.read_bytes()
                    self._validate_image_data(data, image_path)
                base64_str = base64.b64encode(data).decode('utf-8')
                results.append((base64_str, data))
            except Exception as e:
                lazyllm.LOG.error(f'Unexpected error loading image from {image_path}: {str(e)}')
                raise ValueError(f'Failed to load image from {image_path}: {str(e)}')
        return results
