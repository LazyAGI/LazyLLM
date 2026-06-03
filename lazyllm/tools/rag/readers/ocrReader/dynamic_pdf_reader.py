from urllib.parse import urlparse
import threading

from lazyllm import LOG
from ..pdfReader import PDFReader
from ..readerBase import LazyLLMReaderBase
from .mineru_pdf_reader import MineruPDFReader
from .paddleocr_pdf_reader import PaddleOCRPDFReader
from lazyllm import globals as lazyllm_globals
from .ocr_reader_base import read_dynamic_ocr_configs


class DynamicPDFReader(LazyLLMReaderBase):

    def __init__(self, *, ocr_type: str | None = None, ocr_url: str = '', patch_applied: bool = False,
                 mineru_upload_mode: bool | None = None,
                 mineru_backend: str = 'hybrid-auto-engine', image_cache_dir: str | None = None,
                 post_func=None, timeout: int | None = None,
                 return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._ocr_type = self._normalize_type(ocr_type)
        self._ocr_url = ocr_url.rstrip('/')
        self._patch_applied = patch_applied
        self._mineru_upload_mode = mineru_upload_mode
        self._mineru_backend = mineru_backend
        self._image_cache_dir = image_cache_dir
        self._post_func = post_func
        self._timeout = timeout
        self._return_trace = return_trace
        self._reader_cache = {}
        self._reader_lock = threading.Lock()

    def __getstate__(self):
        # threading.Lock is not picklable; docs.start() cloudpickles readers for RelayServer.
        state = self.__dict__.copy()
        state.pop('_reader_lock', None)
        state['_reader_cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reader_lock = threading.Lock()
        self._reader_cache = {}

    @staticmethod
    def _normalize_type(ocr_type: str | None) -> str:
        normalized = (ocr_type or '').strip().lower()
        if normalized == 'paddle':
            return 'paddleocr'
        return normalized

    @staticmethod
    def _mineru_upload_mode_from_url(ocr_url: str) -> bool:
        hostname = (urlparse(ocr_url).hostname or '').lower()
        return hostname != 'mineru'

    def _mineru_upload_mode_for(self, ocr_url: str) -> bool:
        if self._mineru_upload_mode is not None:
            return self._mineru_upload_mode
        return self._mineru_upload_mode_from_url(ocr_url)

    @staticmethod
    def _has_value(value: str | None) -> bool:
        return isinstance(value, str) and value.strip() != ''

    def _explicit_ocr_url(self, extra_info: dict | None) -> str | None:
        if extra_info is not None and 'ocr_url' in extra_info:
            return str(extra_info.get('ocr_url') or '').rstrip('/')
        dynamic_cfg = read_dynamic_ocr_configs()
        if dynamic_cfg and 'ocr_url' in dynamic_cfg:
            return str(dynamic_cfg.get('ocr_url') or '').rstrip('/')
        return None

    @staticmethod
    def _auth_mapping() -> dict:
        try:
            auth = lazyllm_globals.config['dynamic_ocr_auth']
        except Exception:
            return {}
        return auth if isinstance(auth, dict) else {}

    def _has_mineru_auth(self, info: dict) -> bool:
        if self._has_value(info.get('mineru_api_key')):
            return True
        return self._has_value(self._auth_mapping().get('mineru'))

    def _has_paddle_auth(self, info: dict) -> bool:
        if self._has_value(info.get('paddle_api_key')):
            return True
        return self._has_value(self._auth_mapping().get('paddleocr'))

    def _is_dynamic_request(self, info: dict) -> bool:
        if read_dynamic_ocr_configs():
            return True
        return self._has_mineru_auth(info) or self._has_paddle_auth(info)

    def _merged_routing_cfg(self, extra_info: dict | None) -> dict:
        info = {}
        dynamic_cfg = read_dynamic_ocr_configs()
        if dynamic_cfg:
            info.update({k: v for k, v in dynamic_cfg.items() if v is not None})
        if extra_info:
            info.update({k: v for k, v in extra_info.items() if v is not None})
        return info

    def _resolve_route(self, extra_info: dict | None) -> tuple[str, str]:
        info = self._merged_routing_cfg(extra_info)
        ocr_type = self._normalize_type(info.get('ocr_type', self._ocr_type))
        explicit_url = self._explicit_ocr_url(extra_info)
        if explicit_url is not None:
            ocr_url = explicit_url
        elif self._is_dynamic_request(info):
            ocr_url = ''
        else:
            ocr_url = str(self._ocr_url or '').rstrip('/')
        return ocr_type, ocr_url

    def _build_reader(self, ocr_type: str, ocr_url: str,
                      mineru_upload_mode: bool) -> LazyLLMReaderBase:
        if ocr_type in ('', 'none'):
            return PDFReader(split_doc=True, return_trace=self._return_trace)

        if ocr_type == 'mineru':
            return MineruPDFReader(
                url=ocr_url,
                backend=self._mineru_backend,
                upload_mode=mineru_upload_mode,
                post_func=self._post_func,
                timeout=self._timeout,
                patch_applied=self._patch_applied,
                image_cache_dir=self._image_cache_dir,
                dynamic_auth=True,
            )

        if ocr_type == 'paddleocr':
            return PaddleOCRPDFReader(
                url=ocr_url,
                images_dir=self._image_cache_dir,
                dynamic_auth=True,
            )

        raise ValueError(f'Unsupported OCR server type: {ocr_type!r}')

    def _load_data(self, file, extra_info=None, use_cache: bool = True, **kwargs):
        ocr_type, ocr_url = self._resolve_route(extra_info)
        mineru_upload_mode = self._mineru_upload_mode_for(ocr_url) if ocr_type == 'mineru' else None
        cache_key = (ocr_type, ocr_url, mineru_upload_mode)
        reader = self._reader_cache.get(cache_key)
        if reader is None:
            with self._reader_lock:
                reader = self._reader_cache.get(cache_key)
                if reader is None:
                    reader = self._build_reader(ocr_type, ocr_url, mineru_upload_mode)
                    self._reader_cache[cache_key] = reader
                    LOG.info(
                        f'[DynamicPDFReader] created cached reader: type={ocr_type}, '
                        f'url={ocr_url or ""}, key={cache_key}'
                    )
        if isinstance(reader, PDFReader):
            return reader.forward(file)
        return reader.forward(file, extra_info=extra_info, use_cache=use_cache, **kwargs)
