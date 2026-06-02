from urllib.parse import urlparse
import threading

from lazyllm import LOG
from ..pdfReader import PDFReader
from ..readerBase import LazyLLMReaderBase
from .mineru_pdf_reader import MineruPDFReader
from .paddleocr_pdf_reader import PaddleOCRPDFReader
from .ocr_reader_base import read_dynamic_ocr_configs, read_static_api_key

MINERU_ONLINE_FALLBACK_URL = 'https://mineru.net'


class DynamicPDFReader(LazyLLMReaderBase):

    def __init__(self, *, ocr_type: str | None = None, ocr_url: str = '', patch_applied: bool = False,
                 mineru_upload_mode: str | None = None,
                 mineru_backend: str = 'hybrid-auto-engine', image_cache_dir: str | None = None,
                 post_func=None, timeout: int | None = None, ocr_dynamic: bool = False,
                 return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._ocr_type = (ocr_type or '').strip().lower()
        self._ocr_url = ocr_url.rstrip('/')
        self._patch_applied = patch_applied
        self._mineru_upload_mode = mineru_upload_mode
        self._mineru_backend = mineru_backend
        self._image_cache_dir = image_cache_dir
        self._post_func = post_func
        self._timeout = timeout
        self._ocr_dynamic = ocr_dynamic
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
    def _parse_bool(value: str | None) -> bool | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized == '':
            return None
        if normalized in ('1', 'true', 'yes', 'on'):
            return True
        if normalized in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(f'mineru_upload_mode must be a boolean string, got: {value!r}')

    @staticmethod
    def _default_mineru_upload_mode(ocr_url: str) -> bool:
        hostname = (urlparse(ocr_url).hostname or '').lower()
        return hostname != 'mineru'

    @staticmethod
    def _has_value(value: str | None) -> bool:
        return isinstance(value, str) and value.strip() != ''

    @classmethod
    def _resolve_dynamic_type(cls, ocr_type: str, mineru_key: str | None, paddle_key: str | None) -> str:
        normalized_type = (ocr_type or '').strip().lower()
        if normalized_type == 'mineru':
            if cls._has_value(mineru_key):
                return 'mineru'
            LOG.warning('[DynamicPDFReader] ocr_dynamic=true but mineru_api_key is empty, fallback to PDFReader')
            return 'none'
        if normalized_type == 'paddleocr':
            if cls._has_value(paddle_key):
                return 'paddleocr'
            LOG.warning('[DynamicPDFReader] ocr_dynamic=true but paddle_api_key is empty, fallback to PDFReader')
            return 'none'
        if cls._has_value(mineru_key):
            return 'mineru'
        if cls._has_value(paddle_key):
            return 'paddleocr'
        LOG.info('[DynamicPDFReader] ocr_dynamic=true and no OCR key configured, fallback to PDFReader')
        return 'none'

    @classmethod
    def _coerce_bool(cls, value) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        return cls._parse_bool(str(value))

    def _build_reader(self, normalized_type: str, normalized_url: str,
                      mineru_upload_mode: bool | None) -> LazyLLMReaderBase:
        if normalized_type in ('', 'none'):
            return PDFReader(split_doc=True, return_trace=self._return_trace)

        if normalized_type == 'mineru':
            upload_mode = mineru_upload_mode
            if upload_mode is None:
                upload_mode = self._default_mineru_upload_mode(normalized_url)
            return MineruPDFReader(
                url=normalized_url,
                backend=self._mineru_backend,
                upload_mode=upload_mode,
                post_func=self._post_func,
                timeout=self._timeout,
                patch_applied=self._patch_applied,
                image_cache_dir=self._image_cache_dir,
            )

        if normalized_type == 'paddleocr':
            return PaddleOCRPDFReader(
                url=normalized_url,
                images_dir=self._image_cache_dir,
            )

        raise ValueError(f'Unsupported OCR server type: {normalized_type!r}')

    def _resolve_options(self, extra_info: dict | None) -> tuple:
        info = {}
        dynamic_cfg = read_dynamic_ocr_configs()
        if dynamic_cfg:
            info.update({k: v for k, v in dynamic_cfg.items() if v is not None})
        if extra_info:
            info.update({k: v for k, v in extra_info.items() if v is not None})
        ocr_type = str(info.get('ocr_type', self._ocr_type or '')).strip().lower()
        ocr_url = str(info.get('ocr_url', self._ocr_url or '')).rstrip('/')
        mineru_upload_mode = self._coerce_bool(
            info.get('mineru_upload_mode', self._mineru_upload_mode))
        mineru_key = info.get('mineru_api_key')
        paddle_key = info.get('paddle_api_key')
        if mineru_key is None:
            mineru_key = read_static_api_key('mineru_api_key')
        if paddle_key is None:
            paddle_key = read_static_api_key('paddle_api_key')
        ocr_dynamic = self._coerce_bool(info.get('ocr_dynamic'))
        if ocr_dynamic is None:
            ocr_dynamic = self._ocr_dynamic
        if ocr_dynamic:
            ocr_type = self._resolve_dynamic_type(ocr_type, mineru_key, paddle_key)
        if ocr_type == 'mineru' and not self._has_value(ocr_url) and self._has_value(mineru_key):
            ocr_url = MINERU_ONLINE_FALLBACK_URL
        return ocr_type, ocr_url, mineru_upload_mode, mineru_key, paddle_key

    def _load_data(self, file, extra_info=None, use_cache: bool = True, **kwargs):
        ocr_type, ocr_url, mineru_upload_mode, mineru_key, paddle_key = (
            self._resolve_options(extra_info)
        )
        runtime_info = {}
        dynamic_cfg = read_dynamic_ocr_configs()
        if dynamic_cfg:
            runtime_info.update({k: v for k, v in dynamic_cfg.items() if v is not None})
        if extra_info:
            runtime_info.update({k: v for k, v in extra_info.items() if v is not None})
        effective_ocr_dynamic = self._coerce_bool(runtime_info.get('ocr_dynamic'))
        if effective_ocr_dynamic is None:
            effective_ocr_dynamic = self._ocr_dynamic
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
        merged_info = dict(extra_info or {})
        if mineru_key is not None:
            merged_info['mineru_api_key'] = mineru_key
        if paddle_key is not None:
            merged_info['paddle_api_key'] = paddle_key
        # PDFReader._load_data only accepts file; OCR readers accept extra_info/use_cache.
        if isinstance(reader, PDFReader):
            return reader.forward(file)
        return reader.forward(file, extra_info=merged_info, use_cache=use_cache, **kwargs)
