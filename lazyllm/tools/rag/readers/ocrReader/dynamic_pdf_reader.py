from lazyllm import LOG
from lazyllm import globals as lazyllm_globals

from ..pdfReader import PDFReader
from ..readerBase import LazyLLMReaderBase
from .mineru_pdf_reader import MineruPDFReader
from .paddleocr_pdf_reader import PaddleOCRPDFReader
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_reader_cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reader_cache = {}

    @staticmethod
    def _normalize_type(ocr_type: str | None) -> str:
        normalized = (ocr_type or '').strip().lower()
        if normalized == 'paddle':
            return 'paddleocr'
        return normalized

    @staticmethod
    def _mineru_upload_mode_from_url(ocr_url: str) -> bool:
        from .ocr_reader_base import _is_mineru_official_online_url
        return not _is_mineru_official_online_url(ocr_url)

    def _mineru_upload_mode_for(self, ocr_url: str) -> bool:
        if self._mineru_upload_mode is not None:
            return self._mineru_upload_mode
        return self._mineru_upload_mode_from_url(ocr_url)

    @staticmethod
    def _injected_auth(key: str) -> str | None:
        try:
            auth = lazyllm_globals.config['dynamic_ocr_auth'] or {}
        except Exception:
            return None
        value = auth.get(key) if isinstance(auth, dict) else None
        return value if isinstance(value, str) and value.strip() else None

    def _resolve_route(self, extra_info: dict | None) -> tuple[str, str]:
        dynamic_cfg = read_dynamic_ocr_configs() or {}
        extra_info = extra_info or {}

        ocr_type = self._normalize_type(
            extra_info.get('ocr_type') or dynamic_cfg.get('ocr_type') or self._ocr_type
        )

        if 'ocr_url' in extra_info or 'ocr_url' in dynamic_cfg:
            raw = extra_info.get('ocr_url', dynamic_cfg.get('ocr_url'))
            return ocr_type, str(raw or '').rstrip('/')

        mineru_key = extra_info.get('mineru_api_key') or self._injected_auth('mineru')
        paddle_key = extra_info.get('paddle_api_key') or self._injected_auth('paddleocr')
        if ocr_type == 'mineru' and mineru_key:
            return ocr_type, ''
        if ocr_type == 'paddleocr' and paddle_key:
            return ocr_type, ''

        return ocr_type, str(self._ocr_url or '').rstrip('/')

    def _build_reader(self, ocr_type: str, ocr_url: str) -> LazyLLMReaderBase:
        if ocr_type in ('', 'none'):
            return PDFReader(split_doc=True, return_trace=self._return_trace)

        if ocr_type == 'mineru':
            return MineruPDFReader(
                url=ocr_url,
                backend=self._mineru_backend,
                upload_mode=self._mineru_upload_mode_for(ocr_url),
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

    def _get_reader(self, ocr_type: str, ocr_url: str) -> LazyLLMReaderBase:
        cache_key = (ocr_type, ocr_url)
        if cache_key not in self._reader_cache:
            self._reader_cache[cache_key] = self._build_reader(ocr_type, ocr_url)
            LOG.info(
                f'[DynamicPDFReader] created cached reader: type={ocr_type}, '
                f'url={ocr_url or ""}, key={cache_key}'
            )
        return self._reader_cache[cache_key]

    def _load_data(self, file, extra_info=None, use_cache: bool = True, **kwargs):
        ocr_type, ocr_url = self._resolve_route(extra_info)
        reader = self._get_reader(ocr_type, ocr_url)
        if isinstance(reader, PDFReader):
            return reader.forward(file)
        return reader.forward(file, extra_info=extra_info, use_cache=use_cache, **kwargs)
