from ..pdfReader import PDFReader
from ..readerBase import LazyLLMReaderBase
from .mineru_pdf_reader import MineruPDFReader
from .paddleocr_pdf_reader import PaddleOCRPDFReader
from .ocr_reader_base import read_dynamic_ocr_configs


class DynamicPDFReader(LazyLLMReaderBase):

    def __init__(self, *, ocr_type: str | None = None, ocr_url: str = '',
                 image_cache_dir: str | None = None, post_func=None, timeout: int | None = None,
                 return_trace: bool = True):
        super().__init__(return_trace=return_trace)
        self._ocr_type = self._normalize_type(ocr_type)
        self._ocr_url = ocr_url.rstrip('/')
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

    def _resolve_route(self, extra_info: dict | None) -> tuple[str, str]:
        dynamic_cfg = read_dynamic_ocr_configs() or {}
        extra_info = extra_info or {}
        has_dynamic_type = 'ocr_type' in extra_info or 'ocr_type' in dynamic_cfg

        ocr_type = self._normalize_type(
            extra_info.get('ocr_type') or dynamic_cfg.get('ocr_type') or self._ocr_type
        )

        if 'ocr_url' in extra_info or 'ocr_url' in dynamic_cfg:
            raw = extra_info.get('ocr_url', dynamic_cfg.get('ocr_url'))
            return ocr_type, str(raw or '').rstrip('/')
        if has_dynamic_type and ocr_type in ('mineru', 'paddleocr'):
            return ocr_type, ''

        return ocr_type, str(self._ocr_url or '').rstrip('/')

    def _build_reader(self, ocr_type: str, ocr_url: str) -> LazyLLMReaderBase:
        if ocr_type in ('', 'none'):
            return PDFReader(split_doc=True, return_trace=self._return_trace)

        if ocr_type == 'mineru':
            kwargs = {
                'url': ocr_url,
                'post_func': self._post_func,
                'timeout': self._timeout,
                'dynamic_auth': True,
            }
            if self._image_cache_dir is not None:
                kwargs['image_cache_dir'] = self._image_cache_dir
            return MineruPDFReader(**kwargs)

        if ocr_type == 'paddleocr':
            kwargs = {
                'url': ocr_url,
                'dynamic_auth': True,
            }
            if self._image_cache_dir is not None:
                kwargs['images_dir'] = self._image_cache_dir
            return PaddleOCRPDFReader(**kwargs)

        raise ValueError(f'Unsupported OCR server type: {ocr_type!r}')

    def _get_reader(self, ocr_type: str, ocr_url: str) -> LazyLLMReaderBase:
        cache_key = (ocr_type, ocr_url)
        if cache_key not in self._reader_cache:
            self._reader_cache[cache_key] = self._build_reader(ocr_type, ocr_url)
        return self._reader_cache[cache_key]

    def _load_data(self, file, extra_info=None, use_cache: bool = True, **kwargs):
        ocr_type, ocr_url = self._resolve_route(extra_info)
        reader = self._get_reader(ocr_type, ocr_url)
        if isinstance(reader, PDFReader):
            return reader.forward(file)
        return reader.forward(file, extra_info=extra_info, use_cache=use_cache, **kwargs)
