from lazyllm import globals as lazyllm_globals

from ..pdfReader import PDFReader
from ..readerBase import LazyLLMReaderBase
from .mineru_pdf_reader import MineruPDFReader
from .mineru_ppt_reader import MineruPPTReader
from .paddleocr_pdf_reader import PaddleOCRPDFReader


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

    @property
    def appendix_hash_key(self):
        return f'{self._ocr_type}|{self._ocr_url}|{self._timeout}|{self._image_cache_dir}'

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
        dynamic_cfg = lazyllm_globals.config['dynamic_ocr_configs'] or {}
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

    @staticmethod
    def _reader_type(ocr_type: str, file) -> str:
        if ocr_type == 'mineru' and MineruPPTReader.is_ppt_file(file):
            return 'mineru_ppt'
        return ocr_type

    @staticmethod
    def _reader_cache_key(ocr_type: str, ocr_url: str, file) -> tuple[str, str]:
        return DynamicPDFReader._reader_type(ocr_type, file), ocr_url

    def _build_reader(self, reader_type: str, ocr_url: str) -> LazyLLMReaderBase:
        if reader_type in ('', 'none'):
            return PDFReader(split_doc=True, return_trace=self._return_trace)

        kwargs = dict[str, str | bool](url=ocr_url, dynamic_auth=True)
        if self._image_cache_dir:
            kwargs['image_cache_dir'] = self._image_cache_dir

        if reader_type == 'mineru_ppt':
            kwargs.update(timeout=self._timeout, post_func=self._post_func)
            return MineruPPTReader(**kwargs)
        if reader_type == 'mineru':
            kwargs.update(timeout=self._timeout, post_func=self._post_func)
            return MineruPDFReader(**kwargs)
        if reader_type == 'paddleocr':
            return PaddleOCRPDFReader(**kwargs)

        raise ValueError(f'Unsupported OCR server type: {reader_type!r}')

    def _get_reader(self, reader_type: str, ocr_url: str) -> LazyLLMReaderBase:
        cache_key = (reader_type, ocr_url)
        if cache_key not in self._reader_cache:
            self._reader_cache[cache_key] = self._build_reader(reader_type, ocr_url)
        return self._reader_cache[cache_key]

    def _load_data(self, file, extra_info=None, **kwargs):
        ocr_type, ocr_url = self._resolve_route(extra_info)
        reader_type, ocr_url = self._reader_cache_key(ocr_type, ocr_url, file)
        reader = self._get_reader(reader_type, ocr_url)
        if isinstance(reader, PDFReader):
            return reader.forward(file)
        return reader.forward(
            file,
            extra_info=extra_info,
            **kwargs,
        )
