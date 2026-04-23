from typing import Optional, Callable, Set, List, Dict
from pathlib import Path
from enum import Enum

from ..readerBase import _RichReader
from ...doc_node import DocNode
from .ocr_ir import Block


class ServiceVariant(str, Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'

    @classmethod
    def _missing_(cls, value):
        supported = ', '.join(f'{m.value!r}' for m in cls)
        raise ValueError(f'Invalid service_variant: {value!r}, only support: {supported}')


class _OcrReaderBase(_RichReader):
    def __init__(self,
            url,
            image_cache_dir: Path,
            service_variant: ServiceVariant = 'online',
            droped_types: Set[str] = set(),
            split_doc: bool = True,
            post_func: Optional[Callable] = None,
            return_trace: bool = True):
        super().__init__(post_func=post_func, split_doc=split_doc, return_trace=return_trace)
        self._url = url
        self._image_cache_dir = image_cache_dir
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        self._service_variant = ServiceVariant(service_variant)
        self._droped_types = droped_types
        self._page_size = None

    def _fetch_response(self, file: Path, use_cache: bool = True) -> str:
        '''Fetch raw response string from the OCR service.'''
        raise NotImplementedError

    def _build_nodes_from_response(self, response: str, file: Path,
                       extra_info: Optional[Dict] = None) -> List[DocNode]:
        '''Parse OCR service response into DocNodes.'''
        raise NotImplementedError

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None, use_cache: bool = True, **kwargs
        ) -> List[DocNode]:
        response_raw_text = self._fetch_response(file, use_cache=use_cache)
        return self._build_nodes_from_response(response_raw_text, file, extra_info)


class _Adapter:
    def _adapt_raw(self, raw: dict) -> List[Block]:
        '''Adapt raw JSON response to intermediate block representation.

        Subclasses implement service-specific adaptation logic directly.'''
        raise NotImplementedError

    def _build_nodes_from_blocks(self, blocks: List[Block], file: Path,
            extra_info: Optional[Dict] = None) -> List[DocNode]:
        '''Build DocNodes from parsed intermediate blocks.'''
        raise NotImplementedError
