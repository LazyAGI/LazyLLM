import json
from typing import Optional, Callable, Set, List, Dict
from pathlib import Path
from enum import Enum

from lazyllm.thirdparty import bs4
from ..readerBase import _RichReader
from ...doc_node import DocNode
from .ocr_ir import Block, Cell
from .ocr_postprocessor import l1_normalize, l2_associate

class ServiceVariant(str, Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'

    @classmethod
    def _missing_(cls, value):
        supported = ', '.join(f'{m.value!r}' for m in cls)
        raise ValueError(f'Invalid service_variant: {value!r}, only support: {supported}')


class _Adapter:
    def _adapt_json_to_IR(self, raw: dict) -> List[Block]:
        '''Adapt raw JSON response to intermediate block representation.

        Subclasses implement service-specific adaptation logic directly.'''
        raise NotImplementedError

    def _build_nodes_from_blocks(self, blocks: List[Block], file: Path,
            extra_info: Optional[Dict] = None) -> List[DocNode]:
        '''Build DocNodes from parsed intermediate blocks.'''
        raise NotImplementedError


class _OcrReaderBase(_RichReader, _Adapter):
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

    def _fetch_response(self, file: Path, use_cache: bool = True) -> str:
        '''Fetch raw response string from the OCR service.'''
        raise NotImplementedError

    def _build_nodes_from_response(self, response_text: str, file: Path,
            extra_info: Optional[Dict] = None) -> List[DocNode]:
        '''Parse OCR service response into DocNodes.'''
        raw = json.loads(response_text)
        blocks = self._adapt_json_to_IR(raw)
        # Post processing
        blocks = l1_normalize(blocks, self._page_size)
        blocks = l2_associate(blocks)
        return self._build_nodes_from_blocks(blocks, file, extra_info)

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None, use_cache: bool = True, **kwargs
        ) -> List[DocNode]:
        response_raw_text = self._fetch_response(file, use_cache=use_cache)
        return self._build_nodes_from_response(response_raw_text, file, extra_info)

    @staticmethod
    def _parse_table_html( html_text: str) -> List[Cell]:
        soup = bs4.BeautifulSoup(html_text, 'html.parser')
        cells: List[Cell] = []
        for row_idx, tr in enumerate(soup.find('table').find_all('tr')):
            for col_idx, td in enumerate(tr.find_all(['td', 'th'])):
                cells.append(Cell(
                    row=row_idx,
                    col=col_idx,
                    rowspan=int(td.get('rowspan', 1)),
                    colspan=int(td.get('colspan', 1)),
                    text=td.get_text(strip=True),
                ))
        return cells

    @staticmethod
    def _first(val):
        if isinstance(val, list) and val:
            return val[0]
        if isinstance(val, str):
            return val
        return None

    @staticmethod
    def _make_anchor(text: str) -> str:
        return text.strip().replace(' ', '-').replace('\n', '-')[:64]
