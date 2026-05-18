import json
import os
import shutil
from typing import Optional, Callable, Set, List, Dict, Tuple
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
    def __init__(self, url,
                 image_cache_dir: Path,
                 service_variant: str = 'online',
                 dropped_types: Optional[Set[str]] = None,
                 split_doc: bool = True,
                 post_func: Optional[Callable] = None,
                 return_trace: bool = True):
        super().__init__(post_func=post_func, split_doc=split_doc, return_trace=return_trace)
        self._url = url
        self._image_cache_dir = Path(image_cache_dir) if image_cache_dir is not None else None
        if self._image_cache_dir is not None:
            self._image_cache_dir.mkdir(parents=True, exist_ok=True)
            # Clear any stale contents from previous runs
            for item in self._image_cache_dir.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()
        self._service_variant = ServiceVariant(service_variant)
        self._dropped_types = dropped_types if dropped_types is not None else set()

    @staticmethod
    def _split_large_pdf(pdf_path: str, max_size_mb: int = 200) -> List[Tuple[str, int]]:
        '''Split a large PDF into equal-page parts until each is under max_size_mb.

        Split files are placed in {pdf_path}.splits/ directory.
        If already split and sizes are valid, reuse existing files.

        Returns a list of (sub_file_path, start_page_index) tuples, ordered by page.
        If the original file is small enough, returns [(pdf_path, 0)].
        '''
        from pypdf import PdfReader, PdfWriter

        max_size_bytes = max_size_mb * 1024 * 1024
        original_size = os.path.getsize(pdf_path)

        if original_size <= max_size_bytes:
            return [(pdf_path, 0)]

        splits_dir = Path(f'{pdf_path}.splits')

        # Check cache
        if splits_dir.exists():
            pdf_files = sorted(splits_dir.glob('*.pdf'))
            if pdf_files and all(os.path.getsize(f) <= max_size_bytes for f in pdf_files):
                return [(str(f), _OcrReaderBase._parse_page_start(f.name)) for f in pdf_files]

        # Clear and re-split
        if splits_dir.exists():
            shutil.rmtree(splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        basename = Path(pdf_path).stem

        # Calculate number of parts: floor(size / max_size) + 1
        num_parts = original_size // max_size_bytes + 1
        pages_per_part = max(1, total_pages // num_parts)

        # Build initial chunks as (list_of_page_indices, start_offset)
        chunks = []
        start_page = 0
        while start_page < total_pages:
            end_page = min(start_page + pages_per_part, total_pages)
            chunks.append((list(range(start_page, end_page)), start_page))
            start_page = end_page

        # Iteratively split oversized chunks until all are under max_size_bytes
        final_result = []
        while chunks:
            page_indices, offset = chunks.pop(0)
            writer = PdfWriter()
            for i in page_indices:
                writer.add_page(reader.pages[i])

            part_name = f'{basename}_part_{offset}-{offset + len(page_indices)}.pdf'
            part_path = splits_dir / part_name
            with open(part_path, 'wb') as f:
                writer.write(f)

            if os.path.getsize(part_path) <= max_size_bytes or len(page_indices) == 1:
                final_result.append((str(part_path), offset))
            else:
                # Oversized and multi-page: split in half and re-queue
                mid = len(page_indices) // 2
                chunks.insert(0, (page_indices[mid:], offset + mid))
                chunks.insert(0, (page_indices[:mid], offset))
                os.remove(part_path)  # Remove the oversized temp file

        return sorted(final_result, key=lambda x: x[1])

    @staticmethod
    def _parse_page_start(filename: str) -> int:
        '''Parse start page from split filename, e.g. 'doc_part_0-50.pdf' -> 0.'''
        import re
        match = re.search(r'_part_(\d+)-\d+\.pdf$', filename)
        return int(match.group(1)) if match else 0

    def _fetch_response(self, file: Path, use_cache: bool = False) -> str:
        '''Fetch raw response string from the OCR service.'''
        raise NotImplementedError

    def _build_nodes_from_response(self, response_text: str, file: Path,
                                   extra_info: Optional[Dict] = None) -> List[DocNode]:
        '''Parse OCR service response into DocNodes.'''
        raw = json.loads(response_text)
        blocks = self._adapt_json_to_IR(raw)
        # Post processing
        blocks = l1_normalize(blocks)
        blocks = l2_associate(blocks)
        return self._build_nodes_from_blocks(blocks, file, extra_info)

    def _load_data(self, file, extra_info: Optional[Dict] = None, use_cache: bool = True,
                   **kwargs) -> List[DocNode]:
        # Preserve URL strings; Path() would corrupt https:// into https:/
        file_path = file if isinstance(file, str) and file.startswith(('http://', 'https://')) else Path(file)
        response_raw_text = self._fetch_response(file_path, use_cache=use_cache)
        return self._build_nodes_from_response(response_raw_text, file_path, extra_info)

    @staticmethod
    def _parse_table_html(html_text: str) -> List[Cell]:
        soup = bs4.BeautifulSoup(html_text, 'html.parser')
        cells: List[Cell] = []
        table = soup.find('table')
        if table is None:
            return []
        for row_idx, tr in enumerate(table.find_all('tr')):
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
