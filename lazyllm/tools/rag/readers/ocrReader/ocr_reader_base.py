import json
import os
import shutil
from contextlib import nullcontext
from typing import Optional, Callable, Set, List, Dict, Tuple
from pathlib import Path
from urllib.parse import urlparse

from lazyllm import globals as lazyllm_globals
from lazyllm.thirdparty import bs4, pypdf
from lazyllm.common import AuthStrategy, BearerTokenStrategy, CredentialMixin
from ..readerBase import _RichReader
from ...doc_node import DocNode
from .ocr_ir import Block, Cell
from .ocr_postprocessor import l1_normalize, l2_associate


MINERU_OFFICIAL_ONLINE_HOST = 'mineru.net'
PADDLE_OFFICIAL_ONLINE_HOST_SUFFIX = 'aistudio-app.com'
PADDLE_OFFICIAL_ONLINE_URL = 'https://paddleocr.aistudio-app.com/api/v2/ocr/jobs'


def _parse_url_host(url: Optional[str]) -> str:
    raw = (url or '').strip()
    if not raw:
        return ''
    if '://' not in raw:
        raw = f'http://{raw}'
    return (urlparse(raw).hostname or '').lower()


def is_mineru_official_online_url(url: Optional[str]) -> bool:
    '''True when MinerU should use the official mineru.net upload API (not local HTTP).'''
    host = _parse_url_host(url)
    if not host:
        return True
    return host == MINERU_OFFICIAL_ONLINE_HOST or host.endswith(f'.{MINERU_OFFICIAL_ONLINE_HOST}')


def is_paddle_official_online_url(url: Optional[str]) -> bool:
    host = _parse_url_host(url)
    if not host:
        return True
    return host.endswith(PADDLE_OFFICIAL_ONLINE_HOST_SUFFIX) or PADDLE_OFFICIAL_ONLINE_HOST_SUFFIX in host


def read_dynamic_ocr_configs() -> Optional[Dict]:
    try:
        cfg = lazyllm_globals.config['dynamic_ocr_configs']
    except Exception:
        return None
    return cfg if isinstance(cfg, dict) else None


def read_static_api_key(key: str) -> Optional[str]:
    import lazyllm
    val = lazyllm.config[key]
    return val if val else None


class _Adapter:
    def _adapt_json_to_IR(self, raw: dict) -> List[Block]:
        '''Adapt raw JSON response to intermediate block representation.

        Subclasses implement service-specific adaptation logic directly.'''
        raise NotImplementedError

    def _build_nodes_from_blocks(self, blocks: List[Block], file: Path,
                                 extra_info: Optional[Dict] = None) -> List[DocNode]:
        '''Build DocNodes from parsed intermediate blocks.'''
        raise NotImplementedError


class _OcrReaderBase(_RichReader, _Adapter, CredentialMixin):
    def __init__(self, url,
                 image_cache_dir: Path,
                 dropped_types: Optional[Set[str]] = None,
                 split_doc: bool = True,
                 post_func: Optional[Callable] = None,
                 return_trace: bool = True,
                 token: Optional[str] = None,
                 dynamic_auth: bool = False,
                 auth_strategy: Optional[AuthStrategy] = None,
                 auth_source_key: Optional[str] = None):
        super().__init__(post_func=post_func, split_doc=split_doc, return_trace=return_trace)
        credential = self._default_credential(token, dynamic_auth)
        self.__init_credential__(credential, strategy=auth_strategy or BearerTokenStrategy())
        # auth_source_key links this reader to globals.config['dynamic_ocr_auth'][key].
        self._auth_source_key = (auth_source_key or self.__class__.__name__.replace('PDFReader', '')).lower()
        self._url = url
        self._image_cache_dir = Path(image_cache_dir) if image_cache_dir is not None else None
        if self._image_cache_dir is not None:
            self._image_cache_dir.mkdir(parents=True, exist_ok=True)
            # Clear any stale contents from previous runs
            for item in self._image_cache_dir.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()
        self._dropped_types = dropped_types if dropped_types is not None else set()

    def _resolve_dynamic_token(self) -> str:
        # Dynamic OCR auth is request/session scoped and injected via globals.config.
        mapping = lazyllm_globals.config['dynamic_ocr_auth'] or {}
        return mapping.get(self._auth_source_key, '')

    def _missing_dynamic_token_error(self) -> str:
        return (
            f'dynamic_ocr_auth["{self._auth_source_key}"] is not set in globals.config; '
            f'use inject_ocr_config(..., ocr_auth={{"{self._auth_source_key}": "..."}}) '
            f'or set globals.config["dynamic_ocr_auth"] before OCR parsing'
        )

    def _effective_token(self, explicit_token: Optional[str] = None) -> str:
        if explicit_token is None:
            return self.get_current_token()
        with self.override_credential(secret_key=explicit_token):
            return self.get_current_token()

    def _auth_headers(self, headers: Optional[Dict[str, str]] = None, *,
                      explicit_token: Optional[str] = None) -> Dict[str, str]:
        if explicit_token is None:
            return self.inject_auth_header(headers)
        with self.override_credential(secret_key=explicit_token):
            return self.inject_auth_header(headers)

    def _auth_scope(self, explicit_token: Optional[str] = None):
        # Apply per-request token override only within one OCR parse call.
        if explicit_token is None:
            return nullcontext()
        return self.override_credential(secret_key=explicit_token)

    @staticmethod
    def _split_large_pdf(pdf_path: str, max_size_mb: int = 200,
                         max_pages: int = 200) -> List[Tuple[str, int]]:
        '''Split a large PDF into parts, each under max_size_mb and max_pages.

        Split files are placed in {pdf_path}.splits/ directory.
        If already split and sizes/pages are valid, reuse existing files.

        Returns a list of (sub_file_path, start_page_index) tuples, ordered by page.
        If the original file fits both limits, returns [(pdf_path, 0)].
        '''

        max_size_bytes = max_size_mb * 1024 * 1024
        original_size = os.path.getsize(pdf_path)

        reader = pypdf.PdfReader(pdf_path)
        total_pages = len(reader.pages)

        need_split = original_size > max_size_bytes or total_pages > max_pages

        if not need_split:
            return [(pdf_path, 0)]

        splits_dir = Path(f'{pdf_path}.splits')

        # Check cache
        if splits_dir.exists():
            pdf_files = sorted(splits_dir.glob('*.pdf'))
            if pdf_files and all(
                os.path.getsize(f) <= max_size_bytes
                and _OcrReaderBase._parse_page_count(f.name) <= max_pages
                for f in pdf_files
            ):
                return [(str(f), _OcrReaderBase._parse_page_start(f.name)) for f in pdf_files]

        # Clear and re-split
        if splits_dir.exists():
            shutil.rmtree(splits_dir)
        splits_dir.mkdir(parents=True, exist_ok=True)

        basename = Path(pdf_path).stem

        # Calculate number of parts needed to satisfy BOTH size and page limits
        size_parts = max(1, (original_size + max_size_bytes - 1) // max_size_bytes)
        page_parts = max(1, (total_pages + max_pages - 1) // max_pages)
        num_parts = max(size_parts, page_parts)
        pages_per_part = max(1, total_pages // num_parts)

        # Build initial chunks
        chunks = []
        start_page = 0
        while start_page < total_pages:
            end_page = min(start_page + pages_per_part, total_pages)
            chunks.append((list(range(start_page, end_page)), start_page))
            start_page = end_page

        # Iteratively split until all chunks satisfy both limits
        final_result = []
        while chunks:
            page_indices, offset = chunks.pop(0)
            writer = pypdf.PdfWriter()
            for i in page_indices:
                writer.add_page(reader.pages[i])

            part_name = f'{basename}_part_{offset}-{offset + len(page_indices)}.pdf'
            part_path = splits_dir / part_name
            with open(part_path, 'wb') as f:
                writer.write(f)

            part_size_ok = os.path.getsize(part_path) <= max_size_bytes
            part_pages_ok = len(page_indices) <= max_pages
            if (part_size_ok and part_pages_ok) or len(page_indices) == 1:
                final_result.append((str(part_path), offset))
            else:
                # Oversized or too many pages: split in half and re-queue
                mid = len(page_indices) // 2
                chunks.insert(0, (page_indices[mid:], offset + mid))
                chunks.insert(0, (page_indices[:mid], offset))
                os.remove(part_path)

        return sorted(final_result, key=lambda x: x[1])

    @staticmethod
    def _parse_page_start(filename: str) -> int:
        '''Parse start page from split filename, e.g. 'doc_part_0-50.pdf' -> 0.'''
        import re
        match = re.search(r'_part_(\d+)-\d+\.pdf$', filename)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _parse_page_count(filename: str) -> int:
        '''Parse page count from split filename, e.g. 'doc_part_0-50.pdf' -> 50.'''
        import re
        match = re.search(r'_part_\d+-(\d+)\.pdf$', filename)
        return int(match.group(1)) - _OcrReaderBase._parse_page_start(filename) if match else 0

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
