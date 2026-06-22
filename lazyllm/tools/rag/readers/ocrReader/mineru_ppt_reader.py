from pathlib import Path
from typing import List, Optional, Tuple

from typing_extensions import override

from .mineru_pdf_reader import MineruPDFReader, _DEFAULT_BBOX
from .ocr_ir import Block

_PPT_SUFFIXES = frozenset({'.ppt', '.pptx', '.pptm'})


class MineruPPTReader(MineruPDFReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auth_source_key = 'mineru'

    @staticmethod
    def is_ppt_file(file_path) -> bool:
        return Path(file_path).suffix.lower() in _PPT_SUFFIXES

    @staticmethod
    @override
    def _split_large_pdf(pdf_path: str, max_size_mb: int = 200,
                         max_pages: int = 200) -> List[Tuple[str, int]]:
        return [(pdf_path, 0)]

    @override
    def _adapt_one(self, item: dict) -> Optional[Block]:
        if item.get('bbox') is None:
            item = {**item, 'bbox': _DEFAULT_BBOX}
        return super()._adapt_one(item)
