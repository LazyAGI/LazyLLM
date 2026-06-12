from pathlib import Path
from typing import List, Optional

from typing_extensions import override

from .mineru_pdf_reader import MineruPDFReader

_PPT_SUFFIXES = frozenset({'.ppt', '.pptx', '.pptm'})
# Mineru online API does not return bbox for office files.
_DEFAULT_BBOX = [0, 0, 0, 0]


class MineruPPTReader(MineruPDFReader):

    @staticmethod
    def is_ppt_file(file_path) -> bool:
        return Path(file_path).suffix.lower() in _PPT_SUFFIXES

    @staticmethod
    def _split_for_upload(file_path: str) -> List[tuple]:
        return [(file_path, 0)]

    @override
    def _resolve_bbox(self, item: dict) -> Optional[List]:
        bbox = item.get('bbox')
        return bbox if bbox is not None else _DEFAULT_BBOX
