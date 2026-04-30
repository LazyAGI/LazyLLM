import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, vals: List[float]) -> 'BBox':
        return cls(x0=float(vals[0]), y0=float(vals[1]), x1=float(vals[2]), y1=float(vals[3]))


@dataclass
class PageRef:
    index: int
    bbox: BBox


@dataclass
class Cell:
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ''
    bbox: Optional[BBox] = None

    def __post_init__(self):
        if isinstance(self.text, str):
            self.text = unicodedata.normalize('NFKC', self.text)


@dataclass
class SectionPath:
    anchors: List[str] = field(default_factory=list)
    level: int = 0


# ----- Block base with shared fields -----
@dataclass
class Block:
    page: PageRef
    section: SectionPath = field(default_factory=SectionPath)
    lines: List[str] = field(default_factory=list)
    index: int = -1

    def __post_init__(self):
        for f in self.__dataclass_fields__.values():
            val = getattr(self, f.name)
            if isinstance(val, str):
                setattr(self, f.name, unicodedata.normalize('NFKC', val))
            elif isinstance(val, list):
                setattr(self, f.name, [
                    unicodedata.normalize('NFKC', s) if isinstance(s, str) else s
                    for s in val
                ])
            elif isinstance(val, SectionPath):
                val.anchors = [
                    unicodedata.normalize('NFKC', s) if isinstance(s, str) else s
                    for s in val.anchors
                ]

    @property
    def ty(self) -> str:
        raise NotImplementedError

    def text_content(self) -> str:
        raise NotImplementedError

    def update_metadata(self, d: dict) -> None:
        pass


@dataclass
class HeadingBlock(Block):
    level: int = -1
    md_title_level: int = -1
    text: str = ''
    anchor: str = ''

    def __post_init__(self):
        super().__post_init__()
        # Parse markdown title level
        m = re.match(r'^(#+)\s*(.*)$', self.text.strip())
        if m:
            self.md_title_level = len(m.group(1))
            self.text = m.group(2).strip()

        if self.level == -1 and self.md_title_level > 0:
            self.level = self.md_title_level

        if not self.anchor:
            self.anchor = self._make_anchor(self.text)

    @property
    def ty(self) -> str:
        return 'heading'

    def text_content(self) -> str:
        return self.text

    def update_metadata(self, d: dict) -> None:
        d['text_level'] = self.level

    @staticmethod
    def _make_anchor(text: str) -> str:
        return text.strip().replace(' ', '-').replace('\n', '-')[:64]


@dataclass
class ParagraphBlock(Block):
    text: str = ''

    @property
    def ty(self) -> str:
        return 'paragraph'

    def text_content(self) -> str:
        return self.text


@dataclass
class TableBlock(Block):
    caption: Optional[str] = None
    footnote: Optional[str] = None
    cells: List[Cell] = field(default_factory=list)
    page_range: Tuple[int, int] = (0, 0)
    merged_across_pages: bool = False
    source_pages: List[int] = field(default_factory=list)

    @property
    def ty(self) -> str:
        return 'table'

    def text_content(self) -> str:
        lines = []
        if self.caption:
            lines.append(self.caption)
        lines.append(self._cells_to_markdown())
        if self.footnote:
            lines.append(self.footnote)
        return '\n'.join(lines)

    def update_metadata(self, d: dict) -> None:
        d['table_caption'] = self.caption
        d['table_footnote'] = self.footnote

    def _cells_to_markdown(self) -> str:
        if not self.cells:
            return ''
        rows: dict = {}
        for cell in self.cells:
            rows.setdefault(cell.row, []).append(cell)
        md_lines = []
        first_row = min(rows.keys())
        for row_idx in sorted(rows.keys()):
            row_cells = sorted(rows[row_idx], key=lambda c: c.col)
            md_lines.append('| ' + ' | '.join(c.text for c in row_cells) + ' |')
            if row_idx == first_row:
                md_lines.append('|' + '|'.join([' --- '] * len(row_cells)) + '|')
        return '\n'.join(md_lines)


@dataclass
class FormulaBlock(Block):
    latex: str = ''
    inline: bool = False

    @property
    def ty(self) -> str:
        return 'formula'

    def text_content(self) -> str:
        return self.latex


@dataclass
class FigureBlock(Block):
    image_path: Optional[Path] = None
    caption: Optional[str] = None
    footnote: Optional[str] = None
    ocr_text: Optional[str] = None

    @property
    def ty(self) -> str:
        return 'figure'

    def text_content(self) -> str:
        parts = []
        if self.caption:
            parts.append(self.caption)
        if self.ocr_text:
            parts.append(self.ocr_text)
        return '\n'.join(parts)

    def update_metadata(self, d: dict) -> None:
        d['image_path'] = str(self.image_path) if self.image_path else None
        d['image_caption'] = self.caption


@dataclass
class CodeBlock(Block):
    language: Optional[str] = None
    text: str = ''
    caption: Optional[str] = None

    @property
    def ty(self) -> str:
        return 'code'

    def text_content(self) -> str:
        parts = []
        if self.caption:
            parts.append(self.caption)
        parts.append(self.text)
        return '\n'.join(parts)

    def update_metadata(self, d: dict) -> None:
        d['code_type'] = self.language
        d['code_caption'] = self.caption


@dataclass
class ListBlock(Block):
    ordered: bool = False
    items: List[str] = field(default_factory=list)

    @property
    def ty(self) -> str:
        return 'list'

    def text_content(self) -> str:
        lines = []
        for idx, item in enumerate(self.items, 1):
            prefix = f'{idx}. ' if self.ordered else '- '
            lines.append(f'{prefix}{item}')
        return '\n'.join(lines)
