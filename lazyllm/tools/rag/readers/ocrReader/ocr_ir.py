from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
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


@dataclass
class SectionPath:
    anchors: List[str] = field(default_factory=list)
    level: int = 0


# ----- Block base with shared fields -----
@dataclass
class Block:
    page: PageRef
    section: SectionPath = field(default_factory=SectionPath)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['ty'] = self.ty
        return d

    @property
    def ty(self) -> str:
        raise NotImplementedError

    def text_content(self) -> str:
        raise NotImplementedError


@dataclass
class HeadingBlock(Block):
    level: int = 0
    text: str = ''
    anchor: str = ''

    @property
    def ty(self) -> str:
        return 'heading'

    def text_content(self) -> str:
        return self.text


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

    def _cells_to_markdown(self) -> str:
        if not self.cells:
            return ''
        # Simple markdown generation from cells
        rows: dict = {}
        for cell in self.cells:
            rows.setdefault(cell.row, []).append(cell)
        md_lines = []
        for row_idx in sorted(rows.keys()):
            row_cells = sorted(rows[row_idx], key=lambda c: c.col)
            md_lines.append('| ' + ' | '.join(c.text for c in row_cells) + ' |')
            if row_idx == min(rows.keys()):
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


BlockType = Union[HeadingBlock, ParagraphBlock, TableBlock, FormulaBlock, FigureBlock, CodeBlock, ListBlock]
