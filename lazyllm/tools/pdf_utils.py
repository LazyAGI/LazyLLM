import copy
import math
import os
import stat
import tempfile
import uuid
from pathlib import Path
from typing import List, NamedTuple, Optional

from lazyllm import LOG
from lazyllm.thirdparty import pypdf

_DEFAULT_TARGET_ASPECT_RATIO = math.sqrt(2)


class PdfPageSegment(NamedTuple):
    source_page: int
    top_offset: float
    source_width: float
    source_height: float


class LongPdfNormalization(NamedTuple):
    path: Path
    segments: List[PdfPageSegment]
    changed: bool


def _page_size(page) -> tuple:
    box = page.mediabox
    return float(box.width), float(box.height)


def _is_oversized_page(width: float, height: float, rotation: int, max_aspect_ratio: float) -> bool:
    return width > 0 and height / width > max_aspect_ratio and rotation not in (90, 270)


def _write_pdf(writer, output_path: Path) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f'.{output_path.name}.', suffix='.tmp', dir=output_path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer.write(stream)
        os.replace(temp_name, output_path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def normalize_long_pdf(
    input_path: Path,
    output_path: Optional[Path] = None,
    max_aspect_ratio: float = 3.0,
    target_aspect_ratio: float = _DEFAULT_TARGET_ASPECT_RATIO,
) -> LongPdfNormalization:
    input_path = Path(input_path)
    if max_aspect_ratio <= 0 or target_aspect_ratio <= 0:
        raise ValueError('PDF page aspect ratios must be positive')

    reader = pypdf.PdfReader(str(input_path))
    if not reader.pages:
        return LongPdfNormalization(input_path, [], False)

    pages = list(reader.pages)
    page_specs = [(*_page_size(page), int(page.rotation or 0) % 360) for page in pages]
    if not any(_is_oversized_page(*spec, max_aspect_ratio) for spec in page_specs):
        segments = [PdfPageSegment(i, 0.0, width, height)
                    for i, (width, height, _) in enumerate(page_specs)]
        return LongPdfNormalization(input_path, segments, False)

    output_path = Path(output_path) if output_path else input_path.with_suffix('.normalized.pdf')
    if output_path.resolve() == input_path.resolve():
        raise ValueError('Normalized PDF output must differ from the input path')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = pypdf.PdfWriter()
    segments = []
    for source_page, (page, spec) in enumerate(zip(pages, page_specs)):
        width, height, rotation = spec
        if not _is_oversized_page(width, height, rotation, max_aspect_ratio):
            writer.add_page(page)
            segments.append(PdfPageSegment(source_page, 0.0, width, height))
            continue

        left, bottom, right, top = (float(value) for value in page.mediabox)
        segment_height = width * target_aspect_ratio
        segment_count = max(1, math.ceil(height / segment_height))
        for segment_index in range(segment_count):
            segment_top = top - segment_index * segment_height
            segment_bottom = max(bottom, segment_top - segment_height)
            segment_page = copy.copy(page)
            box = pypdf.generic.RectangleObject((left, segment_bottom, right, segment_top))
            segment_page.mediabox = box
            segment_page.cropbox = box
            segment_page.trimbox = box
            segment_page.bleedbox = box
            segment_page.artbox = box
            writer.add_page(segment_page)
            segments.append(PdfPageSegment(source_page, segment_index * segment_height, width, height))

    _write_pdf(writer, output_path)
    return LongPdfNormalization(output_path, segments, True)


def normalize_long_pdf_inplace(
    input_path: Path,
    max_aspect_ratio: float = 3.0,
    target_aspect_ratio: float = _DEFAULT_TARGET_ASPECT_RATIO,
) -> bool:
    input_path = Path(input_path)
    source_mode = stat.S_IMODE(input_path.stat().st_mode)
    output_path = input_path.with_name(f'.{input_path.name}.{uuid.uuid4().hex}.normalized.pdf')
    try:
        result = normalize_long_pdf(
            input_path,
            output_path,
            max_aspect_ratio=max_aspect_ratio,
            target_aspect_ratio=target_aspect_ratio,
        )
        if not result.changed:
            return False
        os.chmod(result.path, source_mode)
        os.replace(result.path, input_path)
        LOG.info(f'[pdf_utils] Replaced oversized PDF in place: {input_path}')
        return True
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
