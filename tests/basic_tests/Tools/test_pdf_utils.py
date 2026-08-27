import os
from pathlib import Path

from lazyllm.thirdparty import pypdf
from lazyllm.tools.pdf_utils import normalize_long_pdf, normalize_long_pdf_inplace


def _write_pdf(path: Path, sizes) -> None:
    writer = pypdf.PdfWriter()
    for width, height in sizes:
        writer.add_blank_page(width=width, height=height)
    with path.open('wb') as stream:
        writer.write(stream)


def test_normal_pdf_is_unchanged(tmp_path):
    source = tmp_path / 'normal.pdf'
    _write_pdf(source, [(100, 140), (100, 200)])

    result = normalize_long_pdf(source, tmp_path / 'output.pdf')

    assert result.path == source
    assert result.changed is False
    assert [segment.source_page for segment in result.segments] == [0, 1]
    assert not (tmp_path / 'output.pdf').exists()


def test_long_first_page_is_split_from_top_to_bottom(tmp_path):
    source = tmp_path / 'long.pdf'
    output = tmp_path / 'normalized.pdf'
    _write_pdf(source, [(100, 950), (100, 150)])

    result = normalize_long_pdf(source, output, max_aspect_ratio=3, target_aspect_ratio=2)
    normalized = pypdf.PdfReader(str(output))

    assert result.changed is True
    assert len(normalized.pages) == 6
    assert [round(float(page.mediabox.height)) for page in normalized.pages] == [200, 200, 200, 200, 150, 150]
    assert [segment.source_page for segment in result.segments] == [0, 0, 0, 0, 0, 1]
    assert [round(segment.top_offset) for segment in result.segments] == [0, 200, 400, 600, 800, 0]
    assert tuple(float(value) for value in normalized.pages[0].mediabox) == (0.0, 750.0, 100.0, 950.0)
    assert tuple(float(value) for value in normalized.pages[4].mediabox) == (0.0, 0.0, 100.0, 150.0)


def test_mixed_short_and_long_pages_are_all_checked_and_split(tmp_path):
    source = tmp_path / 'mixed.pdf'
    output = tmp_path / 'normalized.pdf'
    _write_pdf(source, [(100, 150), (100, 450), (100, 650)])

    result = normalize_long_pdf(source, output, max_aspect_ratio=3, target_aspect_ratio=2)
    normalized = pypdf.PdfReader(str(output))

    assert result.changed is True
    assert [round(float(page.mediabox.height)) for page in normalized.pages] == [150, 200, 200, 50, 200, 200, 200, 50]
    assert [segment.source_page for segment in result.segments] == [0, 1, 1, 1, 2, 2, 2, 2]
    assert [round(segment.top_offset) for segment in result.segments] == [0, 0, 200, 400, 0, 200, 400, 600]


def test_long_pdf_is_replaced_inplace(tmp_path):
    source = tmp_path / 'long.pdf'
    _write_pdf(source, [(100, 500)])
    source.chmod(0o640)

    result = normalize_long_pdf_inplace(source, max_aspect_ratio=3, target_aspect_ratio=2)
    replaced = pypdf.PdfReader(str(source))

    assert result is True
    assert len(replaced.pages) == 3
    assert [round(float(page.mediabox.height)) for page in replaced.pages] == [200, 200, 100]
    if os.name != 'nt':
        assert source.stat().st_mode & 0o777 == 0o640


def test_normal_pdf_inplace_returns_false(tmp_path):
    source = tmp_path / 'normal.pdf'
    _write_pdf(source, [(100, 200)])

    assert normalize_long_pdf_inplace(source) is False
