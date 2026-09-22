from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from importlib.resources import as_file
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..templates.latex import latex_template_resource, writer_latex_filter_resource
from .serialization import strip_heading_numbering


PANDOC_PATH_ENV = 'LAZYMIND_PANDOC_PATH'
RUNTIME_ROOT_ENV = 'LAZYMIND_RUNTIME_ROOT'
PANDOC_REQUIRED_VERSION = '3.11'
# LazyMind Markdown is GFM with tables, strikeout, task lists, footnotes,
# dollar-delimited math, autolinks, emoji, and raw HTML parsing. YAML metadata
# is disabled so document input cannot override controlled template metadata.
# GFM does not enable raw TeX, so LaTeX commands in Markdown remain plain text.
PANDOC_READER = 'gfm-yaml_metadata_block'
PANDOC_WRITER = 'latex'
DEFAULT_CONVERSION_TIMEOUT_SECONDS = 30.0
DEFAULT_VERSION_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_INPUT_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 20 * 1024 * 1024
_MAX_DIAGNOSTIC_CHARS = 4000
_VERSION_RE = re.compile(r'^pandoc\s+(?P<version>\d+(?:\.\d+)+(?:[-+][^\s]+)?)\s*$', re.IGNORECASE)
_MARKDOWN_ATX_HEADING_RE = re.compile(r'^(?P<prefix>[ \t]{0,3}#{2,6}[ \t]+)(?P<title>.*)$')
_MARKDOWN_FENCE_RE = re.compile(r'^[ \t]{0,3}(?P<fence>`{3,}|~{3,})')
_MARKDOWN_ANCHOR_RE = re.compile(
    r'^(?P<indent>[ \t]{0,3})<a(?P<attrs>\s+[^>]*)>\s*</a>[ \t]*$'
)
_MARKDOWN_ANCHOR_ID_RE = re.compile(
    r'\bid\s*=\s*(?P<quote>["\'])(?P<id>block-[A-Za-z0-9_.:%-]+)(?P=quote)',
    re.IGNORECASE,
)
_MARKDOWN_ANCHOR_KIND_RE = re.compile(
    r'\bdata-kind\s*=\s*(?P<quote>["\'])(?P<kind>[A-Za-z]+)(?P=quote)',
    re.IGNORECASE,
)
_MARKDOWN_ANCHOR_CAPTION_RE = re.compile(
    r'\bdata-caption\s*=\s*(?P<quote>["\'])(?P<caption>.*?)(?P=quote)',
    re.IGNORECASE,
)
_MARKDOWN_OBJECT_CAPTION_RES = {
    'table': re.compile(
        r'^[ \t]*(?:表[ \t]*\d+(?:\.\d+)*|[Tt]able[ \t]+\d+(?:\.\d+)*)'
        r'(?:[ \t]*[：:.])?[ \t]*(?P<caption>.*?)[ \t]*$'
    ),
    'code': re.compile(
        r'^[ \t]*(?:代码[ \t]*\d+(?:\.\d+)*|[Cc]ode[ \t]+\d+(?:\.\d+)*)'
        r'(?:[ \t]*[：:.])?[ \t]*(?P<caption>.*?)[ \t]*$'
    ),
}
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r'^[ \t]*\|?[ \t:|-]+\|?[ \t]*$')
_MARKDOWN_IMAGE_RE = re.compile(
    r'^[ \t]{0,3}!\[(?P<caption>(?:\\.|[^\]])*)\]\(.+\)[ \t]*$'
)
_MARKDOWN_DISPLAY_MATH_RE = re.compile(r'^[ \t]{0,3}\$\$')
_MARKDOWN_BIBLIOGRAPHY_ENTRY_RE = re.compile(
    r'^[ \t]*\\\[(?P<number>[1-9]\d*)\][ \t]+(?P<content>\S.*?)[ \t]*$'
)
_MARKDOWN_CITATION_RE = re.compile(r'\\\[(?P<number>[1-9]\d*)\]')
_GENERATED_ANCHOR_KIND = {
    'section': 'sec',
    'figure': 'figure',
    'table': 'table',
    'code': 'code',
}


def _validate_pandoc_input(markdown, command, timeout_seconds, max_input_bytes, max_output_bytes) -> None:
    if not isinstance(markdown, str):
        raise TypeError('Pandoc input must be a Markdown string.')
    if not command:
        raise ValueError('Pandoc command must not be empty.')
    if timeout_seconds <= 0:
        raise ValueError('Pandoc conversion timeout must be positive.')
    if max_input_bytes <= 0 or max_output_bytes <= 0:
        raise ValueError('Pandoc input and output limits must be positive.')

    input_size = len(markdown.encode('utf-8'))
    if input_size > max_input_bytes:
        raise PandocError(
            'PANDOC_INPUT_TOO_LARGE',
            'Markdown input exceeds the Pandoc conversion limit.',
            details={'input_bytes': input_size, 'max_input_bytes': max_input_bytes},
        )

def _associate_markdown_objects(lines, objects) -> list[_MarkdownObject]:
    associated: list[_MarkdownObject] = []
    for item in objects:
        anchor_search_from = item.start
        caption = item.caption
        caption_index = None
        previous = _previous_nonblank_line(lines, item.start)
        if previous is not None and item.kind in {'table', 'code'}:
            caption_kind, visible_caption = _numbered_object_caption(lines[previous])
            if caption_kind == item.kind and visible_caption is not None:
                caption = visible_caption
                caption_index = previous
                anchor_search_from = previous
        anchor_index = _previous_nonblank_line(lines, anchor_search_from)
        anchor = _MARKDOWN_ANCHOR_RE.match(lines[anchor_index]) if anchor_index is not None else None
        if anchor is None or _MARKDOWN_ANCHOR_ID_RE.search(anchor.group('attrs')) is None:
            anchor_index = None
        associated.append(_MarkdownObject(
            item.kind, item.start, item.end, caption, anchor_index, caption_index,
        ))
    return associated

@dataclass(frozen=True)
class _MarkdownObject:
    kind: str
    start: int
    end: int
    caption: str
    anchor_index: Optional[int] = None
    caption_index: Optional[int] = None


@dataclass(frozen=True)
class _MarkdownBibliography:
    heading_index: int
    anchor_index: Optional[int]
    title: str
    entries: tuple[tuple[int, str], ...]


class PandocError(RuntimeError):
    '''Error raised by the controlled Pandoc execution boundary.'''

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(code, message, details)
        self.code = code
        self.details = dict(details or {})

    def __str__(self) -> str:
        return self.args[1]


def _validated_executable(path: str, *, source: str) -> str:
    candidate = Path(path).expanduser()
    try:
        candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PandocError(
            'PANDOC_NOT_FOUND',
            f'Pandoc executable configured by {source} does not exist.',
            details={'source': source},
        ) from exc
    if not candidate.is_file():
        raise PandocError(
            'PANDOC_NOT_EXECUTABLE',
            f'Pandoc path configured by {source} is not a file.',
            details={'source': source},
        )
    if not os.access(candidate, os.X_OK):
        raise PandocError(
            'PANDOC_NOT_EXECUTABLE',
            f'Pandoc path configured by {source} is not executable.',
            details={'source': source},
        )
    return str(candidate)


def resolve_pandoc_path(
    explicit_path: Optional[str] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    '''Resolve Pandoc from an internal override, the runtime env, or development PATH.'''
    if explicit_path and explicit_path.strip():
        return _validated_executable(explicit_path.strip(), source='explicit_path')

    environment = os.environ if environ is None else environ
    configured_path = str(environment.get(PANDOC_PATH_ENV) or '').strip()
    if configured_path:
        return _validated_executable(configured_path, source=PANDOC_PATH_ENV)

    discovered_path = shutil.which('pandoc')
    if discovered_path:
        return _validated_executable(discovered_path, source='PATH')
    raise PandocError(
        'PANDOC_NOT_FOUND',
        f'Pandoc was not found. Configure {PANDOC_PATH_ENV} or install it on PATH.'
    )


def _diagnostic(value: object) -> str:
    text = str(value or '').strip()
    return text if len(text) <= _MAX_DIAGNOSTIC_CHARS else text[:_MAX_DIAGNOSTIC_CHARS] + '...'


def _pandoc_working_directory(
    *, environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Use a stable directory outside replaceable application bundles."""
    environment = os.environ if environ is None else environ
    runtime_root = str(environment.get(RUNTIME_ROOT_ENV) or '').strip()
    if runtime_root:
        candidate = Path(runtime_root).expanduser()
        if candidate.is_absolute() and candidate.is_dir():
            return str(candidate)
    return tempfile.gettempdir()


def _numbered_object_caption(line: str) -> tuple[Optional[str], Optional[str]]:
    for kind, pattern in _MARKDOWN_OBJECT_CAPTION_RES.items():
        match = pattern.match(line)
        if match is not None:
            return kind, match.group('caption')
    return None, None


def _rich_object_anchor(line: str, kind: str, caption: str) -> str:
    anchor = _MARKDOWN_ANCHOR_RE.match(line)
    if anchor is None or _MARKDOWN_ANCHOR_ID_RE.search(anchor.group('attrs')) is None:
        return line
    attrs = anchor.group('attrs')
    if _MARKDOWN_ANCHOR_KIND_RE.search(attrs) is None:
        attrs += f' data-kind="{kind}"'
    if kind != 'equation' and _MARKDOWN_ANCHOR_CAPTION_RE.search(attrs) is None:
        attrs += f' data-caption="{escape(caption, quote=True)}"'
    return f'{anchor.group("indent")}<a{attrs}></a>'


def _generated_object_anchor(
    kind: str,
    caption: str,
    counters: dict[str, int],
    used_ids: set[str],
) -> str:
    while True:
        counters[kind] += 1
        anchor_id = f'block-{_GENERATED_ANCHOR_KIND[kind]}-{counters[kind]:03d}'
        if anchor_id not in used_ids:
            used_ids.add(anchor_id)
            return (
                f'<a id="{anchor_id}" data-kind="{kind}" '
                f'data-caption="{escape(caption, quote=True)}"></a>'
            )


def _previous_nonblank_line(lines: Sequence[str], index: int) -> Optional[int]:
    index -= 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    return index if index >= 0 else None


def _clean_heading_caption(title: str, materialized_numbering: bool) -> str:
    clean = re.sub(r'[ \t]+#+[ \t]*$', '', title).strip()
    return strip_heading_numbering(clean) if materialized_numbering else clean


def _clean_figure_caption(caption: str) -> str:
    clean = re.sub(r'^图[ \t]*\d+(?:\.\d+)*(?:[ \t]*[：:.])?[ \t]*', '', caption)
    clean = re.sub(
        r'^[Ff]igure[ \t]+\d+(?:\.\d+)*(?:[ \t]*[：:.])?[ \t]*', '', clean,
    )
    return clean.strip()


def _scan_markdown_objects(
    lines: Sequence[str],
    *,
    materialized_numbering: bool,
) -> list[_MarkdownObject]:
    objects: list[_MarkdownObject] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = _MARKDOWN_FENCE_RE.match(line)
        if fence is not None:
            marker = fence.group('fence')
            end = index + 1
            while end < len(lines):
                closing = _MARKDOWN_FENCE_RE.match(lines[end])
                if closing is not None and closing.group('fence')[0] == marker[0] \
                        and len(closing.group('fence')) >= len(marker) \
                        and not lines[end][closing.end():].strip():
                    end += 1
                    break
                end += 1
            objects.append(_MarkdownObject('code', index, end, ''))
            index = end
            continue

        heading = _MARKDOWN_ATX_HEADING_RE.match(line)
        if heading is not None:
            objects.append(_MarkdownObject(
                'section', index, index + 1,
                _clean_heading_caption(heading.group('title'), materialized_numbering),
            ))
            index += 1
            continue

        image = _MARKDOWN_IMAGE_RE.match(line)
        if image is not None:
            objects.append(_MarkdownObject(
                'figure', index, index + 1,
                _clean_figure_caption(image.group('caption')),
            ))
            index += 1
            continue

        if index + 1 < len(lines) and '|' in line \
                and _MARKDOWN_TABLE_SEPARATOR_RE.match(lines[index + 1]) is not None:
            objects.append(_MarkdownObject('table', index, index + 2, ''))
            index += 2
            continue

        if _MARKDOWN_DISPLAY_MATH_RE.match(line) is not None:
            end = index + 1
            if line.count('$$') < 2:
                while end < len(lines):
                    if '$$' in lines[end]:
                        end += 1
                        break
                    end += 1
            objects.append(_MarkdownObject('equation', index, end, ''))
            index = end
            continue
        index += 1

    return _associate_markdown_objects(lines, objects)


def _citation_numbers(lines: Sequence[str], end: int) -> set[int]:
    numbers: set[int] = set()
    fence_marker: Optional[str] = None
    inline_code = re.compile(r'(?P<fence>`+).*?(?P=fence)')
    for line in lines[:end]:
        fence = _MARKDOWN_FENCE_RE.match(line)
        if fence is not None:
            marker = fence.group('fence')
            if fence_marker is None:
                fence_marker = marker
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker) \
                    and not line[fence.end():].strip():
                fence_marker = None
            continue
        if fence_marker is not None:
            continue
        prose = inline_code.sub('', line)
        numbers.update(
            int(match.group('number'))
            for match in _MARKDOWN_CITATION_RE.finditer(prose)
        )
    return numbers


def _find_numbered_bibliography(
    lines: Sequence[str],
    *,
    materialized_numbering: bool,
) -> Optional[_MarkdownBibliography]:
    sections = [
        item for item in _scan_markdown_objects(
            lines, materialized_numbering=materialized_numbering,
        )
        if item.kind == 'section'
    ]
    if not sections:
        return None
    section = sections[-1]
    entries: list[tuple[int, str]] = []
    for line in lines[section.end:]:
        if not line.strip():
            continue
        match = _MARKDOWN_BIBLIOGRAPHY_ENTRY_RE.match(line)
        if match is None:
            return None
        entries.append((int(match.group('number')), match.group('content')))
    if not entries:
        return None
    numbers = [number for number, _ in entries]
    if numbers != list(range(1, len(entries) + 1)):
        return None
    if not (_citation_numbers(lines, section.start) & set(numbers)):
        return None
    heading = _MARKDOWN_ATX_HEADING_RE.match(lines[section.start])
    if heading is None:
        return None
    return _MarkdownBibliography(
        heading_index=section.start,
        anchor_index=section.anchor_index,
        title=_clean_heading_caption(
            heading.group('title'), materialized_numbering,
        ),
        entries=tuple(entries),
    )


def _citation_link(number: int) -> str:
    return f'[\\[{number}\\]](#writer-cite-ref-{number:03d})'


def _rewrite_line_citations(line: str, entry_numbers: set[int]) -> str:
    parts: list[str] = []
    cursor = 0
    while cursor < len(line):
        opening = line.find('`', cursor)
        prose_end = len(line) if opening < 0 else opening
        parts.append(_MARKDOWN_CITATION_RE.sub(
            lambda match: _citation_link(int(match.group('number')))
            if int(match.group('number')) in entry_numbers else match.group(0),
            line[cursor:prose_end],
        ))
        if opening < 0:
            break
        fence_end = opening
        while fence_end < len(line) and line[fence_end] == '`':
            fence_end += 1
        fence = line[opening:fence_end]
        closing = line.find(fence, fence_end)
        if closing < 0:
            parts.append(line[opening:])
            break
        closing += len(fence)
        parts.append(line[opening:closing])
        cursor = closing
    return ''.join(parts)


def _normalize_numbered_bibliography(
    lines: Sequence[str],
    *,
    materialized_numbering: bool,
) -> list[str]:
    bibliography = _find_numbered_bibliography(
        lines, materialized_numbering=materialized_numbering,
    )
    if bibliography is None:
        return list(lines)

    entry_numbers = {number for number, _ in bibliography.entries}
    body: list[str] = []
    fence_marker: Optional[str] = None
    for index, line in enumerate(lines[:bibliography.heading_index]):
        if index == bibliography.anchor_index:
            continue
        fence = _MARKDOWN_FENCE_RE.match(line)
        if fence is not None:
            marker = fence.group('fence')
            if fence_marker is None:
                fence_marker = marker
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker) \
                    and not line[fence.end():].strip():
                fence_marker = None
            body.append(line)
            continue
        if fence_marker is None:
            line = _rewrite_line_citations(line, entry_numbers)
        body.append(line)

    while body and not body[-1].strip():
        body.pop()
    body.extend((
        '',
        '<a data-writer-bibliography="begin" '
        f'data-title="{escape(bibliography.title, quote=True)}"></a>',
        '',
    ))
    for number, content in bibliography.entries:
        body.extend((
            f'<a data-writer-bibitem="ref-{number:03d}"></a>',
            '',
            content,
            '',
        ))
    while body and not body[-1].strip():
        body.pop()
    return body


def normalize_writer_markdown_for_latex(
    markdown: str,
    *,
    materialized_numbering: bool = True,
) -> str:
    '''Canonicalize Writer Markdown for LaTeX without changing the stored source.'''
    newline = '\r\n' if '\r\n' in markdown else '\n'
    trailing_newline = markdown.endswith(('\n', '\r'))
    lines = _normalize_numbered_bibliography(
        markdown.splitlines(), materialized_numbering=materialized_numbering,
    )
    used_ids = {
        match.group('id')
        for line in lines
        for match in _MARKDOWN_ANCHOR_ID_RE.finditer(line)
    }
    counters = {kind: 0 for kind in _GENERATED_ANCHOR_KIND}
    objects = _scan_markdown_objects(
        lines, materialized_numbering=materialized_numbering,
    )
    replacements: dict[int, str] = {}
    insertions: dict[int, str] = {}
    skipped: set[int] = set()
    headings: dict[int, str] = {}
    for item in objects:
        if item.anchor_index is not None:
            replacements[item.anchor_index] = _rich_object_anchor(
                lines[item.anchor_index], item.kind, item.caption,
            )
            skipped.update(range(item.anchor_index + 1, item.start))
        elif item.kind != 'equation':
            insertions[item.start] = _generated_object_anchor(
                item.kind, item.caption, counters, used_ids,
            )
        if item.caption_index is not None:
            skipped.add(item.caption_index)
        if item.kind == 'section' and materialized_numbering:
            heading = _MARKDOWN_ATX_HEADING_RE.match(lines[item.start])
            if heading is not None:
                headings[item.start] = heading.group('prefix') + item.caption

    normalized: list[str] = []
    for index, line in enumerate(lines):
        if index in skipped:
            continue
        if index in replacements:
            normalized.extend((replacements[index], ''))
            continue
        if index in insertions:
            normalized.extend((insertions[index], ''))
        normalized.append(headings.get(index, line))
    result = newline.join(normalized)
    return result + newline if trailing_newline else result


@lru_cache(maxsize=8)
def _check_pandoc_version_cached(
    pandoc_path: str,
    required_version: str,
    timeout_seconds: float,
) -> str:
    try:
        result = subprocess.run(
            [pandoc_path, '--version'],
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_seconds,
            shell=False,
            check=False,
            cwd=_pandoc_working_directory(),
        )
    except subprocess.TimeoutExpired as exc:
        raise PandocError(
            'PANDOC_TIMEOUT',
            'Pandoc version check timed out.',
            details={'stage': 'version_check', 'timeout_seconds': timeout_seconds},
        ) from exc
    except FileNotFoundError as exc:
        raise PandocError(
            'PANDOC_NOT_FOUND', 'Pandoc executable disappeared before version check.'
        ) from exc
    except PermissionError as exc:
        raise PandocError('PANDOC_NOT_EXECUTABLE', 'Pandoc executable cannot be started.') from exc
    except OSError as exc:
        raise PandocError(
            'PANDOC_NOT_EXECUTABLE',
            'Pandoc executable cannot be started.',
            details={'reason': _diagnostic(exc)},
        ) from exc

    if result.returncode != 0:
        raise PandocError(
            'PANDOC_VERSION_UNSUPPORTED',
            'Pandoc version check failed.',
            details={'return_code': result.returncode, 'diagnostic': _diagnostic(result.stderr)},
        )
    first_line = (result.stdout or '').splitlines()[0].strip() if result.stdout else ''
    match = _VERSION_RE.fullmatch(first_line)
    if match is None:
        raise PandocError(
            'PANDOC_VERSION_UNSUPPORTED',
            'Pandoc returned an unrecognized version string.',
            details={'version_line': _diagnostic(first_line)},
        )
    actual_version = match.group('version')
    if actual_version != required_version:
        raise PandocError(
            'PANDOC_VERSION_UNSUPPORTED',
            f'Pandoc {required_version} is required, but {actual_version} was found.',
            details={'required_version': required_version, 'actual_version': actual_version},
        )
    return actual_version


def check_pandoc_version(
    pandoc_path: str,
    *,
    required_version: str = PANDOC_REQUIRED_VERSION,
    timeout_seconds: float = DEFAULT_VERSION_TIMEOUT_SECONDS,
) -> str:
    '''Return the validated Pandoc version, caching successful checks by executable path.'''
    if timeout_seconds <= 0:
        raise ValueError('Pandoc version timeout must be positive.')
    return _check_pandoc_version_cached(pandoc_path, required_version, timeout_seconds)


def build_pandoc_command(
    pandoc_path: str,
    template_path: str,
    filter_path: str,
) -> list[str]:
    '''Build the fixed Markdown-to-LaTeX command; callers cannot inject CLI options.'''
    return [
        pandoc_path,
        f'--from={PANDOC_READER}',
        f'--to={PANDOC_WRITER}',
        '--standalone',
        '--number-sections',
        '--wrap=none',
        '--template',
        template_path,
        '--lua-filter',
        filter_path,
    ]


def _command_option_value(command: Sequence[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError:
        return ''
    return str(command[index + 1]) if index + 1 < len(command) else ''


def _conversion_failure_code(command: Sequence[str], diagnostic: str) -> str:
    '''Classify failures emitted by the controlled template and Lua filter.'''
    normalized = diagnostic.lower()
    filter_path = _command_option_value(command, '--lua-filter').lower()
    template_path = _command_option_value(command, '--template').lower()
    if 'error running filter' in normalized or (filter_path and filter_path in normalized):
        return 'PANDOC_FILTER_FAILED'
    if 'error compiling template' in normalized or (template_path and template_path in normalized):
        return 'PANDOC_TEMPLATE_INVALID'
    return 'PANDOC_CONVERSION_FAILED'


def run_pandoc(
    markdown: str,
    command: Sequence[str],
    *,
    timeout_seconds: float = DEFAULT_CONVERSION_TIMEOUT_SECONDS,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> str:
    '''Execute a pre-built Pandoc command using stdin/stdout under fixed resource limits.'''
    _validate_pandoc_input(markdown, command, timeout_seconds, max_input_bytes, max_output_bytes)

    try:
        result = subprocess.run(
            list(command),
            input=markdown,
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_seconds,
            shell=False,
            check=False,
            cwd=_pandoc_working_directory(),
        )
    except subprocess.TimeoutExpired as exc:
        raise PandocError(
            'PANDOC_TIMEOUT',
            'Pandoc conversion timed out.',
            details={'stage': 'conversion', 'timeout_seconds': timeout_seconds},
        ) from exc
    except FileNotFoundError as exc:
        raise PandocError(
            'PANDOC_NOT_FOUND', 'Pandoc executable disappeared before conversion.'
        ) from exc
    except PermissionError as exc:
        raise PandocError('PANDOC_NOT_EXECUTABLE', 'Pandoc executable cannot be started.') from exc
    except OSError as exc:
        raise PandocError(
            'PANDOC_CONVERSION_FAILED',
            'Pandoc process could not be started.',
            details={'diagnostic': _diagnostic(exc)},
        ) from exc

    if result.returncode != 0:
        diagnostic = _diagnostic(result.stderr)
        code = _conversion_failure_code(command, diagnostic)
        raise PandocError(
            code,
            {
                'PANDOC_FILTER_FAILED': 'The bundled Lua Filter failed.',
                'PANDOC_TEMPLATE_INVALID': 'The selected LaTeX template is invalid.',
            }.get(code, 'Pandoc failed to convert Markdown to LaTeX.'),
            details={'return_code': result.returncode, 'diagnostic': diagnostic},
        )

    latex = result.stdout or ''
    output_size = len(latex.encode('utf-8'))
    if output_size > max_output_bytes:
        raise PandocError(
            'PANDOC_OUTPUT_TOO_LARGE',
            'LaTeX output exceeds the Pandoc conversion limit.',
            details={'output_bytes': output_size, 'max_output_bytes': max_output_bytes},
        )
    return latex


def markdown_to_latex(
    markdown: str,
    *,
    language: str = 'zh-CN',
    pandoc_path: Optional[str] = None,
    required_version: str = PANDOC_REQUIRED_VERSION,
    timeout_seconds: float = DEFAULT_CONVERSION_TIMEOUT_SECONDS,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    materialized_numbering: bool = True,
) -> str:
    '''Convert a Markdown string to standalone LaTeX with the supported Pandoc version.'''
    executable = resolve_pandoc_path(pandoc_path)
    check_pandoc_version(executable, required_version=required_version)
    try:
        template_resource = latex_template_resource(language)
        filter_resource = writer_latex_filter_resource()
    except ValueError as exc:
        raise PandocError(
            'PANDOC_TEMPLATE_INVALID',
            'The selected LaTeX template is unavailable.',
            details={'language': language},
        ) from exc
    with ExitStack() as resources:
        try:
            template_path = resources.enter_context(as_file(template_resource))
            if not template_path.is_file():
                raise FileNotFoundError(str(template_path))
        except OSError as exc:
            raise PandocError(
                'PANDOC_TEMPLATE_INVALID',
                'The selected LaTeX template is unavailable.',
                details={'language': language},
            ) from exc
        try:
            filter_path = resources.enter_context(as_file(filter_resource))
            if not filter_path.is_file():
                raise FileNotFoundError(str(filter_path))
        except OSError as exc:
            raise PandocError(
                'PANDOC_FILTER_FAILED',
                'The bundled Lua Filter is unavailable.',
            ) from exc
        return run_pandoc(
            normalize_writer_markdown_for_latex(
                markdown,
                materialized_numbering=materialized_numbering,
            ),
            build_pandoc_command(executable, str(template_path), str(filter_path)),
            timeout_seconds=timeout_seconds,
            max_input_bytes=max_input_bytes,
            max_output_bytes=max_output_bytes,
        )


__all__ = [
    'PandocError',
    'markdown_to_latex',
    'normalize_writer_markdown_for_latex',
]
