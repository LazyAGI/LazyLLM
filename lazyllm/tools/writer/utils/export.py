from __future__ import annotations

import html
import re
from typing import Any

from lazyllm.thirdparty import mistune

from ..data_models.writer_ir import WriterDocument, WriterSpan
from ..numbering import (
    MARKDOWN_ANCHOR_RE, build_numbering_view_from_markdown, compute_numbering, materialize_markdown,
    strip_markdown_heading_numbering_config,
)
from .conversion import convert_writer_content, render_document_markdown
from .tables import validate_writer_tables


_TEX_ESCAPES = {
    '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '$': r'\$',
    '&': r'\&', '#': r'\#', '%': r'\%', '_': r'\_',
    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
}
_MATH_SOURCE = re.compile(r'(?<!\\)(\$\$[\s\S]+?\$\$|\$(?!\$)[^\n$]+?\$)')
_INTERNAL_LINK = re.compile(r'(?<!!)\[((?:\\.|[^\]\\])*)\]\(\s*<?#block-[^\s)>]+>?\s*\)')
_INLINE_CODE = re.compile(r'(`+).*?\1(?!`)', re.DOTALL)
_CODE_FENCE = re.compile(r'^\s{0,3}(`{3,}|~{3,})')


def _copyable_markdown(markdown: str) -> str:
    lines = []
    fence = ''
    for line in markdown.splitlines():
        marker = _CODE_FENCE.match(line)
        if fence:
            lines.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) \
                    and not line[marker.end():].strip():
                fence = ''
            continue
        if marker:
            fence = marker[1]
            lines.append(line)
            continue
        # Code examples are content, not editor metadata.
        parts = []
        cursor = 0
        for code in _INLINE_CODE.finditer(line):
            parts.append(_INTERNAL_LINK.sub(r'\1', MARKDOWN_ANCHOR_RE.sub('', line[cursor:code.start()])))
            parts.append(code[0])
            cursor = code.end()
        parts.append(_INTERNAL_LINK.sub(r'\1', MARKDOWN_ANCHOR_RE.sub('', line[cursor:])))
        cleaned = ''.join(parts)
        if cleaned.strip():
            lines.append(cleaned)
    return '\n'.join(lines).strip() + '\n'


def _escape_tex(value: str) -> str:
    return ''.join(_TEX_ESCAPES.get(char, char) for char in value)


def _render_tokens(tokens: list[dict[str, Any]], output_format: str) -> str:  # noqa: C901
    latex = output_format == 'latex'
    escape = _escape_tex if latex else str
    parts = []
    for token in tokens:
        kind = token['type']
        raw = str(token.get('raw') or '')
        attrs = token.get('attrs') or {}
        children = token.get('children') or []
        body = _render_tokens(children, output_format) if children and kind not in {'list', 'table'} else ''
        if kind == 'text':
            value = escape(raw)
        elif kind in {'paragraph', 'block_text'}:
            value = body + ('\\par\n' if latex else '\n') if body.strip() else ''
        elif kind == 'heading':
            commands = ['section', 'subsection', 'subsubsection', 'paragraph', 'subparagraph', 'subparagraph']
            command = commands[max(0, min(5, int(attrs.get('level', 1)) - 1))]
            # Heading numbers have already been materialized by Writer.
            value = f'\\{command}*{{{body}}}\n' if latex else body + '\n'
        elif kind in {'strong', 'emphasis', 'strikethrough'}:
            command = {'strong': 'textbf', 'emphasis': 'emph', 'strikethrough': 'textnormal'}[kind]
            value = f'\\{command}{{{body}}}' if latex else body
        elif kind == 'codespan':
            value = f'\\texttt{{{escape(raw)}}}' if latex else raw
        elif kind == 'block_code':
            if latex:
                # Escaped text avoids a code sample terminating a verbatim environment.
                lines = '\\\\\n'.join(_escape_tex(line).replace(' ', r'\ ') for line in raw.rstrip('\n').split('\n'))
                value = f'\\begin{{quote}}\n\\ttfamily\n{lines}\n\\end{{quote}}\n'
            else:
                value = raw.rstrip('\n') + '\n'
        elif kind in {'inline_math', 'block_math'}:
            value = (f'${raw}$' if kind == 'inline_math' else f'\\[\n{raw}\n\\]\n') if latex else raw
            if not latex and kind == 'block_math':
                value += '\n'
        elif kind == 'link':
            url = str(attrs.get('url') or '')
            if url.startswith('#block-'):
                value = body
            else:
                value = f'{body} ({escape(url)})' if url else body
        elif kind == 'image':
            # Copying text cannot transfer image files; retain the resource locator.
            url = str(attrs.get('url') or '')
            value = f'[{body}] ({escape(url)})'
        elif kind == 'list':
            ordered = bool(attrs.get('ordered'))
            start = int(attrs.get('start') or 1)
            items = []
            for index, child in enumerate(children):
                item = _render_tokens(child.get('children') or [], output_format).strip()
                if child['type'] == 'task_list_item':
                    item = ('[x] ' if (child.get('attrs') or {}).get('checked') else '[ ] ') + item
                marker = f'{start + index}.' if ordered else '-'
                items.append(f'\\item[{marker}] {item}' if latex else f'{marker} {item}')
            joined = '\n'.join(items)
            value = f'\\begin{{itemize}}\n{joined}\n\\end{{itemize}}\n' if latex else joined + '\n'
        elif kind in {'list_item', 'task_list_item'}:
            value = body
        elif kind == 'block_quote':
            value = f'\\begin{{quote}}\n{body}\\end{{quote}}\n' if latex else body
        elif kind == 'table':
            rows = []
            for child in children:
                rows.extend([child] if child['type'] == 'table_head' else child.get('children') or [])
            cells = [[_render_tokens(cell.get('children') or [], output_format) for cell in row['children']]
                     for row in rows]
            if latex:
                width = max((len(row) for row in cells), default=1)
                lines = [' & '.join(row + [''] * (width - len(row))) + r' \\' for row in cells]
                value = '\\begin{tabular}{' + 'l' * width + '}\n' + '\n'.join(lines) + '\n\\end{tabular}\n'
            else:
                value = '\n'.join('\t'.join(row) for row in cells) + '\n'
        elif kind in {'table_head', 'table_body', 'table_row', 'table_cell'}:
            value = body
        elif kind == 'blank_line':
            value = ''
        elif kind in {'softbreak', 'linebreak'}:
            value = '\n'
        elif kind == 'thematic_break':
            value = '\\par\\noindent\\rule{\\linewidth}{0.4pt}\n' if latex else '---\n'
        elif kind in {'inline_html', 'block_html'}:
            if re.fullmatch(r'\s*(?:<a\b[^>]*></a>|<!--.*?-->)\s*', raw, re.DOTALL):
                value = ''
            else:
                value = escape(html.unescape(re.sub(r'<[^>]*>', '', raw)))
        else:
            raise ValueError(f'Unsupported document element for {output_format}: {kind!r}.')
        parts.append(value)
    return ''.join(parts)


def _export_markdown(document: WriterDocument) -> str:
    result = document.model_copy(deep=True)
    for block in result.iter_blocks():
        if block.type not in {'paragraph', 'heading', 'quote', 'list_item', 'table_cell'}:
            continue
        spans = block.spans if ''.join(span.text for span in block.spans) == block.content else [
            WriterSpan(text=block.content),
        ]
        expanded = []
        for span in spans:
            if any(span.style.get(key) for key in ('inline_code', 'code', 'notion:rich_text_type')):
                expanded.append(span)
                continue
            for part in _MATH_SOURCE.split(span.text):
                if part:
                    style = {**span.style, 'math_source': True} if _MATH_SOURCE.fullmatch(part) else span.style
                    expanded.append(WriterSpan(text=part, style=style))
        block.spans = expanded
    return render_document_markdown(result)


def export_writer_document(
    document: WriterDocument, output_format: str, *, markdown_source: str | None = None,
) -> str:
    if output_format not in {'markdown', 'latex', 'text'}:
        raise ValueError(f'Unsupported Writer output format: {output_format!r}.')
    validate_writer_tables(document)
    for block in document.iter_blocks():
        if not block.content and not block.children and not block.references \
                and block.provider_payload and block.type not in {'divider'}:
            raise ValueError(f'Document block {block.node_id!r} has no portable content.')
    markdown = markdown_source if markdown_source is not None else _export_markdown(document)
    view = build_numbering_view_from_markdown(markdown)
    markdown = strip_markdown_heading_numbering_config(materialize_markdown(markdown, view, compute_numbering(view)))
    if output_format == 'markdown':
        return _copyable_markdown(markdown)
    if output_format == 'latex':
        return convert_writer_content(markdown, 'markdown', 'latex')
    parser = mistune.create_markdown(renderer='ast', plugins=['table', 'strikethrough', 'math', 'task_lists'])
    return _render_tokens(parser(markdown), output_format).strip() + '\n'
