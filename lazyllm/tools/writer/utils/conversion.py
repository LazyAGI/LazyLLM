from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

from lazyllm.thirdparty import mistune

from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterSpan
from ..numbering import MARKDOWN_ANCHOR_RE
from .artifact import deserialize_artifact_json, serialize_artifact_json


WriterSourceFormat = Literal['markdown', 'lmd', 'writer_document']
WriterTargetFormat = Literal['markdown', 'lmd']


@dataclass
class _InlineContent:
    content: str = ''
    spans: List[WriterSpan] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)


def _normalize_document_id(value: str) -> str:
    normalized = re.sub(r'[^a-zA-Z0-9_-]+', '-', value or '').strip('-')
    return normalized or 'writer-document'


def _semantic_provider_payload(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith('markdown_')}


def _block_snapshot(block: WriterBlock) -> Dict[str, Any]:
    extras = block.model_extra or {}
    return {
        'type': block.type,
        'content': block.content,
        'spans': [span.model_dump(exclude_defaults=True) for span in block.spans],
        'numbering': block.numbering,
        'references': block.references,
        'language': extras.get('language', ''),
        'editable': block.editable,
        'provider_payload': _semantic_provider_payload(block.provider_payload),
        'children': [_block_snapshot(child) for child in block.children],
    }


def _signature(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str,
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _block_signature(block: WriterBlock) -> str:
    return _signature(_block_snapshot(block))


def _document_signature(document: WriterDocument) -> str:
    return _signature({
        'title': document.title,
        'blocks': [_block_snapshot(block) for block in document.blocks],
    })


def _append_span(result: _InlineContent, text: str, style: Dict[str, Any]) -> None:
    if not text:
        return
    result.content += text
    if result.spans and result.spans[-1].style == style:
        result.spans[-1].text += text
    else:
        result.spans.append(WriterSpan(text=text, style=dict(style)))


def _inline_content(
    tokens: Optional[Iterable[Dict[str, Any]]],
    style: Optional[Dict[str, Any]] = None,
    result: Optional[_InlineContent] = None,
) -> _InlineContent:
    output = result or _InlineContent()
    inherited = dict(style or {})
    for token in tokens or []:
        token_type = str(token.get('type') or '')
        children = token.get('children') or []
        if token_type in {'strong', 'emphasis', 'strikethrough'}:
            child_style = dict(inherited)
            child_style[{
                'strong': 'bold',
                'emphasis': 'italic',
                'strikethrough': 'strikethrough',
            }[token_type]] = True
            _inline_content(children, child_style, output)
            continue
        if token_type == 'codespan':
            _append_span(output, str(token.get('raw') or ''), {**inherited, 'inline_code': True})
            continue
        if token_type == 'link':
            url = str((token.get('attrs') or {}).get('url') or '')
            if url.startswith('#block-'):
                output.spans.append(WriterSpan(
                    text='',
                    style={
                        'link': {
                            'type': 'internal_ref',
                            'target_node_id': url.removeprefix('#block-'),
                        },
                    },
                ))
                continue
            start = len(output.content)
            _inline_content(children, inherited, output)
            attrs = token.get('attrs') or {}
            reference = {
                'type': 'link', 'url': str(attrs.get('url') or ''),
                'start': start, 'end': len(output.content),
            }
            if attrs.get('title'):
                reference['title'] = str(attrs['title'])
            output.references.append(reference)
            continue
        if token_type == 'image':
            attrs = token.get('attrs') or {}
            alt_content = _inline_content(children).content
            start = len(output.content)
            _append_span(output, alt_content, inherited)
            reference = {
                'type': 'markdown_image', 'url': str(attrs.get('url') or ''),
                'alt': alt_content, 'start': start, 'end': len(output.content),
            }
            if attrs.get('title'):
                reference['title'] = str(attrs['title'])
            output.references.append(reference)
            continue
        if token_type in {'softbreak', 'linebreak'}:
            _append_span(output, '\n', inherited)
            if token_type == 'linebreak':
                output.references.append({'type': 'hard_break', 'offset': len(output.content) - 1})
            continue
        if token_type in {'inline_html', 'html_inline'}:
            raw = str(token.get('raw') or '')
            start = len(output.content)
            _append_span(output, raw, inherited)
            output.references.append({
                'type': 'html_inline', 'source': raw,
                'start': start, 'end': len(output.content),
            })
            continue
        if children:
            _inline_content(children, inherited, output)
            continue
        _append_span(output, str(token.get('raw') or ''), inherited)
    return output


def _slice_inline(value: _InlineContent, start: int) -> _InlineContent:
    spans: List[WriterSpan] = []
    offset = 0
    for span in value.spans:
        end = offset + len(span.text)
        if end > start:
            spans.append(span.model_copy(update={'text': span.text[max(0, start - offset):]}))
        offset = end
    references: List[Dict[str, Any]] = []
    for reference in value.references:
        reference_start = reference.get('start')
        reference_end = reference.get('end')
        if not isinstance(reference_start, (int, float)) or not isinstance(reference_end, (int, float)):
            references.append(reference)
        elif reference_end > start:
            references.append({
                **reference,
                'start': max(0, int(reference_start) - start),
                'end': max(0, int(reference_end) - start),
            })
    return _InlineContent(value.content[start:], spans, references)


class _MarkdownParser:
    def __init__(self, markdown: str, document_id: str):
        self.markdown = (markdown or '').replace('\r\n', '\n').replace('\r', '\n')
        self.document_id = _normalize_document_id(document_id)
        self.sequence = 0
        self.used_ids: set[str] = set()
        self.pending_anchor_ids: List[str] = []
        self.title = ''
        self.emitted = False
        self.parser = mistune.create_markdown(
            renderer='ast', plugins=['table', 'strikethrough'],
        )

    def next_id(self, block_type: str) -> str:
        if self.pending_anchor_ids:
            candidate = self.pending_anchor_ids.pop(0)
        else:
            self.sequence += 1
            safe_type = re.sub(r'[^a-zA-Z0-9_-]+', '-', block_type).strip('-') or 'block'
            candidate = f'{self.document_id}-{safe_type}-{self.sequence}'
        if candidate in self.used_ids:
            raise ValueError(f'duplicate Markdown anchor target: {candidate!r}')
        self.used_ids.add(candidate)
        return candidate

    @staticmethod
    def leading_anchors(
        tokens: List[Dict[str, Any]],
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        index = 0
        raw = ''
        while index < len(tokens) and tokens[index].get('type') in {'inline_html', 'html_inline'}:
            raw += str(tokens[index].get('raw') or '')
            index += 1
        anchors = [target.removeprefix('block-') for target in MARKDOWN_ANCHOR_RE.findall(raw)]
        if anchors and not MARKDOWN_ANCHOR_RE.sub('', raw).strip():
            return anchors, tokens[index:]
        return [], tokens

    def block(
        self,
        block_type: str,
        content: str,
        *,
        children: Optional[List[WriterBlock]] = None,
        spans: Optional[List[WriterSpan]] = None,
        references: Optional[List[Dict[str, Any]]] = None,
        numbering: Optional[Dict[str, Any]] = None,
        provider_payload: Optional[Dict[str, Any]] = None,
        editable: bool = True,
        **extras: Any,
    ) -> WriterBlock:
        return WriterBlock(
            node_id=self.next_id(block_type),
            type=block_type,
            content=content,
            stage='final',
            children=children or [],
            spans=spans or [],
            references=references or [],
            numbering=numbering or {},
            provider_payload=provider_payload or {},
            editable=editable,
            **extras,
        )

    @staticmethod
    def table_markdown(token: Dict[str, Any]) -> str:
        head: List[Dict[str, Any]] = []
        rows: List[List[Dict[str, Any]]] = []
        for child in token.get('children') or []:
            if child.get('type') == 'table_head':
                head = list(child.get('children') or [])
            elif child.get('type') == 'table_body':
                rows.extend(list(row.get('children') or []) for row in child.get('children') or [])
        if not head:
            return ''

        def cell_text(cell: Dict[str, Any]) -> str:
            return _inline_content(cell.get('children') or []).content.replace('|', '\\|').replace('\n', '<br>')

        header = f'| {" | ".join(cell_text(cell) for cell in head)} |'
        dividers = []
        for cell in head:
            align = str((cell.get('attrs') or {}).get('align') or '')
            dividers.append({'left': ':---', 'right': '---:', 'center': ':---:'}.get(align, '---'))
        lines = [header, f'| {" | ".join(dividers)} |']
        lines.extend(f'| {" | ".join(cell_text(cell) for cell in row)} |' for row in rows)
        return '\n'.join(lines)

    def parse_sequence(self, tokens: Iterable[Dict[str, Any]]) -> List[WriterBlock]:
        blocks: List[WriterBlock] = []
        for token in tokens:
            token_type = str(token.get('type') or '')
            if token_type == 'blank_line':
                continue
            if token_type == 'heading':
                rich = _inline_content(token.get('children') or [])
                level = max(1, min(6, int((token.get('attrs') or {}).get('level') or 1)))
                if level == 1 and not self.emitted and not self.title:
                    self.title = rich.content.strip()
                else:
                    blocks.append(self.block(
                        'heading', rich.content, spans=rich.spans,
                        references=rich.references,
                        numbering={'level': max(level - 1, 1)},
                    ))
                self.emitted = True
                continue
            if token_type in {'paragraph', 'block_text'}:
                anchors, children = self.leading_anchors(list(token.get('children') or []))
                self.pending_anchor_ids.extend(anchors)
                if not children:
                    continue
                rich = _inline_content(children)
                visible = [item for item in children if item.get('type') not in {'softbreak', 'linebreak'}]
                if len(visible) == 1 and visible[0].get('type') == 'image':
                    image = visible[0]
                    attrs = image.get('attrs') or {}
                    alt = _inline_content(image.get('children') or []).content
                    reference = {
                        'type': 'media_asset', 'url': str(attrs.get('url') or ''),
                        'path': str(attrs.get('url') or ''), 'alt': alt,
                    }
                    if attrs.get('title'):
                        reference['title'] = str(attrs['title'])
                    blocks.append(self.block('image', alt, references=[reference]))
                else:
                    task = re.match(r'^\[([ xX])\]\s+', rich.content)
                    if task:
                        rich = _slice_inline(rich, task.end())
                    stripped = rich.content.strip()
                    is_math = bool(re.fullmatch(r'(?:\$\$[\s\S]*\$\$|\\\[[\s\S]*\\\])', stripped))
                    blocks.append(self.block(
                        'math' if is_math else 'paragraph',
                        stripped if is_math else rich.content,
                        spans=[] if is_math else rich.spans,
                        references=[] if is_math else rich.references,
                        numbering={
                            'task': True, 'checked': task.group(1).lower() == 'x',
                        } if task else None,
                        editable=not is_math,
                    ))
                self.emitted = True
                continue
            if token_type == 'list':
                attrs = token.get('attrs') or {}
                ordered = bool(attrs.get('ordered'))
                start = int(attrs.get('start') or 1)
                marker = str(token.get('bullet') or ('.' if ordered else '-'))
                ordinal = start
                for item_token in token.get('children') or []:
                    if item_token.get('type') != 'list_item':
                        continue
                    item_parts = self.parse_sequence(item_token.get('children') or [])
                    first = item_parts.pop(0) if item_parts else None
                    content = first.content if first else ''
                    spans = first.spans if first else []
                    references = first.references if first else []
                    task = re.match(r'^\[([ xX])\]\s+', content)
                    task_checked: Optional[bool] = None
                    if first and first.numbering.get('task'):
                        task_checked = bool(first.numbering.get('checked'))
                    if task:
                        rich = _slice_inline(_InlineContent(content, spans, references), task.end())
                        content, spans, references = rich.content, rich.spans, rich.references
                        task_checked = task.group(1).lower() == 'x'
                    numbering: Dict[str, Any] = {
                        'ordered': ordered,
                        'marker': f'{ordinal}.' if ordered else marker,
                    }
                    if ordered:
                        numbering.update({'number': [ordinal], 'start': start})
                    if task_checked is not None:
                        numbering.update({'task': True, 'checked': task_checked})
                    blocks.append(self.block(
                        'list_item', content, spans=spans, references=references,
                        children=[*(first.children if first else []), *item_parts],
                        numbering=numbering,
                    ))
                    ordinal += 1
                self.emitted = True
                continue
            if token_type == 'block_quote':
                parts = self.parse_sequence(token.get('children') or [])
                first = parts.pop(0) if parts else None
                blocks.append(self.block(
                    'quote', first.content if first else '',
                    spans=first.spans if first else [],
                    references=first.references if first else [],
                    children=[*(first.children if first else []), *parts],
                ))
                self.emitted = True
                continue
            if token_type == 'block_code':
                info = str((token.get('attrs') or {}).get('info') or '').strip().split()
                language = info[0] if info else ''
                provider_payload = {}
                if language:
                    provider_payload['code_language'] = language
                if len(info) > 1:
                    provider_payload['code_meta'] = ' '.join(info[1:])
                blocks.append(self.block(
                    'code', str(token.get('raw') or '').rstrip('\n'),
                    provider_payload=provider_payload,
                    **({'language': language} if language else {}),
                ))
                self.emitted = True
                continue
            if token_type == 'table':
                blocks.append(self.block('table', self.table_markdown(token), editable=False))
                self.emitted = True
                continue
            if token_type == 'thematic_break':
                blocks.append(self.block('divider', '---'))
                self.emitted = True
                continue
            if token_type in {'block_html', 'html_block'}:
                blocks.append(self.block('html', str(token.get('raw') or ''), editable=False))
                self.emitted = True
                continue
            raw = str(token.get('raw') or '')
            if not raw and token.get('children'):
                raw = _inline_content(token.get('children') or []).content
            if raw.strip():
                blocks.append(self.block(token_type or 'raw_markdown', raw, editable=False))
                self.emitted = True
        return blocks

    @staticmethod
    def heading_tree(blocks: Iterable[WriterBlock]) -> List[WriterBlock]:
        roots: List[WriterBlock] = []
        stack: List[tuple[int, WriterBlock]] = []
        for block in blocks:
            if block.type == 'heading':
                level = max(1, min(6, int(block.numbering.get('level') or 2)))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                if stack:
                    stack[-1][1].children.append(block)
                else:
                    roots.append(block)
                stack.append((level, block))
            elif stack:
                stack[-1][1].children.append(block)
            else:
                roots.append(block)
        return roots

    def parse(self) -> WriterDocument:
        blocks = self.heading_tree(self.parse_sequence(self.parser(self.markdown)))
        document = WriterDocument(
            document_id=self.document_id,
            stage='final',
            title=self.title,
            blocks=blocks,
            ui_editable=False,
            metadata={
                'source': 'lazyllm-markdown-conversion',
                'markdown_source': self.markdown,
            },
        )
        document.metadata['markdown_signature'] = _document_signature(document)
        return document


def writer_document_from_markdown(
    markdown: str,
    document_id: str = 'writer-document',
) -> WriterDocument:
    return _MarkdownParser(markdown, document_id).parse()


def writer_document_from_lmd(content: str) -> WriterDocument:
    raw = json.loads(content)
    if isinstance(raw, dict) and 'data' in raw:
        return deserialize_artifact_json(content, WriterDocument)
    return WriterDocument.model_validate(raw)


def _escape_markdown_text(value: str) -> str:
    return re.sub(r'([\\`*_\[\]{}#+.!|>\-])', r'\\\1', value)


def _inline_code(value: str) -> str:
    longest = max([len(item) for item in re.findall(r'`+', value)] or [0])
    fence = '`' * (longest + 1)
    padding = ' ' if re.search(r'(^`|`$|^\s|\s$)', value) else ''
    return f'{fence}{padding}{value}{padding}{fence}'


def _span_style(span: WriterSpan) -> Dict[str, Any]:
    if isinstance(span.style, dict):
        return span.style
    extra = span.model_extra or {}
    value = extra.get('stype')
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(item): True for item in value}
    return {}


def _render_styled_text(value: str, span: Optional[WriterSpan]) -> str:
    result = _escape_markdown_text(value)
    if span is None:
        return result
    style = _span_style(span)
    if style.get('inline_code') or style.get('code'):
        result = _inline_code(value)
    if style.get('strong') or style.get('bold'):
        result = f'**{result}**'
    if style.get('italic'):
        result = f'*{result}*'
    if style.get('strike') or style.get('strikethrough'):
        result = f'~~{result}~~'
    return result


def _span_at(block: WriterBlock, position: int) -> Optional[WriterSpan]:
    offset = 0
    for span in block.spans:
        end = offset + len(span.text)
        if offset <= position < end:
            return span
        offset = end
    return None


def _render_styled_range(block: WriterBlock, start: int, end: int) -> str:
    content = block.content
    if ''.join(span.text for span in block.spans) != content:
        return _escape_markdown_text(content[start:end])
    boundaries = {start, end}
    offset = 0
    for span in block.spans:
        offset += len(span.text)
        if start < offset < end:
            boundaries.add(offset)
    positions = sorted(boundaries)
    return ''.join(
        _render_styled_text(content[position:positions[index + 1]], _span_at(block, position))
        for index, position in enumerate(positions[:-1])
    )


def _render_inline(block: WriterBlock) -> str:
    content = block.content
    media = next((item for item in block.references if item.get('type') == 'media_asset'), None)
    if media:
        url = str(media.get('url') or media.get('path') or '').replace(' ', '%20')
        if url:
            title_value = str(media.get('title') or '')
            title = f' "{title_value.replace(chr(34), chr(92) + chr(34))}"' if title_value else ''
            alt = content or str(media.get('alt') or '')
            return f'![{_escape_markdown_text(alt)}]({url}{title})'

    references = list(block.references)
    if ''.join(span.text for span in block.spans) == content:
        offset = 0
        for span in block.spans:
            style = _span_style(span)
            link = style.get('link')
            if isinstance(link, dict):
                target = str(link.get('target_node_id') or '')
                url = ''
                if link.get('type') == 'internal_ref' and target:
                    url = f'#block-{target.removeprefix("block-")}'
                elif isinstance(link.get('url'), str):
                    url = link['url']
                if url:
                    reference = {
                        'type': 'link', 'url': url,
                        'start': offset, 'end': offset + len(span.text),
                    }
                    if reference not in references:
                        references.append(reference)
            offset += len(span.text)

    references = sorted(
        (
            item for item in references
            if item.get('type') in {'link', 'markdown_image'}
            and isinstance(item.get('start'), (int, float))
            and isinstance(item.get('end'), (int, float))
        ),
        key=lambda item: int(item['start']),
    )
    cursor = 0
    parts: List[str] = []
    for reference in references:
        start = max(cursor, min(len(content), int(reference['start'])))
        end = max(start, min(len(content), int(reference['end'])))
        parts.append(_render_styled_range(block, cursor, start))
        url = str(reference.get('url') or reference.get('href') or '').replace(' ', '%20')
        title_value = str(reference.get('title') or '')
        title = f' "{title_value.replace(chr(34), chr(92) + chr(34))}"' if title_value else ''
        if reference.get('type') == 'markdown_image':
            alt = str(reference.get('alt') or content[start:end])
            parts.append(f'![{_escape_markdown_text(alt)}]({url}{title})')
        else:
            parts.append(f'[{_render_styled_range(block, start, end)}]({url}{title})')
        cursor = end
    parts.append(_render_styled_range(block, cursor, len(content)))
    output = ''.join(parts)
    hard_breaks = sorted(
        int(item['offset']) for item in block.references
        if item.get('type') == 'hard_break' and isinstance(item.get('offset'), (int, float))
    )
    if hard_breaks and '\n' in output:
        output = output.replace('\n', '  \n', len(hard_breaks))
    return output


def _list_ordinal(block: WriterBlock, fallback: int) -> int:
    number = block.numbering.get('number')
    if isinstance(number, list) and number:
        try:
            value = int(number[-1])
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    for key in ('value', 'start'):
        try:
            value = int(block.numbering.get(key))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return fallback


def _render_list_item(block: WriterBlock, depth: int, fallback: int) -> str:
    ordered = bool(block.numbering.get('ordered'))
    marker = f'{_list_ordinal(block, fallback)}.' if ordered else str(block.numbering.get('marker') or '-')
    checkbox = ''
    if block.numbering.get('task'):
        checkbox = f'[{"x" if block.numbering.get("checked") else " "}] '
    indent = '  ' * depth
    continuation = indent + (' ' * (len(marker) + 1))
    lines = f'{checkbox}{_render_inline(block)}'.split('\n')
    result = f'{indent}{marker} {lines[0] if lines else ""}'
    if len(lines) > 1:
        result += '\n' + '\n'.join(f'{continuation}{line}' for line in lines[1:])
    children = _render_block_sequence(block.children, depth + 1, allow_raw=False)
    if children:
        result += f'\n{children}'
    return result


def _code_fence(content: str) -> str:
    longest = max([len(item) for item in re.findall(r'^\s*(`{3,})', content, re.MULTILINE)] or [2])
    return '`' * (longest + 1)


def _preserved_block_source(block: WriterBlock) -> Optional[str]:
    source = block.provider_payload.get('markdown_source')
    signature = block.provider_payload.get('markdown_signature')
    if isinstance(source, str) and signature == _block_signature(block):
        return source
    return None


def _render_block(block: WriterBlock, depth: int, allow_raw: bool) -> str:
    if allow_raw:
        raw = _preserved_block_source(block)
        if raw is not None:
            return raw
    if block.type == 'document':
        return _render_block_sequence(block.children, depth, allow_raw)
    anchor = (
        f'<a id="block-{block.node_id}"></a>'
        if block.type in {'heading', 'image', 'table', 'code'}
        else ''
    )
    if block.type == 'heading':
        level = min(max(int(block.numbering.get('level') or 1) + 1, 2), 6)
        current = f'{"#" * level} {_render_inline(block)}'
    elif block.type == 'paragraph':
        current = _render_inline(block)
    elif block.type == 'quote':
        body = '\n\n'.join(filter(None, [
            _render_inline(block),
            _render_block_sequence(block.children, depth, allow_raw=False),
        ]))
        return '\n'.join(f'> {line}' if line else '>' for line in body.split('\n'))
    elif block.type == 'code':
        if re.match(r'^\s*(```|~~~)', block.content):
            current = block.content
        else:
            extras = block.model_extra or {}
            language = str(extras.get('language') or block.provider_payload.get('code_language') or '').strip()
            meta = str(block.provider_payload.get('code_meta') or '').strip()
            info = f'{language}{(" " + meta) if meta else ""}'
            fence = _code_fence(block.content)
            current = f'{fence}{info}\n{block.content}\n{fence}'
    elif block.type == 'divider':
        current = block.content.strip() if re.fullmatch(r'(?:[-*_]\s*){3,}', block.content.strip()) else '---'
    elif block.type == 'image':
        current = _render_inline(block)
    else:
        current = block.content
    current = '\n'.join(filter(None, [anchor, current]))
    children = _render_block_sequence(block.children, depth, allow_raw)
    return '\n\n'.join(filter(None, [current, children]))


def _render_block_sequence(
    blocks: Iterable[WriterBlock],
    list_depth: int = 0,
    allow_raw: bool = True,
) -> str:
    values = list(blocks)
    parts: List[str] = []
    index = 0
    while index < len(values):
        block = values[index]
        if block.type != 'list_item':
            rendered = _render_block(block, list_depth, allow_raw)
            if rendered:
                parts.append(rendered)
            index += 1
            continue
        ordered = bool(block.numbering.get('ordered'))
        fallback = _list_ordinal(block, 1)
        group: List[str] = []
        while index < len(values) and values[index].type == 'list_item' \
                and bool(values[index].numbering.get('ordered')) == ordered:
            group.append(_render_list_item(values[index], list_depth, fallback))
            fallback += 1
            index += 1
        parts.append('\n'.join(group))
    return '\n\n'.join(parts)


def render_block_markdown(block: WriterBlock, level: int = 2) -> str:
    if block.type == 'heading':
        block = block.model_copy(update={
            'numbering': {**block.numbering, 'level': max(level - 1, 1)},
        })
    return _render_block(block, 0, allow_raw=True).strip()


def render_document_markdown(document: WriterDocument) -> str:
    source = document.metadata.get('markdown_source')
    signature = document.metadata.get('markdown_signature')
    if isinstance(source, str) and signature == _document_signature(document):
        return source
    title = f'# {_escape_markdown_text(document.title.strip())}' if document.title.strip() else ''
    body = _render_block_sequence(document.blocks)
    rendered = '\n\n'.join(filter(None, [title, body])).rstrip()
    return f'{rendered}\n' if rendered else ''


def writer_document_to_markdown(document: WriterDocument) -> str:
    from ..numbering import build_numbering_view_from_ir, compute_numbering, materialize_ir

    numbering = compute_numbering(build_numbering_view_from_ir(document))
    return render_document_markdown(materialize_ir(document, numbering))


def writer_document_to_lmd(document: WriterDocument) -> str:
    return serialize_artifact_json(document, created_by='lazyllm-writer-conversion')


def convert_writer_content(
    content: str,
    source_format: WriterSourceFormat,
    target_format: WriterTargetFormat,
    *,
    document_id: str = 'writer-document',
) -> str:
    if source_format not in {'markdown', 'lmd', 'writer_document'}:
        raise ValueError(f'Unsupported Writer source format: {source_format!r}.')
    if target_format not in {'markdown', 'lmd'}:
        raise ValueError(f'Unsupported Writer target format: {target_format!r}.')
    if source_format == target_format:
        return content
    if source_format == 'markdown':
        document = writer_document_from_markdown(content, document_id)
    elif source_format == 'lmd':
        document = writer_document_from_lmd(content)
    else:
        document = WriterDocument.model_validate(json.loads(content))
    return (
        writer_document_to_markdown(document)
        if target_format == 'markdown'
        else writer_document_to_lmd(document)
    )


__all__ = [
    'WriterSourceFormat',
    'WriterTargetFormat',
    'convert_writer_content',
    'render_block_markdown',
    'render_document_markdown',
    'writer_document_from_lmd',
    'writer_document_from_markdown',
    'writer_document_to_lmd',
    'writer_document_to_markdown',
]
