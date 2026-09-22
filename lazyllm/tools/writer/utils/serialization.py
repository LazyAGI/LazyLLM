from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from lazyllm.thirdparty import mistune

from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.writer_ir import (
    ContextRelation,
    WritingSubTask,
    WriterBlock,
    WriterDocument,
    WriterSpan,
    WriterStage,
)
from ..numbering import (
    MARKDOWN_ANCHOR_RE,
    find_markdown_images,
    parse_markdown_anchor_numbering,
    parse_markdown_heading_numbering_config,
    strip_markdown_heading_numbering_config,
)
from .markdown_inline import parse_markdown_inline
from .markdown_ids import markdown_anchor_ids, next_markdown_node_id


CALLOUT_MARKER_RE = re.compile(r'^\[!([\w-]+)\]([+-]?)(?:[ \t]+([^\n]*))?(?:\n|$)')


def slice_spans_from(spans: List[WriterSpan], start: int) -> List[WriterSpan]:
    sliced: List[WriterSpan] = []
    offset = 0
    for span in spans:
        end = offset + len(span.text)
        if end <= start:
            offset = end
            continue
        if offset < start:
            sliced.append(span.model_copy(update={'text': span.text[start - offset:]}))
        else:
            sliced.append(span)
        offset = end
    return sliced


class MarkdownSelectionError(ValueError):
    def __init__(self, code: str, message: str, **details: Any):  # noqa: B042
        super().__init__(message)
        self.error_code = code
        self.details = details


_NUMBERED_HEADING_RE = re.compile(
    r'^\s*(?:[（(](?:\d+|[a-zA-Z]+|[ivxlcdmIVXLCDM]+|[一二三四五六七八九十百]+)[）)]\s*'
    r'|[①-⑳]\s*'
    r'|\d+(?:\.\d+)*(?!\s*年)(?:\s*[、.．：:]\s*|\s+)'
    r'|第\s*(?:\d+(?:\.\d+)*|[一二三四五六七八九十百千万零〇两]+)\s*[章节部分篇]\s*[：:、.．]?\s*'
    r'|[一二三四五六七八九十百千万零〇两]+\s*[、.．：:]\s*)'
)
_NUMBERED_CAPTION_RE = re.compile(
    r'^\s*(?:图|表|代码)\s*\d+(?:\.\d+)*\s*[：:.\s]?\s*'
)
_INTERNAL_LINK_URL_RE = re.compile(r'^#(?:block-)?[A-Za-z0-9_.:-]+$')


def strip_heading_numbering(value: str) -> str:
    '''Remove visible heading numbering from generated/persisted heading text.'''
    text = (value or '').strip()
    match = _NUMBERED_HEADING_RE.match(text)
    return text[match.end():].strip() if match else text


def strip_caption_numbering(value: str) -> str:
    '''Remove visible float numbering from generated/persisted caption text.'''
    text = (value or '').strip()
    match = _NUMBERED_CAPTION_RE.match(text)
    return text[match.end():].strip() if match else text


def strip_math_delimiters(content: str) -> str:
    '''Return a math expression without common Markdown/LaTeX delimiters.'''
    value = content.strip()
    for left, right in (
        ('$$', '$$'),
        ('\\[', '\\]'),
        ('\\(', '\\)'),
        ('$', '$'),
    ):
        if value.startswith(left) and value.endswith(right):
            return value[len(left):-len(right)].strip()
    return value


def to_prompt_json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if hasattr(obj, 'model_dump'):
            return obj.model_dump(exclude_defaults=True)
        return str(obj)

    return json.dumps(value, ensure_ascii=False, indent=2, default=default)


def locate_markdown_paragraph(
    markdown: str,
    selected_text: str,
    *,
    heading_path: Optional[List[str]] = None,
    occurrence: int = 1,
) -> str:
    '''Return the unique source paragraph containing rendered selected text.'''
    selected = _normalize_markdown_text(selected_text)
    if not selected:
        raise MarkdownSelectionError(
            'SELECTION_UNSUPPORTED', 'selected_text must not be empty.',
        )
    if heading_path:
        start, end = markdown_section_range(markdown, heading_path, occurrence)
        markdown = markdown[start:end]
    parser = mistune.create_markdown(renderer='ast', plugins=['table'])
    matches: List[str] = []
    for block in _markdown_source_blocks(markdown):
        tokens = [token for token in parser(block) if token.get('type') != 'blank_line']
        visible = _normalize_markdown_text(' '.join(
            _markdown_visible_text(token) for token in tokens
        ))
        if selected in visible or selected in _normalize_markdown_text(block):
            if len(tokens) != 1 or tokens[0].get('type') != 'paragraph':
                raise MarkdownSelectionError(
                    'SELECTION_UNSUPPORTED',
                    'Only Markdown paragraphs can be rewritten.',
                )
            matches.append(block)
    if not matches:
        raise MarkdownSelectionError(
            'SELECTION_STALE',
            'The selected text no longer identifies a Markdown paragraph.',
        )
    if len(matches) > 1:
        raise MarkdownSelectionError(
            'SELECTION_AMBIGUOUS',
            f'The selected text matches {len(matches)} Markdown paragraphs.',
            match_count=len(matches),
        )
    return matches[0]


def validate_markdown_paragraph(markdown: str) -> str:
    candidate = (markdown or '').strip()
    parser = mistune.create_markdown(renderer='ast', plugins=['table'])
    tokens = [token for token in parser(candidate) if token.get('type') != 'blank_line']
    if len(tokens) != 1 or tokens[0].get('type') != 'paragraph':
        raise MarkdownSelectionError(
            'INVALID_GENERATED_BLOCK',
            'The generated Markdown must contain exactly one paragraph.',
        )
    return candidate


def markdown_image_sources(markdown: str) -> set[str]:
    sources: set[str] = set()
    pending = list(mistune.create_markdown(renderer='ast', plugins=['table'])(markdown))
    while pending:
        token = pending.pop()
        if token.get('type') == 'image':
            sources.add(str((token.get('attrs') or {}).get('url') or ''))
        elif token.get('type') in {'block_html', 'inline_html'}:
            sources.update(image.source for image in find_markdown_images(token.get('raw') or ''))
        pending.extend(token.get('children') or [])
    return sources


def _markdown_source_blocks(markdown: str) -> List[str]:
    blocks: List[str] = []
    start: Optional[int] = None
    fence: Optional[str] = None
    for match in re.finditer(r'.*(?:\n|$)', markdown):
        line = match.group(0)
        if not line and match.start() == len(markdown):
            continue
        fence_match = re.match(r'^\s*(```+|~~~+)', line)
        if fence is not None:
            if fence_match and fence_match.group(1)[0] == fence[0]:
                fence = None
            continue
        if fence_match:
            fence = fence_match.group(1)
        if line.strip():
            start = match.start() if start is None else start
            continue
        if start is not None:
            blocks.append(markdown[start:match.start()].rstrip('\r\n'))
            start = None
    if start is not None:
        blocks.append(markdown[start:].rstrip('\r\n'))
    return [block for block in blocks if block.strip()]


def _normalize_markdown_text(value: str) -> str:
    return re.sub(r'\s+', ' ', (value or '').replace('\u00a0', ' ')).strip()


def _markdown_visible_text(token: Dict[str, Any]) -> str:
    if token.get('type') in {'text', 'codespan'}:
        return str(token.get('raw') or '')
    if token.get('type') in {'softbreak', 'linebreak'}:
        return ' '
    return ''.join(_markdown_visible_text(child) for child in token.get('children') or [])


MARKDOWN_OUTLINE_INSTRUCTION_RE = re.compile(
    r'^\s*<!--\s*writer:outline\s+(\{.*\})\s*-->\s*$'
)


def parse_markdown_outline_instructions(markdown: str) -> Dict[str, Dict[str, Any]]:
    '''Read heading-owned outline instructions from invisible Markdown sidecars.'''
    result: Dict[str, Dict[str, Any]] = {}
    pending_node_id = ''
    current_heading_id = ''
    fence: Optional[str] = None
    for line in markdown.splitlines():
        fence_match = re.match(r'^\s*(```+|~~~+)', line)
        if fence_match:
            marker = fence_match.group(1)[0]
            fence = marker if fence is None else None if fence == marker else fence
            continue
        if fence is not None:
            continue
        anchor = re.search(r'<a\s+id=["\']block-([^"\']+)["\'][^>]*>\s*</a>', line)
        if anchor:
            pending_node_id = anchor.group(1)
            continue
        if re.match(r'^#{1,6}\s+', line):
            current_heading_id = pending_node_id
            pending_node_id = ''
            continue
        sidecar = MARKDOWN_OUTLINE_INSTRUCTION_RE.match(line)
        if sidecar and current_heading_id:
            try:
                payload = json.loads(sidecar.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payload['node_id'] = current_heading_id
                result[current_heading_id] = payload
            continue
        if line.strip():
            current_heading_id = ''
    return result


def _strip_markdown_outline_instructions(markdown: str) -> str:
    '''Remove outline sidecars before treating Markdown as visible prose.'''
    trailing_newline = markdown.endswith('\n')
    value = '\n'.join(
        line for line in markdown.splitlines()
        if not MARKDOWN_OUTLINE_INSTRUCTION_RE.match(line)
    )
    return f'{value}\n' if trailing_newline else value


def apply_markdown_outline_instructions(
    markdown: str,
    document: 'WriterDocument',
) -> str:
    '''Persist WriterBlock instruction fields immediately after Markdown headings.'''
    block_by_id = {
        block.node_id: block
        for block in document.iter_blocks()
        if block.type == 'heading'
    }
    source = _strip_markdown_outline_instructions(markdown)
    trailing_newline = source.endswith('\n')
    output: List[str] = []
    pending_node_id = ''
    fence: Optional[str] = None
    for line in source.splitlines():
        output.append(line)
        fence_match = re.match(r'^\s*(```+|~~~+)', line)
        if fence_match:
            marker = fence_match.group(1)[0]
            fence = marker if fence is None else None if fence == marker else fence
            continue
        if fence is not None:
            continue
        anchor = re.search(r'<a\s+id=["\']block-([^"\']+)["\'][^>]*>\s*</a>', line)
        if anchor:
            pending_node_id = anchor.group(1)
            continue
        if not re.match(r'^#{1,6}\s+', line):
            if line.strip():
                pending_node_id = ''
            continue
        block = block_by_id.get(pending_node_id)
        pending_node_id = ''
        if block is None:
            continue
        payload = {
            'node_id': block.node_id,
            **({'outline_description': block.outline_description} if block.outline_description else {}),
            'target_chars': block.target_chars,
            'context_relations': [item.model_dump() for item in block.context_relations],
            'subtasks': [item.model_dump() for item in block.subtasks],
        }
        output.append(
            '<!-- writer:outline '
            + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
            + ' -->'
        )
    value = '\n'.join(output)
    return f'{value}\n' if trailing_newline else value


def _markdown_fence_marker(line: str, fence: Optional[str]) -> Optional[str]:
    if fence is not None:
        closing = r' {0,3}' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}[ \t]*'
        return None if re.fullmatch(closing, line) else fence
    match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
    if match and (match.group(1)[0] != '`' or '`' not in match.group(2)):
        return match.group(1)
    return None


def _markdown_heading_ranges(markdown: str) -> List[tuple[int, int, int, List[str], int]]:
    headings: List[tuple[int, int, int, List[str], int]] = []
    stack: List[tuple[int, str]] = []
    occurrences: dict[tuple[str, ...], int] = {}
    fence: Optional[str] = None
    anchor_start: Optional[int] = None
    offset = 0
    for raw_line in markdown.splitlines(keepends=True):
        start = offset
        offset += len(raw_line)
        line = raw_line.rstrip('\r\n')
        previous_fence = fence
        fence = _markdown_fence_marker(line, fence)
        if previous_fence is not None or fence is not None:
            anchor_start = None
            continue
        if re.fullmatch(
            r'''[ \t]*<a\b[^>]*\bid\s*=\s*["'][^"']+["'][^>]*(?:/\s*>|>\s*</a\s*>)[ \t]*''',
            line, re.IGNORECASE,
        ):
            anchor_start = start if anchor_start is None else anchor_start
            continue
        match = re.match(r'^ {0,3}(#{1,6})(?:[ \t]+|$)(.*?)[ \t]*$', line)
        if not match:
            if line.strip():
                anchor_start = None
            continue
        level = len(match.group(1))
        title = re.sub(r'(?:^|[ \t]+)#+[ \t]*$', '', match.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = [item[1] for item in stack]
        path_key = tuple(heading_path)
        occurrence = occurrences.get(path_key, 0) + 1
        occurrences[path_key] = occurrence
        headings.append((start if anchor_start is None else anchor_start, offset, level, heading_path, occurrence))
        anchor_start = None
    return headings


def markdown_section_range(markdown: str, heading_path: List[str], occurrence: int = 1) -> tuple[int, int]:
    headings = _markdown_heading_ranges(markdown)
    for index, (start, _, level, path, current_occurrence) in enumerate(headings):
        if path != heading_path or current_occurrence != occurrence:
            continue
        end = next((item[0] for item in headings[index + 1:] if item[2] <= level), len(markdown))
        return start, end
    raise ValueError('content_ref is absent from Markdown document.')


def parse_markdown_sections(markdown: str) -> List[tuple[int, List[str], int, str]]:
    sections: List[tuple[int, List[str], int, str]] = []
    headings = _markdown_heading_ranges(markdown)
    for index, (_, body_start, level, path, occurrence) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(markdown)
        body_lines: List[str] = []
        fence: Optional[str] = None
        for line in markdown[body_start:end].splitlines():
            if fence is None and MARKDOWN_OUTLINE_INSTRUCTION_RE.match(line):
                continue
            fence = _markdown_fence_marker(line, fence)
            body_lines.append(line)
        body = '\n'.join(body_lines).strip()
        sections.append((level, path, occurrence, body))
    return sections


def get_markdown_outline_targets(
    markdown: str,
) -> tuple[str, List[tuple[int, List[str], int, str]]]:
    sections = parse_markdown_sections(markdown)
    titles = [heading_path[-1] for level, heading_path, _, _ in sections if level == 1]
    if len(titles) != 1:
        raise ValueError('Markdown outline must contain exactly one H1 title.')

    title = titles[0]
    targets = [section for section in sections if section[0] == 2]
    if not targets:
        raise ValueError('Markdown outline must contain at least one H2 section.')
    if any(len(heading_path) < 2 or heading_path[0] != title
           for _, heading_path, _, _ in targets):
        raise ValueError('Markdown outline H2 sections must appear under the H1 title.')
    return title, targets


def _split_markdown_image_paragraph(token: Dict[str, Any]) -> List[Dict[str, Any]]:
    children = token.get('children') or []
    if token.get('type') != 'paragraph' or not any(child.get('type') == 'image' for child in children):
        return [token]

    parts: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []

    def emit(items: List[Dict[str, Any]]) -> None:
        part = {**token, 'children': items}
        if _markdown_token_text(part).strip():
            parts.append(part)

    for child in children:
        if child.get('type') != 'image':
            pending.append(child)
            continue
        # An inline anchor immediately before an image belongs to that image,
        # even when earlier text shares the same Markdown paragraph.
        anchor_start = len(pending)
        while anchor_start and (
            pending[anchor_start - 1].get('type') in {'inline_html', 'softbreak', 'linebreak'}
            or (pending[anchor_start - 1].get('type') == 'text'
                and not str(pending[anchor_start - 1].get('raw') or '').strip())
        ):
            anchor_start -= 1
        suffix = pending[anchor_start:]
        if not MARKDOWN_ANCHOR_RE.search(''.join(_markdown_token_text(item) for item in suffix)):
            anchor_start = len(pending)
        emit(pending[:anchor_start])
        emit([*pending[anchor_start:], child])
        pending = []
    emit(pending)
    return parts


def parse_document_markdown(  # noqa: C901
    markdown: str,
    document_id: str,
    stage: WriterStage = 'draft',
    outline: Optional[WriterDocument] = None,
    media_assets: Optional[MediaAssetLibrary] = None,
) -> WriterDocument:
    '''Convert the drafting Markdown subset into the existing WriterDocument IR.'''
    outline_instructions = parse_markdown_outline_instructions(markdown or '')
    visible_markdown = _strip_markdown_outline_instructions(markdown or '')
    heading_numbering = parse_markdown_heading_numbering_config(visible_markdown)
    tokens = mistune.create_markdown(renderer='ast', plugins=['table', 'math'])(
        strip_markdown_heading_numbering_config(visible_markdown),
    )
    outline_ids: Dict[str, List[str]] = defaultdict(list)
    if outline:
        for block in outline.iter_blocks():
            if block.type == 'heading' and block.content.strip():
                outline_ids[block.content.strip()].append(block.node_id)

    anchor_ids = markdown_anchor_ids(tokens)
    reserved_ids = anchor_ids | {block.node_id for block in outline.iter_blocks()} if outline else anchor_ids
    used_ids = set()
    sequence = 0
    pending_anchor_ids: List[tuple[str, Dict[str, Any]]] = []

    def next_id(kind: str, title: str = '') -> str:
        nonlocal sequence
        if kind == 'heading' and title:
            candidates = outline_ids.get(title.strip()) or []
            while candidates:
                candidate = candidates.pop(0)
                if candidate not in used_ids and candidate not in anchor_ids:
                    used_ids.add(candidate)
                    return candidate
        candidate, sequence = next_markdown_node_id(document_id, kind, sequence, reserved_ids, used_ids)
        return candidate

    def normalize_anchor_target(raw: str) -> str:
        target = raw.strip()
        if target.startswith('block-'):
            target = target[len('block-'):]
        return target

    def take_pending_anchor(kind: str, title: str = '') -> tuple[str, Dict[str, Any]]:
        if pending_anchor_ids:
            raw_id, numbering = pending_anchor_ids.pop(0)
            node_id = normalize_anchor_target(raw_id)
            if node_id in used_ids:
                raise ValueError(f'duplicate Markdown anchor target: {node_id!r}')
            used_ids.add(node_id)
            return node_id, numbering
        return next_id(kind, title), {}

    def take_pending_node_id(kind: str, title: str = '') -> str:
        return take_pending_anchor(kind, title)[0]

    title = outline.title if outline else ''
    blocks: List[WriterBlock] = []
    heading_stack: List[tuple[int, WriterBlock]] = []

    def append_block(block: WriterBlock) -> None:
        if heading_stack:
            heading_stack[-1][1].children.append(block)
        else:
            blocks.append(block)

    for token in (part for item in tokens for part in _split_markdown_image_paragraph(item)):
        token_type = token.get('type')
        if token_type == 'blank_line':
            continue
        if token_type == 'block_math':
            append_block(WriterBlock(
                node_id=take_pending_node_id('math'), type='math',
                content='$$\n' + token['raw'] + '\n$$', stage=stage, editable=False,
            ))
            continue
        if token_type == 'heading':
            level = int((token.get('attrs') or {}).get('level') or 1)
            content = _markdown_token_text(token).strip()
            if level == 1 and not blocks and not heading_stack:
                title = content or title
                continue
            node_id, anchor_numbering = take_pending_anchor('heading', content)
            spans = _markdown_spans_from_token(token)
            block = WriterBlock(
                node_id=node_id, type='heading',
                content=content, stage=stage,
                spans=spans if any(span.style.get('math_source') for span in spans) else [],
                numbering={
                    'level': max(level - 1, 1),
                    **anchor_numbering,
                },
            )
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if heading_stack:
                heading_stack[-1][1].children.append(block)
            else:
                blocks.append(block)
            heading_stack.append((level, block))
            continue
        if token_type == 'list':
            ordered = bool((token.get('attrs') or {}).get('ordered'))
            for item in token.get('children') or []:
                content = _markdown_token_text(item).strip()
                if content:
                    spans = _markdown_spans_from_token(item)
                    append_block(WriterBlock(
                        node_id=next_id('list-item'), type='list_item',
                        content=content, stage=stage,
                        spans=spans if any(span.style.get('math_source') for span in spans) else [],
                        numbering={'ordered': ordered},
                    ))
            continue

        if token_type == 'table':
            rows: List[WriterBlock] = []
            for section in token.get('children') or []:
                header = section.get('type') == 'table_head'
                row_tokens = (
                    [section] if header else list(section.get('children') or [])
                )
                for row_token in row_tokens:
                    cells: List[WriterBlock] = []
                    for cell_token in row_token.get('children') or []:
                        rich = parse_markdown_inline(cell_token.get('children') or [])
                        align = str((cell_token.get('attrs') or {}).get('align') or '')
                        cells.append(WriterBlock(
                            node_id=next_id('table-cell'),
                            type='table_cell',
                            content=rich.content,
                            spans=rich.spans,
                            references=rich.references,
                            stage=stage,
                            numbering={
                                'header': header,
                                **({'align': align} if align else {}),
                            },
                        ))
                    rows.append(WriterBlock(
                        node_id=next_id('table-row'), type='table_row',
                        children=cells, stage=stage,
                    ))
            append_block(WriterBlock(
                node_id=take_pending_node_id('table'), type='table',
                children=rows, stage=stage,
            ))
            continue

        if token_type == 'paragraph':
            children = token.get('children') or []
            raw_paragraph = _markdown_token_text(token).strip()
            anchor_matches = list(MARKDOWN_ANCHOR_RE.finditer(raw_paragraph))
            without_anchors = raw_paragraph
            for match in reversed(anchor_matches):
                without_anchors = (
                    without_anchors[:match.start()] + without_anchors[match.end():]
                )
            html_images = [
                image for image in find_markdown_images(without_anchors)
                if image.syntax == 'html'
            ]
            if len(html_images) == 1 and (
                without_anchors[:html_images[0].start]
                + without_anchors[html_images[0].end:]
            ).strip() == '':
                image = html_images[0]
                if anchor_matches:
                    node_id = normalize_anchor_target(anchor_matches[0].group(1))
                    if node_id in used_ids:
                        raise ValueError(f'duplicate Markdown anchor target: {node_id!r}')
                    used_ids.add(node_id)
                else:
                    node_id = take_pending_node_id('image')
                asset_id = next(
                    (
                        key for key, asset in (media_assets.assets if media_assets else {}).items()
                        if image.source in {str(asset.local_path or ''), str(asset.uri or '')}
                    ),
                    '',
                )
                append_block(WriterBlock(
                    node_id=node_id,
                    type='image',
                    content=image.caption,
                    references=([{'type': 'media_asset', 'id': asset_id}] if asset_id else []),
                    stage=stage,
                ))
                continue
            visible = [
                child for child in children
                if child.get('type') not in {'softbreak', 'linebreak', 'inline_html'}
            ]
            if len(visible) == 1 and visible[0].get('type') == 'image':
                image = visible[0]
                attrs = image.get('attrs') or {}
                url = str(attrs.get('url') or '')
                anchor_match = MARKDOWN_ANCHOR_RE.search(''.join(
                    str(child.get('raw') or '')
                    for child in children if child.get('type') == 'inline_html'
                ))
                if anchor_match:
                    node_id = normalize_anchor_target(anchor_match.group(1))
                    if node_id in used_ids:
                        raise ValueError(f'duplicate Markdown anchor target: {node_id!r}')
                    used_ids.add(node_id)
                else:
                    node_id = take_pending_node_id('image')
                asset_id = next(
                    (
                        key for key, asset in (media_assets.assets if media_assets else {}).items()
                        if url in {str(asset.local_path or ''), str(asset.uri or '')}
                    ),
                    '',
                )
                alt = _markdown_token_text(image).removeprefix('![').split('](', 1)[0]
                block = WriterBlock(
                    node_id=node_id,
                    type='image',
                    content=alt,
                    references=([{'type': 'media_asset', 'id': asset_id}] if asset_id else []),
                    stage=stage,
                )
                append_block(block)
                continue
            anchors = [
                (
                    normalize_anchor_target(match.group(1)),
                    parse_markdown_anchor_numbering(match.group(0)),
                )
                for match in anchor_matches
            ]
            without_anchors = raw_paragraph
            for match in reversed(anchor_matches):
                without_anchors = (
                    without_anchors[:match.start()] + without_anchors[match.end():]
                )
            if anchors and not without_anchors.strip():
                pending_anchor_ids.extend(anchors)
                continue

            spans = _markdown_spans_from_token(token)
            content = ''.join(span.text for span in spans).strip()
            if not content and not spans:
                continue
            block = WriterBlock(
                node_id=take_pending_node_id('paragraph'),
                type='paragraph',
                content=content,
                spans=spans,
                stage=stage,
            )
            append_block(block)
            continue

        if token_type == 'block_quote':
            callout = _parse_callout_quote(token)
            if callout is not None:
                content, spans, numbering = callout
                stripped = content.strip()
                append_block(WriterBlock(
                    node_id=take_pending_node_id('callout'),
                    type='callout',
                    content=stripped,
                    spans=spans if content == stripped else [],
                    numbering=numbering,
                    stage=stage,
                ))
                continue

        content, block_type = _markdown_block_content(token)
        if not content.strip():
            continue
        block = WriterBlock(
            node_id=take_pending_node_id(block_type),
            type=block_type,
            content=content.strip(),
            stage=stage,
            spans=_markdown_spans_from_token(token) if block_type in {'code', 'quote'} else [],
            provider_payload={'code_language': str((token.get('attrs') or {}).get('info') or '').split(' ')[0]}
            if block_type == 'code' else {},
        )
        append_block(block)

    document = WriterDocument(
        document_id=document_id,
        stage=stage,
        title=title,
        blocks=blocks,
        ui_editable=False,
        metadata={
            'source': 'parse_document_markdown',
            'outline_id': outline.document_id if outline else None,
            **({
                'heading_numbering': {
                    'ordered_style': heading_numbering['ordered_style'],
                },
            } if heading_numbering else {}),
        },
    )
    for block in document.iter_blocks():
        payload = outline_instructions.get(block.node_id)
        if block.type != 'heading' or not payload:
            continue
        description = payload.get('outline_description')
        if isinstance(description, str):
            block.outline_description = ' '.join(description.split())
        target_chars = payload.get('target_chars')
        if isinstance(target_chars, int) and not isinstance(target_chars, bool) \
                and target_chars > 0:
            block.target_chars = target_chars
        block.context_relations = [
            ContextRelation.model_validate(item)
            for item in payload.get('context_relations') or []
            if isinstance(item, dict)
        ]
        block.subtasks = [
            WritingSubTask.model_validate({**item, 'node_id': block.node_id})
            for item in payload.get('subtasks') or []
            if isinstance(item, dict)
        ]
    return document


def _markdown_token_text(token: Dict[str, Any]) -> str:
    token_type = token.get('type')
    if token_type == 'inline_math':
        return '$' + str(token.get('raw') or '') + '$'
    if token_type in {'text', 'codespan'}:
        return str(token.get('raw') or '')
    if token_type == 'image':
        attrs = token.get('attrs') or {}
        alt = ''.join(_markdown_token_text(child) for child in token.get('children') or [])
        return f'![{alt}]({attrs.get("url") or ""})'
    if token_type == 'link':
        attrs = token.get('attrs') or {}
        label = ''.join(_markdown_token_text(child) for child in token.get('children') or [])
        return f'[{label}]({attrs.get("url") or ""})'
    if token_type in {'softbreak', 'linebreak'}:
        return '\n'
    if 'raw' in token:
        return str(token.get('raw') or '')
    return ''.join(_markdown_token_text(child) for child in token.get('children') or [])


def _markdown_spans_from_token(token: Dict[str, Any]) -> List[WriterSpan]:
    spans: List[WriterSpan] = []

    def walk(node: Dict[str, Any]) -> None:
        node_type = node.get('type')
        if node_type == 'inline_math':
            spans.append(WriterSpan(text='$' + str(node.get('raw') or '') + '$',
                                    style={'math_source': True}))
            return
        if node_type in {'text', 'codespan', 'inline_html'}:
            text = str(node.get('raw') or '')
            if text:
                spans.append(WriterSpan(text=text))
            return
        if node_type in {'softbreak', 'linebreak'}:
            spans.append(WriterSpan(text='\n'))
            return
        if node_type == 'link':
            attrs = node.get('attrs') or {}
            url = str(attrs.get('url') or '')
            if _INTERNAL_LINK_URL_RE.match(url):
                target = url[1:]
                if target.startswith('block-'):
                    target = target[len('block-'):]
                label = ''.join(
                    _markdown_token_text(child) for child in node.get('children') or []
                )
                spans.append(WriterSpan(
                    text=label,
                    style={'link': {'type': 'internal_ref', 'target_node_id': target}},
                ))
                return
            label = ''.join(_markdown_token_text(child) for child in node.get('children') or [])
            spans.append(WriterSpan(
                text=label,
                style={'link': {'url': url}} if url else {},
            ))
            return
        if node_type == 'image':
            attrs = node.get('attrs') or {}
            alt = ''.join(_markdown_token_text(child) for child in node.get('children') or [])
            spans.append(WriterSpan(text=f'![{alt}]({attrs.get("url") or ""})'))
            return
        for child in node.get('children') or []:
            walk(child)

    walk(token)
    return spans


def _parse_callout_quote(
    token: Dict[str, Any],
) -> Optional[tuple[str, List[WriterSpan], Dict[str, Any]]]:
    text = _markdown_token_text(token)
    match = CALLOUT_MARKER_RE.match(text)
    if match is None:
        return None
    title_start = match.start(3) if match.group(3) is not None else match.end()
    spans = _markdown_spans_from_token(token)
    aligned = ''.join(span.text for span in spans) == text
    return (
        text[title_start:],
        slice_spans_from(spans, title_start) if aligned else [],
        {
            'callout_kind': str(match.group(1)),
            'callout_fold': str(match.group(2) or ''),
            'callout_titled': match.group(3) is not None,
        },
    )


def _markdown_block_content(token: Dict[str, Any]) -> tuple[str, str]:
    token_type = token.get('type')
    if token_type == 'block_code':
        return str(token.get('raw') or '').rstrip(), 'code'
    if token_type == 'block_quote':
        return _markdown_token_text(token).strip(), 'quote'
    if token_type == 'thematic_break':
        return '---', 'divider'
    return _markdown_token_text(token), 'paragraph'
