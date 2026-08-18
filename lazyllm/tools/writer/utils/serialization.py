from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from lazyllm.thirdparty import mistune

from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterSpan, WriterStage
from ..numbering import MARKDOWN_ANCHOR_RE


class MarkdownSelectionError(ValueError):
    def __init__(self, code: str, message: str, **details: Any):  # noqa: B042
        super().__init__(message)
        self.error_code = code
        self.details = details


_NUMBERED_HEADING_RE = re.compile(
    r'^\s*(?:\d+(?:\.\d+)*(?!\s*年)(?:\s*[、.．：:]\s*|\s+)'
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


def to_prompt_json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if hasattr(obj, 'model_dump'):
            return obj.model_dump(exclude_defaults=True)
        return str(obj)

    return json.dumps(value, ensure_ascii=False, indent=2, default=default)


def locate_markdown_paragraph(markdown: str, selected_text: str) -> str:
    '''Return the unique source paragraph containing rendered selected text.'''
    selected = _normalize_markdown_text(selected_text)
    if not selected:
        raise MarkdownSelectionError(
            'SELECTION_UNSUPPORTED', 'selected_text must not be empty.',
        )
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


def parse_markdown_sections(markdown: str) -> List[tuple[int, List[str], int, str]]:
    sections: List[tuple[int, List[str], int, str]] = []
    heading_path: List[str] = []
    occurrences: dict[tuple[str, ...], int] = {}
    current: Optional[tuple[int, List[str], int, List[str]]] = None
    fence: Optional[str] = None

    for line in markdown.splitlines():
        fence_match = re.match(r'^\s*(```+|~~~+)', line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            if current is not None:
                current[3].append(line)
            continue
        if fence is not None:
            if current is not None:
                current[3].append(line)
            continue
        match = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
        if not match:
            if current is not None:
                current[3].append(line)
            continue
        if current is not None:
            sections.append((current[0], current[1], current[2], '\n'.join(current[3]).strip()))
        level = len(match.group(1))
        title = match.group(2)
        heading_path = heading_path[:level - 1]
        heading_path.append(title)
        path_key = tuple(heading_path)
        occurrence = occurrences.get(path_key, 0) + 1
        occurrences[path_key] = occurrence
        current = (level, list(heading_path), occurrence, [])

    if current is not None:
        sections.append((current[0], current[1], current[2], '\n'.join(current[3]).strip()))
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


def parse_document_markdown(  # noqa: C901
    markdown: str,
    document_id: str,
    stage: WriterStage = 'draft',
    outline: Optional[WriterDocument] = None,
    media_assets: Optional[MediaAssetLibrary] = None,
) -> WriterDocument:
    '''Convert the drafting Markdown subset into the existing WriterDocument IR.'''
    tokens = mistune.create_markdown(renderer='ast', plugins=['table'])(markdown or '')
    outline_ids: Dict[str, List[str]] = defaultdict(list)
    if outline:
        for block in outline.iter_blocks():
            if block.type == 'heading' and block.content.strip():
                outline_ids[block.content.strip()].append(block.node_id)

    used_ids = set()
    sequence = 0
    pending_anchor_ids: List[str] = []

    def next_id(kind: str, title: str = '') -> str:
        nonlocal sequence
        if kind == 'heading' and title:
            candidates = outline_ids.get(title.strip()) or []
            while candidates:
                candidate = candidates.pop(0)
                if candidate not in used_ids:
                    used_ids.add(candidate)
                    return candidate
        sequence += 1
        candidate = f'{document_id}-{kind}-{sequence}'
        while candidate in used_ids:
            sequence += 1
            candidate = f'{document_id}-{kind}-{sequence}'
        used_ids.add(candidate)
        return candidate

    def normalize_anchor_target(raw: str) -> str:
        target = raw.strip()
        if target.startswith('block-'):
            target = target[len('block-'):]
        return target

    def take_pending_node_id(kind: str, title: str = '') -> str:
        if pending_anchor_ids:
            node_id = normalize_anchor_target(pending_anchor_ids.pop(0))
            if node_id in used_ids:
                raise ValueError(f'duplicate Markdown anchor target: {node_id!r}')
            used_ids.add(node_id)
            return node_id
        return next_id(kind, title)

    title = outline.title if outline else ''
    blocks: List[WriterBlock] = []
    heading_stack: List[tuple[int, WriterBlock]] = []

    def append_block(block: WriterBlock) -> None:
        if heading_stack:
            heading_stack[-1][1].children.append(block)
        else:
            blocks.append(block)

    for token in tokens:
        token_type = token.get('type')
        if token_type == 'blank_line':
            continue
        if token_type == 'heading':
            level = int((token.get('attrs') or {}).get('level') or 1)
            content = _markdown_token_text(token).strip()
            if level == 1 and not blocks and not heading_stack:
                title = content or title
                continue
            block = WriterBlock(
                node_id=take_pending_node_id('heading', content),
                type='heading',
                content=content,
                stage=stage,
                numbering={'level': max(level - 1, 1)},
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
                    append_block(WriterBlock(
                        node_id=next_id('list-item'),
                        type='list_item',
                        content=content,
                        stage=stage,
                        numbering={'ordered': ordered},
                    ))
            continue

        if token_type == 'paragraph':
            children = token.get('children') or []
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
            raw_paragraph = _markdown_token_text(token).strip()
            anchor_matches = list(MARKDOWN_ANCHOR_RE.finditer(raw_paragraph))
            anchor_ids = [
                normalize_anchor_target(match.group(1))
                for match in anchor_matches
            ]
            without_anchors = raw_paragraph
            for match in reversed(anchor_matches):
                without_anchors = (
                    without_anchors[:match.start()] + without_anchors[match.end():]
                )
            if anchor_ids and not without_anchors.strip():
                pending_anchor_ids.extend(anchor_ids)
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

        content, block_type = _markdown_block_content(token)
        if not content.strip():
            continue
        block = WriterBlock(
            node_id=take_pending_node_id(block_type),
            type=block_type,
            content=content.strip(),
            stage=stage,
            spans=_markdown_spans_from_token(token) if block_type in {'table', 'code'} else [],
        )
        append_block(block)

    return WriterDocument(
        document_id=document_id,
        stage=stage,
        title=title,
        blocks=blocks,
        ui_editable=False,
        metadata={
            'source': 'parse_document_markdown',
            'outline_id': outline.document_id if outline else None,
        },
    )


def _markdown_token_text(token: Dict[str, Any]) -> str:
    token_type = token.get('type')
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
        if node_type in {'text', 'codespan'}:
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
        if node_type == 'inline_html':
            text = str(node.get('raw') or '')
            if text:
                spans.append(WriterSpan(text=text))
            return
        for child in node.get('children') or []:
            walk(child)

    walk(token)
    return spans


def _markdown_block_content(token: Dict[str, Any]) -> tuple[str, str]:
    token_type = token.get('type')
    if token_type == 'block_code':
        info = str((token.get('attrs') or {}).get('info') or '').strip()
        return f'```{info}\n{str(token.get("raw") or "").rstrip()}\n```', 'code'
    if token_type == 'block_quote':
        content = _markdown_token_text(token).strip()
        return '\n'.join(f'> {line}' for line in content.splitlines()), 'quote'
    if token_type == 'table':
        rows = []
        for row in token.get('children') or []:
            if row.get('type') == 'table_body':
                rows.extend(row.get('children') or [])
            else:
                rows.append(row)
        table_rows = [
            [_markdown_token_text(cell).strip() for cell in row.get('children') or []]
            for row in rows
        ]
        if not table_rows:
            return '', 'paragraph'
        lines = [f'| {" | ".join(table_rows[0])} |']
        lines.append(f'| {" | ".join("---" for _ in table_rows[0])} |')
        lines.extend(f'| {" | ".join(row)} |' for row in table_rows[1:])
        return '\n'.join(lines), 'table'
    if token_type == 'thematic_break':
        return '---', 'divider'
    return _markdown_token_text(token), 'paragraph'
