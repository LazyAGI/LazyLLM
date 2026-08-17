from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .data_models.writer_ir import WriterBlock, WriterDocument


NumberingKind = Literal['section', 'figure', 'table', 'code']


@dataclass(frozen=True, slots=True)
class NumberingTarget:
    id: str
    kind: NumberingKind
    level: int | None
    order: int
    caption: str | None


@dataclass(frozen=True, slots=True)
class ReferenceOccurrence:
    target_id: str


@dataclass(frozen=True, slots=True)
class NumberingView:
    targets: tuple[NumberingTarget, ...]
    references: tuple[ReferenceOccurrence, ...]


@dataclass(frozen=True, slots=True)
class NumberingEntry:
    kind: NumberingKind
    number_parts: tuple[int, ...]
    caption: str | None


NumberingMap = dict[str, NumberingEntry]

_KIND_BY_TYPE = {
    'heading': 'section',
    'image': 'figure',
    'table': 'table',
    'code': 'code',
}
_LABEL_BY_KIND = {
    'section': '',
    'figure': '图',
    'table': '表',
    'code': '代码',
}
MARKDOWN_ANCHOR_RE = re.compile(
    r'<a\s+id="((?:block-)?[^"]+)"\s*(?:/\s*>|>\s*</a\s*>)',
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r'^(#{2,6})\s+(.+?)\s*$')
_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\([^)]*\)')
_INTERNAL_LINK_RE = re.compile(r'\[([^\]]*)\]\(#(block-[^)]+)\)')
_CODE_FENCE_RE = re.compile(r'^\s*(```+|~~~+)(.*)$')


def encode_anchor_id(node_id: str) -> str:
    return f'block-{node_id}'


def decode_anchor_id(anchor: str) -> str:
    if not anchor.startswith('block-'):
        raise ValueError(f'invalid anchor id: {anchor}')
    return anchor[len('block-'):]


def _is_table_header(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines) or '|' not in lines[index]:
        return False
    separator = lines[index + 1].strip()
    return '-' in separator and bool(re.match(r'^\|?[\s:|-]+\|?$', separator))


def _markdown_semantic_items(markdown: str):
    lines = markdown.splitlines()
    fence: str | None = None
    pending_anchors: list[str] = []
    for index, line in enumerate(lines):
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
                yield index, line, 'code', fence_match.group(2).strip(), pending_anchors
                pending_anchors = []
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        pending_anchors.extend(
            match.group(1) for match in MARKDOWN_ANCHOR_RE.finditer(line)
            if match.group(1).startswith('block-')
        )
        heading = _HEADING_RE.match(line)
        if heading:
            yield index, line, 'heading', heading.group(2).strip(), pending_anchors
            pending_anchors = []
            continue
        for image in _IMAGE_RE.finditer(line):
            yield index, line, 'image', image.group(1).strip(), pending_anchors
            pending_anchors = []
        if _is_table_header(lines, index):
            yield index, line, 'table', '', pending_anchors
            pending_anchors = []

def _iter_blocks(blocks: list[WriterBlock]):
    for block in blocks:
        yield block
        yield from _iter_blocks(block.children)


def _internal_target_ids(block: WriterBlock) -> list[str]:
    target_ids: list[str] = []
    for span in block.spans:
        link = span.style.get('link')
        if isinstance(link, dict) and link.get('type') == 'internal_ref':
            target_id = link.get('target_node_id')
            if isinstance(target_id, str) and target_id:
                target_ids.append(target_id)
    return target_ids


def build_numbering_view_from_ir(document: WriterDocument) -> NumberingView:
    targets: list[NumberingTarget] = []
    references: list[ReferenceOccurrence] = []
    for order, block in enumerate(_iter_blocks(document.blocks), start=1):
        kind = _KIND_BY_TYPE.get(block.type)
        if kind is not None:
            level = block.numbering.get('level') if block.type == 'heading' else None
            targets.append(
                NumberingTarget(
                    id=block.node_id,
                    kind=kind,
                    level=level,
                    order=order,
                    caption=block.content.strip() or None,
                )
            )
        references.extend(
            ReferenceOccurrence(target_id)
            for target_id in _internal_target_ids(block)
        )
    return NumberingView(tuple(targets), tuple(references))


def build_numbering_view_from_markdown(markdown: str) -> NumberingView:
    targets: list[NumberingTarget] = []
    references: list[ReferenceOccurrence] = []
    order = 0
    for _, line, kind, caption, anchors in _markdown_semantic_items(markdown):
        order += 1
        target_id = (
            decode_anchor_id(anchors[0])
            if anchors
            else f'md-{kind}-{order}'
        )
        level = None
        if kind == 'heading':
            heading = _HEADING_RE.match(line)
            level = len(heading.group(1)) - 1 if heading else 1
        normalized_kind = 'figure' if kind == 'image' else kind
        targets.append(NumberingTarget(
            id=target_id,
            kind='section' if kind == 'heading' else normalized_kind,
            level=level,
            order=order,
            caption=caption or None,
        ))
        references.extend(
            ReferenceOccurrence(decode_anchor_id(url))
            for _, url in _INTERNAL_LINK_RE.findall(line)
        )
    return NumberingView(tuple(targets), tuple(references))


def validate_numbering_view(view: NumberingView) -> None:
    target_ids = {target.id for target in view.targets}
    if len(target_ids) != len(view.targets):
        raise ValueError('duplicate target ids')
    for reference in view.references:
        if reference.target_id not in target_ids:
            raise ValueError(f'unresolved reference: {reference.target_id}')
    for target in view.targets:
        if target.kind == 'section' and (
            not isinstance(target.level, int)
            or isinstance(target.level, bool)
            or target.level < 1
        ):
            raise ValueError(f'invalid heading level for {target.id}')


def compute_numbering(view: NumberingView) -> NumberingMap:
    validate_numbering_view(view)
    numbering: NumberingMap = {}
    counters: list[int] = []
    float_counters = {'figure': 0, 'table': 0, 'code': 0}

    for target in view.targets:
        if target.kind == 'section':
            level = target.level or 1
            if level <= len(counters):
                counters = counters[:level]
            counters.extend([0] * (level - len(counters)))
            counters[-1] += 1
            number_parts = tuple(counters)
        else:
            float_counters[target.kind] += 1
            number_parts = (float_counters[target.kind],)
        numbering[target.id] = NumberingEntry(
            kind=target.kind,
            number_parts=number_parts,
            caption=target.caption,
        )
    return numbering


def format_target_number(entry: NumberingEntry) -> str:
    if entry.kind == 'section':
        return f'{".".join(str(part) for part in entry.number_parts)}.'
    return f'{_LABEL_BY_KIND[entry.kind]}{entry.number_parts[0]}'


def format_reference(entry: NumberingEntry) -> str:
    if entry.kind == 'section':
        return f'第{".".join(str(part) for part in entry.number_parts)}章'
    return format_target_number(entry)


def materialize_ir(document: WriterDocument, numbering: NumberingMap) -> WriterDocument:
    result = document.model_copy(deep=True)
    for block in _iter_blocks(result.blocks):
        entry = numbering.get(block.node_id)
        if entry is not None and block.type in {'heading', 'image', 'table', 'code'}:
            prefix = format_target_number(entry)
            block.content = f'{prefix} {block.content}'.strip()
            if block.spans and block.spans[0].text:
                block.spans[0].text = f'{prefix} {block.spans[0].text}'.strip()
        for span in block.spans:
            link = span.style.get('link')
            if (
                isinstance(link, dict)
                and link.get('type') == 'internal_ref'
                and link.get('target_node_id') in numbering
            ):
                span.text = format_reference(numbering[link['target_node_id']])
        if block.spans:
            block.content = ''.join(span.text for span in block.spans)
    return result


def materialize_feishu_links(
    document: WriterDocument,
    document_token: str,
    host: str = 'feishu.cn',
) -> WriterDocument:
    numbering = compute_numbering(build_numbering_view_from_ir(document))
    block_id_by_node_id = {
        block.node_id: block.provider_binding.get('block_id')
        for block in _iter_blocks(document.blocks)
    }
    result = document.model_copy(deep=True)
    for block in _iter_blocks(result.blocks):
        for span in block.spans:
            link = span.style.get('link')
            if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                continue
            target_id = link.get('target_node_id')
            target_block_id = block_id_by_node_id.get(target_id)
            if not target_block_id:
                span.text = format_reference(numbering[target_id]) if target_id in numbering else span.text
                span.style['link'] = {
                    'url': f'https://{host}/docx/{document_token}#{target_id}',
                }
                continue
            span.style['link'] = {
                'url': f'https://{host}/docx/{document_token}#{target_block_id}',
            }
            if target_id in numbering:
                span.text = format_reference(numbering[target_id])
        if block.spans:
            block.content = ''.join(span.text for span in block.spans)
    return result


def materialize_markdown(markdown: str) -> str:
    view = build_numbering_view_from_markdown(markdown)
    numbering = compute_numbering(view)
    targets = list(view.targets)
    target_index = 0
    output: list[str] = []
    fence: str | None = None
    lines = markdown.splitlines()

    def replace_link(match: re.Match[str]) -> str:
        entry = numbering.get(decode_anchor_id(match.group(2)))
        if entry is None:
            return match.group(0)
        return f'[{format_reference(entry)}](#{match.group(2)})'

    for index, line in enumerate(lines):
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
                target = targets[target_index] if target_index < len(targets) else None
                if target is not None and target.kind == 'code':
                    label = format_target_number(numbering[target.id])
                    output.append(f'{label} {target.caption or ""}'.strip())
                    target_index += 1
            elif fence == marker:
                fence = None
            output.append(line)
            continue
        if fence is not None:
            output.append(line)
            continue
        if _is_table_header(lines, index):
            target = targets[target_index] if target_index < len(targets) else None
            if target is not None and target.kind == 'table':
                label = format_target_number(numbering[target.id])
                output.append(f'{label} {target.caption or ""}'.strip())
                target_index += 1
        heading = _HEADING_RE.match(line)
        if heading:
            target = targets[target_index] if target_index < len(targets) else None
            if target is not None and target.kind == 'section':
                line = f'{heading.group(1)} {format_target_number(numbering[target.id])} {heading.group(2).strip()}'
                target_index += 1
        else:
            images = list(_IMAGE_RE.finditer(line))
            if images:
                pieces: list[str] = []
                last = 0
                for image in images:
                    target = targets[target_index] if target_index < len(targets) else None
                    if target is not None and target.kind == 'figure':
                        caption = image.group(1).strip()
                        label = format_target_number(numbering[target.id])
                        replacement = image.group(0).replace(
                            f'![{image.group(1)}]',
                            f'![{label} {caption}]',
                            1,
                        )
                        target_index += 1
                    else:
                        replacement = image.group(0)
                    pieces.append(line[last:image.start()])
                    pieces.append(replacement)
                    last = image.end()
                pieces.append(line[last:])
                line = ''.join(pieces)
        line = _INTERNAL_LINK_RE.sub(replace_link, line)
        output.append(line)
    return '\n'.join(output)


def dematerialize_ir(
    document: WriterDocument,
    base_numbering: NumberingMap,
) -> WriterDocument:
    result = document.model_copy(deep=True)
    for block in _iter_blocks(result.blocks):
        entry = base_numbering.get(block.node_id)
        if entry is not None and _KIND_BY_TYPE.get(block.type) == entry.kind:
            prefix = f'{format_target_number(entry)} '
            if block.content.startswith(prefix):
                block.content = block.content[len(prefix):]
            if block.spans and block.spans[0].text.startswith(prefix):
                block.spans[0].text = block.spans[0].text[len(prefix):]
        for span in block.spans:
            link = span.style.get('link')
            if isinstance(link, dict) and link.get('type') == 'internal_ref':
                span.text = ''
        if block.spans:
            block.content = ''.join(span.text for span in block.spans)
    return result


def dematerialize_markdown(markdown: str, base_numbering: NumberingMap | None = None) -> str:
    semantic_items = list(_markdown_semantic_items(markdown))
    view = build_numbering_view_from_markdown(markdown)
    targets_by_line: dict[int, list[NumberingTarget]] = {}
    for (line_index, *_), target in zip(semantic_items, view.targets):
        targets_by_line.setdefault(line_index, []).append(target)

    output: list[str] = []
    fence: str | None = None
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            output.append(line)
            continue
        if fence is not None:
            output.append(line)
            continue

        next_targets = targets_by_line.get(index + 1, [])
        if len(next_targets) == 1 and next_targets[0].kind in {'table', 'code'}:
            entry = (base_numbering or {}).get(next_targets[0].id)
            if entry is not None and entry.kind == next_targets[0].kind:
                caption = f'{format_target_number(entry)} {entry.caption or ""}'.strip()
                if line == caption:
                    continue

        heading = _HEADING_RE.match(line)
        if heading:
            targets = targets_by_line.get(index, [])
            target = targets[0] if len(targets) == 1 else None
            entry = (base_numbering or {}).get(target.id) if target is not None else None
            if (
                target is not None and target.kind == 'section'
                and entry is not None and entry.kind == target.kind
            ):
                prefix = f'{format_target_number(entry)} '
                if heading.group(2).startswith(prefix):
                    line = f'{heading.group(1)} {heading.group(2)[len(prefix):]}'.rstrip()
        else:
            images = list(_IMAGE_RE.finditer(line))
            targets = [
                target for target in targets_by_line.get(index, [])
                if target.kind == 'figure'
            ]
            if images and len(images) == len(targets):
                pieces: list[str] = []
                last = 0
                for image, target in zip(images, targets):
                    replacement = image.group(0)
                    entry = (base_numbering or {}).get(target.id)
                    prefix = (
                        f'{format_target_number(entry)} '
                        if entry is not None and entry.kind == target.kind
                        else None
                    )
                    if prefix and image.group(1).startswith(prefix):
                        replacement = replacement.replace(
                            f'![{image.group(1)}]',
                            f'![{image.group(1)[len(prefix):]}]',
                            1,
                        )
                    pieces.extend((line[last:image.start()], replacement))
                    last = image.end()
                pieces.append(line[last:])
                line = ''.join(pieces)
        line = _INTERNAL_LINK_RE.sub(r'[](#\2)', line)
        output.append(line)
    return '\n'.join(output)


__all__ = [
    'NumberingEntry',
    'NumberingMap',
    'NumberingTarget',
    'NumberingView',
    'ReferenceOccurrence',
    'MARKDOWN_ANCHOR_RE',
    'build_numbering_view_from_ir',
    'build_numbering_view_from_markdown',
    'compute_numbering',
    'decode_anchor_id',
    'dematerialize_markdown',
    'dematerialize_ir',
    'encode_anchor_id',
    'format_reference',
    'format_target_number',
    'materialize_markdown',
    'materialize_feishu_links',
    'materialize_ir',
    'validate_numbering_view',
]
