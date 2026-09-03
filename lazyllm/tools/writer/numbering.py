from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .data_models.writer_ir import WriterBlock, WriterDocument


NumberingKind = Literal['section', 'figure', 'table', 'code']
HeadingNumberingMode = Literal['ordered', 'unordered']
OrderedHeadingNumberingStyle = Literal['hierarchical', 'chinese', 'parenthesized']


@dataclass(frozen=True, slots=True)
class NumberingTarget:
    id: str
    kind: NumberingKind
    level: int | None
    caption: str | None
    mode: HeadingNumberingMode | None = None
    restart: bool = False


@dataclass(frozen=True, slots=True)
class ReferenceOccurrence:
    target_id: str


@dataclass(frozen=True, slots=True)
class NumberingView:
    targets: tuple[NumberingTarget, ...]
    references: tuple[ReferenceOccurrence, ...]
    ordered_style: OrderedHeadingNumberingStyle = 'hierarchical'


@dataclass(frozen=True, slots=True)
class NumberingEntry:
    kind: NumberingKind
    number_parts: tuple[int, ...]
    caption: str | None
    label: str = ''
    mode: HeadingNumberingMode | None = None
    restart: bool = False


NumberingMap = dict[str, NumberingEntry]


@dataclass(frozen=True, slots=True)
class MarkdownImage:
    raw: str
    start: int
    end: int
    caption: str
    source: str
    syntax: Literal['markdown', 'html']

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
    r'<a\s+id="((?:block-)?[^"]+)"(?:\s+[^>]*)?\s*(?:/\s*>|>\s*</a\s*>)',
    re.IGNORECASE,
)
MARKDOWN_NUMBERING_ATTR_RE = re.compile(
    r'\snumbering=(?:"([^"]*)"|\'([^\']*)\')',
    re.IGNORECASE,
)
MARKDOWN_HEADING_NUMBERING_CONFIG_RE = re.compile(
    r'<!--\s*heading-numbering:\s*(\{[^\r\n]*\})\s*-->',
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r'^(#{2,6})\s+(.+?)\s*$')
_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]*)\)')
_HTML_IMAGE_RE = re.compile(r'<img\b[^>]*?/?>', re.IGNORECASE)
_HTML_IMAGE_ATTR_RE = re.compile(
    r'\b(src|alt)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'=<>`]+))',
    re.IGNORECASE,
)
_INTERNAL_LINK_RE = re.compile(r'\[([^\]]*)\]\(#(block-[^)]+)\)')
_CODE_FENCE_RE = re.compile(r'^\s*(```+|~~~+)(.*)$')

_ORDERED_STYLES = {'hierarchical', 'chinese', 'parenthesized'}


def find_markdown_images(line: str) -> tuple[MarkdownImage, ...]:
    """Return CommonMark and standalone HTML images in source order."""
    images = [
        MarkdownImage(
            raw=match.group(0), start=match.start(), end=match.end(),
            caption=match.group(1).strip(), source=match.group(2).strip(),
            syntax='markdown',
        )
        for match in _IMAGE_RE.finditer(line)
    ]
    for match in _HTML_IMAGE_RE.finditer(line):
        attributes: dict[str, str] = {}
        for attribute in _HTML_IMAGE_ATTR_RE.finditer(match.group(0)):
            attributes[attribute.group(1).lower()] = (
                attribute.group(2) or attribute.group(3) or attribute.group(4) or ''
            )
        source = attributes.get('src', '').strip()
        if source:
            images.append(MarkdownImage(
                raw=match.group(0), start=match.start(), end=match.end(),
                caption=attributes.get('alt', '').strip(), source=source,
                syntax='html',
            ))
    return tuple(sorted(images, key=lambda image: image.start))


def parse_markdown_heading_numbering_config(markdown: str) -> dict[str, Any]:
    match = MARKDOWN_HEADING_NUMBERING_CONFIG_RE.search(markdown)
    if match is None:
        return {}
    try:
        config = json.loads(match.group(1))
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid Markdown heading numbering config') from exc
    if not isinstance(config, dict):
        raise ValueError('Markdown heading numbering config must be an object')
    style = config.get('ordered_style', 'hierarchical')
    if set(config) - {'ordered_style'} or style not in _ORDERED_STYLES:
        raise ValueError('invalid Markdown heading numbering config')
    return {'ordered_style': style}


def strip_markdown_heading_numbering_config(markdown: str) -> str:
    return MARKDOWN_HEADING_NUMBERING_CONFIG_RE.sub('', markdown)


def format_markdown_heading_numbering_config(
    config: Mapping[str, Any],
) -> str:
    payload: dict[str, Any] = {}
    style = config.get('ordered_style', 'hierarchical')
    if style != 'hierarchical':
        payload['ordered_style'] = style
    if not payload:
        return ''
    value = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    return f'<!-- heading-numbering: {value} -->'


def parse_markdown_anchor_numbering(anchor: str) -> dict[str, Any]:
    match = MARKDOWN_NUMBERING_ATTR_RE.search(anchor)
    if match is None:
        return {}
    value = match.group(1) or match.group(2) or ''
    if value == 'mode=unordered':
        return {'mode': 'unordered'}
    if value == 'restart':
        return {'restart': True}
    raise ValueError(f'invalid Markdown heading numbering attribute: {value!r}')


def format_markdown_anchor_numbering(numbering: Mapping[str, Any]) -> str:
    if numbering.get('style') is not None:
        raise ValueError('heading numbering style must be global')
    if numbering.get('mode') == 'unordered':
        return ' numbering="mode=unordered"'
    if numbering.get('restart'):
        return ' numbering="restart"'
    return ''


def encode_anchor_id(node_id: str) -> str:
    return f'block-{node_id}'


def decode_anchor_id(anchor: str) -> str:
    if not anchor.startswith('block-'):
        raise ValueError(f'invalid anchor id: {anchor}')
    return anchor[len('block-'):]


def ensure_markdown_heading_anchors(markdown: str) -> str:
    '''Give every numberable Markdown heading a stable hierarchical id.'''
    has_trailing_newline = markdown.endswith('\n')
    used_anchors = {
        match.group(1)
        for match in MARKDOWN_ANCHOR_RE.finditer(markdown)
        if match.group(1).startswith('block-')
    }
    output: list[str] = []
    pending_anchors: list[str] = []
    counters: list[int] = []
    fence: str | None = None
    for line in markdown.splitlines():
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

        anchors = [
            match.group(1)
            for match in MARKDOWN_ANCHOR_RE.finditer(line)
            if match.group(1).startswith('block-')
        ]
        if anchors:
            pending_anchors.extend(anchors)
            output.append(line)
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            depth = len(heading.group(1)) - 1
            counters = counters[:depth]
            counters.extend([0] * (depth - len(counters)))
            counters[-1] += 1
            expected = 'block-sec-' + '-'.join(
                f'{value:03d}' for value in counters
            )
            if not pending_anchors:
                while expected in used_anchors:
                    counters[-1] += 1
                    expected = 'block-sec-' + '-'.join(
                        f'{value:03d}' for value in counters
                    )
                output.append(f'<a id="{expected}"></a>')
                used_anchors.add(expected)
            pending_anchors = []
        elif line.strip() and pending_anchors:
            pending_anchors = []
        output.append(line)
    result = '\n'.join(output)
    return f'{result}\n' if has_trailing_newline else result


def _is_table_header(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines) or '|' not in lines[index]:
        return False
    separator = lines[index + 1].strip()
    return '-' in separator and bool(re.match(r'^\|?[\s:|-]+\|?$', separator))


def _markdown_semantic_items(markdown: str):
    lines = markdown.splitlines()
    fence: str | None = None
    pending_anchors: list[tuple[str, dict[str, Any]]] = []
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
            (match.group(1), parse_markdown_anchor_numbering(match.group(0)))
            for match in MARKDOWN_ANCHOR_RE.finditer(line)
            if match.group(1).startswith('block-')
        )
        heading = _HEADING_RE.match(line)
        if heading:
            yield index, line, 'heading', heading.group(2).strip(), pending_anchors
            pending_anchors = []
            continue
        images = find_markdown_images(line)
        for image in images:
            yield index, line, 'image', image.caption, pending_anchors
            pending_anchors = []
        if _is_table_header(lines, index):
            yield index, line, 'table', '', pending_anchors
            pending_anchors = []
        elif not images and MARKDOWN_ANCHOR_RE.sub('', line).strip():
            # A target anchor applies only to the next semantic target. Do not
            # leak it across ordinary or unsupported content into a later heading.
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
    config = document.metadata.get('heading_numbering')
    configured_style = config.get('ordered_style') if isinstance(config, dict) else None
    ordered_style = configured_style or 'hierarchical'
    for block in _iter_blocks(document.blocks):
        kind = _KIND_BY_TYPE.get(block.type)
        if kind is not None:
            numbering = block.numbering if block.type == 'heading' else {}
            level = numbering.get('level') if block.type == 'heading' else None
            mode = numbering.get('mode')
            if block.type == 'heading' and numbering.get('style') is not None:
                raise ValueError('heading numbering style must be global')
            targets.append(
                NumberingTarget(
                    id=block.node_id,
                    kind=kind,
                    level=level,
                    caption=block.content.strip() or None,
                    mode=mode,
                    restart=bool(numbering.get('restart')),
                )
            )
        references.extend(
            ReferenceOccurrence(target_id)
            for target_id in _internal_target_ids(block)
        )
    return NumberingView(tuple(targets), tuple(references), ordered_style)


def build_numbering_view_from_markdown(markdown: str) -> NumberingView:
    targets: list[NumberingTarget] = []
    references: list[ReferenceOccurrence] = []
    items = list(_markdown_semantic_items(markdown))
    config = parse_markdown_heading_numbering_config(markdown)
    ordered_style = config.get('ordered_style', 'hierarchical')
    for order, (_, line, kind, caption, anchors) in enumerate(items, start=1):
        anchor_id, anchor_numbering = anchors[0] if anchors else ('', {})
        target_id = decode_anchor_id(anchor_id) if anchor_id else f'md-{kind}-{order}'
        level = None
        if kind == 'heading':
            heading = _HEADING_RE.match(line)
            level = len(heading.group(1)) - 1 if heading else 1
        normalized_kind = 'figure' if kind == 'image' else kind
        mode = anchor_numbering.get('mode') if kind == 'heading' else None
        targets.append(NumberingTarget(
            id=target_id,
            kind='section' if kind == 'heading' else normalized_kind,
            level=level,
            caption=caption or None,
            mode=mode,
            restart=bool(anchor_numbering.get('restart')) if kind == 'heading' else False,
        ))
        references.extend(
            ReferenceOccurrence(decode_anchor_id(url))
            for _, url in _INTERNAL_LINK_RE.findall(line)
        )
    return NumberingView(tuple(targets), tuple(references), ordered_style)


def validate_numbering_view(view: NumberingView) -> None:
    if view.ordered_style not in _ORDERED_STYLES:
        raise ValueError(f'invalid global ordered heading style: {view.ordered_style!r}')
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
        if target.mode is not None and target.mode not in {'ordered', 'unordered'}:
            raise ValueError(f'invalid heading numbering mode for {target.id}')
        if target.restart and target.mode == 'unordered':
            raise ValueError(f'unordered heading cannot restart numbering: {target.id}')


def _chinese_number(value: int) -> str:
    if value < 1:
        return str(value)
    digits = '零一二三四五六七八九'
    units = ['', '十', '百', '千']
    parts: list[str] = []
    for position, digit in enumerate(reversed(str(value))):
        if digit != '0':
            parts.append(digits[int(digit)] + units[position])
        elif parts and not parts[-1].startswith('零'):
            parts.append('零')
    result = ''.join(reversed(parts))
    if 11 <= value <= 19:
        result = result.removeprefix('一')
    return result or '零'


def _roman_number(value: int, upper: bool) -> str:
    pairs = (
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
    )
    result = ''
    remaining = value
    for limit, letter in pairs:
        while remaining >= limit:
            result += letter
            remaining -= limit
    return result if upper else result.lower()


def _letter_number(value: int, upper: bool) -> str:
    result = ''
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord('A' if upper else 'a') + remainder) + result
    return result or 'a'


def _hierarchical_parts(parts: tuple[int, ...]) -> str:
    return f'{".".join(str(part) for part in parts)}.'


def _format_heading_label(
    style: OrderedHeadingNumberingStyle,
    parts: tuple[int, ...],
    depth: int,
) -> str:
    value = parts[-1] if parts else 1
    if style == 'hierarchical':
        return _hierarchical_parts(parts)
    if style == 'chinese':
        if depth == 1:
            return f'{_chinese_number(value)}、'
        if depth == 2:
            return f'（{_chinese_number(value)}）'
        if depth == 3:
            return f'{value}.'
        if depth == 4:
            return f'（{value}）'
        if depth == 5 and 1 <= value <= 20:
            return f'{chr(0x245F + value)}'
    elif style == 'parenthesized':
        if depth == 1:
            return f'({value})'
        if depth == 2:
            return f'({_letter_number(value, False)})'
        if depth == 3:
            return f'({_roman_number(value, False)})'
        if depth == 4:
            return f'({_letter_number(value, True)})'
        if depth == 5:
            return f'({_roman_number(value, True)})'
    return _hierarchical_parts(parts)


def compute_numbering(view: NumberingView) -> NumberingMap:
    validate_numbering_view(view)
    numbering: NumberingMap = {}
    counters: list[int] = []
    root_level: int | None = None
    previous_level = 0
    float_counters = {'figure': 0, 'table': 0, 'code': 0}

    for target in view.targets:
        if target.kind == 'section':
            raw_level = max(1, int(target.level or 1))
            if root_level is None:
                root_level = raw_level
            level = max(1, raw_level - root_level + 1)
            # A document cannot start at a nested level or skip a hierarchy.
            level = min(level, previous_level + 1)
            mode = target.mode or 'ordered'
            style = None if mode == 'unordered' else view.ordered_style

            if level <= len(counters):
                counters = counters[:level]
            if mode == 'ordered':
                if target.restart:
                    counters = counters[:level - 1] + [0]
                    counters.extend([0] * (level - len(counters)))
                else:
                    counters.extend([0] * (level - len(counters)))
                counters[-1] += 1
                number_parts = tuple(counters)
                previous_level = level
            else:
                number_parts = ()
        else:
            float_counters[target.kind] += 1
            number_parts = (float_counters[target.kind],)
            mode = None
            style = None
        numbering[target.id] = NumberingEntry(
            kind=target.kind,
            number_parts=number_parts,
            caption=target.caption,
            label=(
                _format_heading_label(style, number_parts, level)
                if target.kind == 'section' and style is not None else ''
            ),
            mode=mode,
            restart=target.restart,
        )
    return numbering


def format_target_number(entry: NumberingEntry) -> str:
    if entry.kind == 'section':
        if entry.mode == 'unordered':
            return ''
        if entry.label:
            return entry.label
        return f'{".".join(str(part) for part in entry.number_parts)}.'
    return f'{_LABEL_BY_KIND[entry.kind]}{entry.number_parts[0]}'


def materialize_ir(document: WriterDocument, numbering: NumberingMap) -> WriterDocument:
    result = document.model_copy(deep=True)
    for block in _iter_blocks(result.blocks):
        entry = numbering.get(block.node_id)
        if entry is not None and block.type in {'heading', 'image', 'table', 'code'}:
            prefix = format_target_number(entry)
            if block.type == 'code':
                block.provider_payload['numbering_caption'] = prefix
                continue
            if prefix:
                block.content = f'{prefix} {block.content}'
                if block.spans and block.spans[0].text:
                    block.spans[0].text = f'{prefix} {block.spans[0].text}'
    return result


def _markdown_targets_by_line(
    markdown: str,
    view: NumberingView,
) -> dict[int, list[NumberingTarget]]:
    items = list(_markdown_semantic_items(markdown))
    if len(items) != len(view.targets):
        raise ValueError('Markdown numbering view does not match document')
    targets_by_line: dict[int, list[NumberingTarget]] = {}
    for (line_index, *_), target in zip(items, view.targets):
        targets_by_line.setdefault(line_index, []).append(target)
    return targets_by_line


def materialize_markdown(
    markdown: str,
    view: NumberingView,
    numbering: NumberingMap,
) -> str:  # noqa: C901
    targets_by_line = _markdown_targets_by_line(markdown, view)
    output: list[str] = []
    fence: str | None = None
    lines = markdown.splitlines()

    for index, line in enumerate(lines):
        fence_match = _CODE_FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
                target = next((
                    item for item in targets_by_line.get(index, [])
                    if item.kind == 'code'
                ), None)
                if target is not None:
                    label = format_target_number(numbering[target.id])
                    output.append(label)
            elif fence == marker:
                fence = None
            output.append(line)
            continue
        if fence is not None:
            output.append(line)
            continue
        if _is_table_header(lines, index):
            target = next((
                item for item in targets_by_line.get(index, [])
                if item.kind == 'table'
            ), None)
            if target is not None:
                label = format_target_number(numbering[target.id])
                output.append(f'{label} {target.caption or ""}'.strip())
        heading = _HEADING_RE.match(line)
        if heading:
            target = next((
                item for item in targets_by_line.get(index, [])
                if item.kind == 'section'
            ), None)
            if target is not None:
                prefix = format_target_number(numbering[target.id])
                visible_prefix = f'{prefix} ' if prefix else ''
                line = f'{heading.group(1)} {visible_prefix}{heading.group(2)}'
        else:
            images = find_markdown_images(line)
            targets = [
                target for target in targets_by_line.get(index, [])
                if target.kind == 'figure'
            ]
            if images and len(images) == len(targets):
                pieces: list[str] = []
                last = 0
                for image, target in zip(images, targets):
                    caption = image.caption
                    label = format_target_number(numbering[target.id])
                    replacement = image.raw
                    if image.syntax == 'markdown':
                        replacement = replacement.replace(
                            f'![{caption}]', f'![{label} {caption}]', 1,
                        )
                    pieces.append(line[last:image.start])
                    pieces.append(replacement)
                    last = image.end
                pieces.append(line[last:])
                line = ''.join(pieces)
        output.append(line)
    return '\n'.join(output)


def dematerialize_ir(
    document: WriterDocument,
    base_numbering: NumberingMap,
) -> WriterDocument:
    result = document.model_copy(deep=True)
    for block in _iter_blocks(result.blocks):
        block.provider_payload.pop('numbering_caption', None)
        entry = base_numbering.get(block.node_id)
        if entry is not None and _KIND_BY_TYPE.get(block.type) == entry.kind:
            number = format_target_number(entry)
            prefix = f'{number} ' if number else ''
            if prefix and block.content.startswith(prefix):
                block.content = block.content[len(prefix):]
            if block.spans:
                first = block.spans[0].text
                if prefix and first.startswith(prefix):
                    block.spans[0].text = first[len(prefix):]
        if block.spans:
            block.content = ''.join(span.text for span in block.spans)
    return result


def dematerialize_markdown(markdown: str, base_numbering: NumberingMap | None = None) -> str:  # noqa: C901
    view = build_numbering_view_from_markdown(markdown)
    submitted_numbering = compute_numbering(view)
    targets_by_line = _markdown_targets_by_line(markdown, view)

    def entry_for(target: NumberingTarget | None) -> NumberingEntry | None:
        if target is None:
            return None
        return (
            (base_numbering or {}).get(target.id)
            or submitted_numbering.get(target.id)
        )

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
            entry = entry_for(next_targets[0])
            if entry is not None and entry.kind == next_targets[0].kind:
                caption = (
                    format_target_number(entry)
                    if next_targets[0].kind == 'code'
                    else f'{format_target_number(entry)} {entry.caption or ""}'.strip()
                )
                if line == caption:
                    continue

        heading = _HEADING_RE.match(line)
        if heading:
            targets = targets_by_line.get(index, [])
            target = targets[0] if len(targets) == 1 else None
            entry = entry_for(target)
            if (
                target is not None and target.kind == 'section'
                and entry is not None and entry.kind == target.kind
            ):
                number = format_target_number(entry)
                prefix = f'{number} ' if number else ''
                title = heading.group(2)
                if prefix and title.startswith(prefix):
                    title = title[len(prefix):]
                line = f'{heading.group(1)} {title}'.rstrip()
        else:
            images = find_markdown_images(line)
            targets = [
                target for target in targets_by_line.get(index, [])
                if target.kind == 'figure'
            ]
            if images and len(images) == len(targets):
                pieces: list[str] = []
                last = 0
                for image, target in zip(images, targets):
                    replacement = image.raw
                    entry = entry_for(target)
                    prefix = (
                        f'{format_target_number(entry)} '
                        if entry is not None and entry.kind == target.kind
                        else None
                    )
                    if prefix and image.syntax == 'markdown' and image.caption.startswith(prefix):
                        replacement = replacement.replace(
                            f'![{image.caption}]',
                            f'![{image.caption[len(prefix):]}]',
                            1,
                        )
                    pieces.extend((line[last:image.start], replacement))
                    last = image.end
                pieces.append(line[last:])
                line = ''.join(pieces)
        output.append(line)
    return '\n'.join(output)


def _validate_numbering_update(update: Mapping[str, Any]) -> str:
    update_type = update.get('type')
    if update_type == 'ordered_style':
        if update.get('ordered_style') not in _ORDERED_STYLES:
            raise ValueError('invalid global ordered heading style')
        return update_type
    if update_type == 'heading':
        if not isinstance(update.get('target_id'), str) or not update['target_id']:
            raise ValueError('heading numbering target_id is required')
        if 'mode' in update and update['mode'] not in {'ordered', 'unordered'}:
            raise ValueError('invalid heading numbering mode')
        if 'restart' in update and not isinstance(update['restart'], bool):
            raise ValueError('heading numbering restart must be a boolean')
        if not {'mode', 'restart'} & update.keys():
            raise ValueError('heading numbering update is empty')
        if update.get('mode') == 'unordered' and update.get('restart'):
            raise ValueError('unordered heading cannot restart numbering')
        return update_type
    raise ValueError('invalid numbering update type')


def _updated_heading_numbering(
    current: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    numbering = dict(current)
    mode = update.get('mode', numbering.get('mode', 'ordered'))
    numbering.pop('mode', None)
    numbering.pop('style', None)
    if mode == 'unordered':
        numbering['mode'] = 'unordered'
        numbering.pop('restart', None)
    elif update.get('restart', numbering.get('restart', False)):
        numbering['restart'] = True
    else:
        numbering.pop('restart', None)
    return numbering


def apply_numbering_update_ir(
    document: WriterDocument,
    update: Mapping[str, Any],
) -> WriterDocument:
    update_type = _validate_numbering_update(update)
    result = document.model_copy(deep=True)
    if update_type == 'ordered_style':
        config = result.metadata.get('heading_numbering')
        result.metadata['heading_numbering'] = {
            **(config if isinstance(config, dict) else {}),
            'ordered_style': update['ordered_style'],
        }
        return result

    block = result.block_by_id(update['target_id'])
    if block is None or block.type != 'heading' or not block.editable:
        raise ValueError('heading numbering target is not editable')
    block.numbering = _updated_heading_numbering(block.numbering, update)
    return result


def _with_markdown_heading_numbering_config(
    markdown: str,
    config: Mapping[str, Any],
) -> str:
    lines = [
        line for line in markdown.splitlines()
        if MARKDOWN_HEADING_NUMBERING_CONFIG_RE.fullmatch(line.strip()) is None
    ]
    comment = format_markdown_heading_numbering_config(config)
    if not comment:
        return '\n'.join(lines)
    insert_at = 0
    if lines and lines[0].strip() == '---':
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() in {'---', '...'}:
                insert_at = index + 1
                break
    lines.insert(insert_at, comment)
    return '\n'.join(lines)


def apply_numbering_update_markdown(
    markdown: str,
    update: Mapping[str, Any],
) -> str:
    update_type = _validate_numbering_update(update)
    config = parse_markdown_heading_numbering_config(markdown)
    if update_type == 'ordered_style':
        config['ordered_style'] = update['ordered_style']
        return _with_markdown_heading_numbering_config(markdown, config)

    view = build_numbering_view_from_markdown(markdown)
    if not any(
        target.id == update['target_id'] and target.kind == 'section'
        for target in view.targets
    ):
        raise ValueError('heading numbering target not found')
    anchor_id = encode_anchor_id(update['target_id'])
    match = next((
        item for item in MARKDOWN_ANCHOR_RE.finditer(markdown)
        if item.group(1) == anchor_id
    ), None)
    if match is None:
        raise ValueError('heading numbering anchor not found')
    numbering = _updated_heading_numbering(
        parse_markdown_anchor_numbering(match.group(0)),
        update,
    )
    anchor = f'<a id="{anchor_id}"{format_markdown_anchor_numbering(numbering)}></a>'
    return f'{markdown[:match.start()]}{anchor}{markdown[match.end():]}'


__all__ = [
    'HeadingNumberingMode',
    'OrderedHeadingNumberingStyle',
    'NumberingEntry',
    'NumberingMap',
    'NumberingTarget',
    'NumberingView',
    'ReferenceOccurrence',
    'MARKDOWN_ANCHOR_RE',
    'apply_numbering_update_ir',
    'apply_numbering_update_markdown',
    'build_numbering_view_from_ir',
    'build_numbering_view_from_markdown',
    'compute_numbering',
    'decode_anchor_id',
    'dematerialize_markdown',
    'dematerialize_ir',
    'encode_anchor_id',
    'ensure_markdown_heading_anchors',
    'format_markdown_anchor_numbering',
    'format_target_number',
    'find_markdown_images',
    'format_markdown_heading_numbering_config',
    'materialize_markdown',
    'materialize_ir',
    'parse_markdown_anchor_numbering',
    'parse_markdown_heading_numbering_config',
    'strip_markdown_heading_numbering_config',
    'validate_numbering_view',
]
