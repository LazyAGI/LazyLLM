from typing import Any

from ..numbering import MARKDOWN_ANCHOR_RE


def markdown_anchor_ids(tokens: list[dict[str, Any]]) -> set[str]:
    '''Reserve real HTML anchors, excluding code and escaped Markdown examples.'''
    html = ''.join(
        str(token.get('raw') or '')
        if token.get('type') in {'inline_html', 'html_inline', 'block_html', 'html_block'} else '\0'
        for token in tokens
    )
    ids = {match.group(1).strip().removeprefix('block-') for match in MARKDOWN_ANCHOR_RE.finditer(html)}
    for token in tokens:
        if token.get('children'):
            ids.update(markdown_anchor_ids(token['children']))
    return ids


def next_markdown_node_id(
    document_id: str, kind: str, sequence: int,
    reserved_ids: set[str], used_ids: set[str],
) -> tuple[str, int]:
    '''Allocate without taking an existing anchor's identity, regardless of source order.'''
    while True:
        sequence += 1
        candidate = f'{document_id}-{kind}-{sequence}'
        if candidate not in reserved_ids and candidate not in used_ids:
            used_ids.add(candidate)
            return candidate, sequence
