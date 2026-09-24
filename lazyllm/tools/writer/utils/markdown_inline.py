from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..data_models.writer_ir import WriterSpan


@dataclass
class MarkdownInlineContent:
    content: str = ''
    spans: List[WriterSpan] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)


def _append_span(result: MarkdownInlineContent, text: str, style: Dict[str, Any]) -> None:
    if not text:
        return
    result.content += text
    if result.spans and result.spans[-1].style == style:
        result.spans[-1].text += text
    else:
        result.spans.append(WriterSpan(text=text, style=dict(style)))


def parse_markdown_inline(  # noqa: C901
    tokens: Optional[Iterable[Dict[str, Any]]],
    style: Optional[Dict[str, Any]] = None,
    result: Optional[MarkdownInlineContent] = None,
) -> MarkdownInlineContent:
    output = result or MarkdownInlineContent()
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
            parse_markdown_inline(children, child_style, output)
            continue
        if token_type == 'inline_math':
            _append_span(output, '$' + str(token.get('raw') or '') + '$', {**inherited, 'math_source': True})
            continue
        if token_type == 'codespan':
            _append_span(output, str(token.get('raw') or ''), {**inherited, 'inline_code': True})
            continue
        if token_type == 'link':
            url = str((token.get('attrs') or {}).get('url') or '')
            if url.startswith('#block-'):
                parse_markdown_inline(children, {
                    **inherited,
                    'link': {
                        'type': 'internal_ref',
                        'target_node_id': url.removeprefix('#block-'),
                    },
                }, output)
                continue
            start = len(output.content)
            parse_markdown_inline(children, inherited, output)
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
            alt_content = parse_markdown_inline(children).content
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
            parse_markdown_inline(children, inherited, output)
            continue
        _append_span(output, str(token.get('raw') or ''), inherited)
    return output


__all__ = ['MarkdownInlineContent', 'parse_markdown_inline']
