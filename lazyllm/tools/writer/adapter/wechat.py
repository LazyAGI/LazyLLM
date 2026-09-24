from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from html import escape, unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from ..data_models.revision import PatchHunk
from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterSpan, WriterStage
from ..numbering import (
    NumberingEntry,
    build_numbering_view_from_ir,
    compute_numbering,
    format_target_number,
)
from ..templates.wechat import WeChatTemplate, get_wechat_template
from ..utils import strip_heading_numbering, validate_writer_tables
from .base import NativeBlock, NativePatchOperation, WriterAdapterBase

_VOID_TAGS = {
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
    'meta', 'param', 'source', 'track', 'wbr',
}
_KNOWN_INLINE_TAGS = {'a', 'b', 'br', 'code', 'del', 'em', 'i', 's', 'span', 'strong', 'sub', 'sup', 'u'}
_KNOWN_BLOCK_TAGS = {'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'img', 'ol', 'p', 'pre', 'table', 'ul'}

_WECHAT_SAFE_CSS_PROPERTIES = frozenset({
    'background-color', 'border', 'border-bottom', 'border-collapse', 'border-left', 'border-right',
    'border-top', 'border-radius', 'color', 'display', 'font-family', 'font-size',
    'font-style', 'font-weight', 'letter-spacing', 'line-height', 'margin',
    'margin-bottom', 'margin-left', 'margin-right', 'margin-top', 'max-height',
    'max-width', 'min-height', 'min-width', 'object-fit', 'opacity', 'padding',
    'padding-bottom', 'padding-left', 'padding-right', 'padding-top', 'text-align',
    'text-decoration', 'text-indent', 'vertical-align', 'white-space', 'width',
    'height', 'word-break',
})

_WECHAT_TABLE_STYLE = {
    'background-color': '#ffffff', 'border-collapse': 'collapse', 'color': '#111111',
    'font-size': '14px', 'line-height': '1.5', 'margin': '12px 0', 'width': '100%',
}
_WECHAT_TABLE_HEADER_STYLE = {
    'background-color': '#ffffff', 'border': '1px solid #222222',
    'font-weight': '700', 'padding': '6px 8px', 'vertical-align': 'top',
}
_WECHAT_TABLE_CELL_STYLE = {
    'background-color': '#ffffff', 'border': '1px solid #222222',
    'padding': '6px 8px', 'vertical-align': 'top', 'word-break': 'break-word',
}

_WECHAT_SAFE_IMAGE_ATTRIBUTES = frozenset({
    'alt', 'align', 'class', 'height', 'id', 'src', 'style', 'title', 'width',
})
_WECHAT_SAFE_WRAPPER_ATTRIBUTES = frozenset({'class', 'id', 'style', 'title'})


class _HtmlNode:
    def __init__(
        self,
        tag: str,
        start: int,
        start_end: int,
        *,
        attrs: dict[str, str] | None = None,
        text: str = '',
    ) -> None:
        self.tag = tag
        self.start = start
        self.start_end = start_end
        self.end = start_end
        self.attrs = attrs or {}
        self.text = text
        self.children: list[_HtmlNode] = []


class _WeChatHTMLParser(HTMLParser):
    '''Parse HTML while retaining source offsets for lossless block fragments.'''

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self._line_starts = [0]
        self._line_starts.extend(index + 1 for index, value in enumerate(source) if value == '\n')
        self.root = _HtmlNode('#root', 0, 0)
        self.stack = [self.root]

    def _offset(self) -> int:
        line, column = self.getpos()
        return self._line_starts[line - 1] + column

    def _tag_end(self, start: int) -> int:
        end = self.source.find('>', start)
        return len(self.source) if end < 0 else end + 1

    def _append(self, node: _HtmlNode) -> None:
        self.stack[-1].children.append(node)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self._offset()
        start_end = self._tag_end(start)
        node = _HtmlNode(
            tag.lower(), start, start_end,
            attrs={str(key).lower(): '' if value is None else str(value) for key, value in attrs},
        )
        self._append(node)
        if node.tag not in _VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self._offset()
        start_end = self._tag_end(start)
        self._append(_HtmlNode(
            tag.lower(), start, start_end,
            attrs={str(key).lower(): '' if value is None else str(value) for key, value in attrs},
        ))

    def handle_endtag(self, tag: str) -> None:
        start = self._offset()
        end = self._tag_end(start)
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                for node in self.stack[index:]:
                    node.end = start
                self.stack = self.stack[:index]
                self.stack[-1].children[-1].end = end
                return

    def handle_data(self, data: str) -> None:
        start = self._offset()
        self._append(_HtmlNode('#text', start, start + len(data), text=data))

    def handle_entityref(self, name: str) -> None:
        start = self._offset()
        raw = self.source[start:self._tag_end(start)]
        raw = raw[:raw.find(';') + 1] if ';' in raw else f'&{name};'
        self._append(_HtmlNode('#text', start, start + len(raw), text=raw))

    def handle_charref(self, name: str) -> None:
        self.handle_entityref(f'#{name}')

    def handle_comment(self, data: str) -> None:
        start = self._offset()
        self._append(_HtmlNode('#raw', start, self._tag_end(start), text=self.source[start:self._tag_end(start)]))

    def handle_decl(self, decl: str) -> None:
        start = self._offset()
        self._append(_HtmlNode('#raw', start, self._tag_end(start), text=self.source[start:self._tag_end(start)]))

    def close(self) -> None:
        super().close()
        for node in self.stack[1:]:
            node.end = len(self.source)


def _html_raw(source: str, node: _HtmlNode) -> str:
    return source[node.start:node.end]


def _sanitize_css(value: Any) -> str:
    '''Keep a conservative, deterministic subset of source CSS declarations.'''
    css = str(value or '').strip()
    if not css:
        return ''
    declarations: list[str] = []
    for declaration in css.split(';'):
        if ':' not in declaration:
            continue
        key, raw_value = declaration.split(':', 1)
        key = key.strip().lower()
        raw_value = raw_value.strip()
        if not key or not raw_value or key not in _WECHAT_SAFE_CSS_PROPERTIES:
            continue
        lowered = raw_value.lower()
        if any(token in lowered for token in ('expression(', 'javascript:', 'vbscript:', 'url(')):
            continue
        declarations.append(f'{key}:{raw_value}')
    return ';'.join(declarations)


def _style_text(style: Mapping[str, Any] | str | None) -> str:
    if isinstance(style, str):
        return _sanitize_css(style)
    if not isinstance(style, Mapping):
        return ''
    declarations = [
        f'{str(key).strip().lower()}:{str(value).strip()}'
        for key, value in style.items()
        if str(key).strip().lower() in _WECHAT_SAFE_CSS_PROPERTIES
        and str(value).strip()
    ]
    return _sanitize_css(';'.join(declarations))


def _serialize_attrs(
    attrs: Mapping[str, Any] | None,
    *,
    allowed: frozenset[str],
    overrides: Mapping[str, Any] | None = None,
) -> str:
    '''Serialize a small safe attribute subset for regenerated provider HTML.'''
    values = {
        str(key).lower(): str(value or '').strip()
        for key, value in (attrs.items() if isinstance(attrs, Mapping) else [])
        if (
            (str(key).lower() in allowed or str(key).lower().startswith('data-'))
            and str(value or '').strip()
        )
    }
    if overrides:
        values.update({
            str(key).lower(): str(value or '').strip()
            for key, value in overrides.items()
            if str(key).lower() in allowed and str(value or '').strip()
        })
    if 'style' in values:
        values['style'] = _sanitize_css(values['style'])
        if not values['style']:
            values.pop('style')
    return ''.join(
        f' {key}="{escape(value, quote=True)}"'
        for key, value in values.items()
    )


def _element_children(node: _HtmlNode) -> list[_HtmlNode]:
    return [child for child in node.children if child.tag != '#text' or child.text.strip()]


def _node_text(node: _HtmlNode) -> str:
    if node.tag == '#text':
        return unescape(node.text)
    if node.tag == 'br':
        return '\n'
    return ''.join(_node_text(child) for child in node.children)


def _descendants(node: _HtmlNode, tag: str) -> Iterable[_HtmlNode]:
    for child in node.children:
        if child.tag == tag:
            yield child
        yield from _descendants(child, tag)


def _inline_style(node: _HtmlNode) -> dict[str, Any]:  # noqa: C901
    style: dict[str, Any] = {}
    if node.tag in {'b', 'strong'}:
        style['bold'] = True
    if node.tag in {'i', 'em'}:
        style['italic'] = True
    if node.tag in {'del', 's'}:
        style['strikethrough'] = True
    if node.tag == 'code':
        style['inline_code'] = True
    if node.tag == 'u':
        style['underline'] = True
    if node.tag == 'sub':
        style['subscript'] = True
    if node.tag == 'sup':
        style['superscript'] = True
    css = node.attrs.get('style', '')
    if css:
        style['wechat_css'] = css
        for declaration in css.split(';'):
            if ':' not in declaration:
                continue
            key, value = (item.strip().lower() for item in declaration.split(':', 1))
            if key == 'font-weight' and value in {'bold', '700', '800', '900'}:
                style['bold'] = True
            elif key == 'font-style' and value == 'italic':
                style['italic'] = True
            elif key == 'text-decoration' and 'line-through' in value:
                style['strikethrough'] = True
    if node.tag == 'a' and node.attrs.get('href'):
        style['link'] = {'url': node.attrs['href']}
    return style


def _inline_content(
    node: _HtmlNode,
    inherited: dict[str, Any] | None = None,
) -> tuple[str, list[WriterSpan], list[dict[str, Any]]]:
    content = ''
    spans: list[WriterSpan] = []
    references: list[dict[str, Any]] = []
    current_style = dict(inherited or {})
    current_style.update(_inline_style(node))
    if node.tag == '#text':
        text = unescape(node.text)
        if text:
            if spans and spans[-1].style == current_style:
                spans[-1].text += text
            else:
                spans.append(WriterSpan(text=text, style=current_style))
            return text, spans, references
        return '', spans, references
    if node.tag == 'br':
        return '\n', [WriterSpan(text='\n', style=current_style)], [{'type': 'hard_break', 'offset': 0}]
    for child in node.children:
        child_content, child_spans, child_refs = _inline_content(child, current_style)
        start = len(content)
        content += child_content
        for span in child_spans:
            if spans and spans[-1].style == span.style:
                spans[-1].text += span.text
            else:
                spans.append(span)
        for reference in child_refs:
            adjusted = dict(reference)
            if 'start' in adjusted:
                adjusted['start'] = int(adjusted['start']) + start
                adjusted['end'] = int(adjusted['end']) + start
            elif 'offset' in adjusted:
                adjusted['offset'] = int(adjusted['offset']) + start
            references.append(adjusted)
    if node.tag == 'a' and node.attrs.get('href') and content:
        references.append({'type': 'link', 'url': node.attrs['href'], 'start': 0, 'end': len(content)})
    return content, spans, references


def _source_snapshot(block: WriterBlock) -> dict[str, Any]:
    return {
        'type': block.type,
        'content': block.content,
        'spans': [span.model_dump(exclude_defaults=True) for span in block.spans],
        'numbering': deepcopy(block.numbering),
        'references': deepcopy(block.references),
        'children': [_source_snapshot(child) for child in block.children],
    }


def _document_snapshot(document: WriterDocument) -> list[dict[str, Any]]:
    return [_source_snapshot(block) for block in document.blocks]


def _heading_style_map(blocks: Iterable[WriterBlock]) -> dict[str, str]:
    styles: dict[str, str] = {}
    for root in blocks:
        for block in root.iter_blocks():
            if block.type != 'heading':
                continue
            style = _style_text(block.provider_payload.get('wechat_heading_style'))
            if not style:
                continue
            level = block.provider_payload.get('wechat_heading_level')
            if not isinstance(level, int) or isinstance(level, bool):
                level = int(block.numbering.get('level') or 1)
            styles.setdefault(str(max(1, min(5, level))), style)
    return styles


def _caption_style(blocks: Iterable[WriterBlock]) -> str:
    for root in blocks:
        for block in root.iter_blocks():
            if block.type != 'image':
                continue
            caption = block.provider_payload.get('wechat_image_caption')
            if (
                not isinstance(caption, Mapping)
                or not isinstance(caption.get('attrs'), Mapping)
            ):
                continue
            style = _style_text(caption['attrs'].get('style'))
            if style:
                return style
    return ''


def _raw_if_unchanged(block: WriterBlock) -> str | None:
    raw = block.provider_payload.get('raw_html')
    snapshot = block.provider_payload.get('source_snapshot')
    if isinstance(raw, str) and snapshot == _source_snapshot(block):
        return raw
    return None


def _replace_raw_heading_number(
    raw_html: str,
    source_label: str,
    target_label: str,
) -> str | None:
    if not source_label:
        return None
    parser = _WeChatHTMLParser(raw_html)
    parser.feed(raw_html)
    parser.close()
    for node in _descendants(parser.root, '#text'):
        decoded = unescape(node.text)
        if not decoded.lstrip().startswith(source_label):
            continue
        offset = node.text.find(source_label)
        if offset < 0 or unescape(node.text[:offset]).strip():
            return None
        start = node.start + offset
        end = start + len(source_label)
        return f'{raw_html[:start]}{escape(target_label)}{raw_html[end:]}'
    return None


class WeChatWriterAdapter(WriterAdapterBase):
    '''Render Writer IR into the conservative HTML subset accepted by MP drafts.'''

    provider = 'wechat'

    @staticmethod
    def can_reuse_raw(block: WriterBlock) -> bool:
        '''Return whether a block still matches its original WeChat HTML.'''
        return _raw_if_unchanged(block) is not None

    def blocks_to_ir(
        self,
        blocks: list[NativeBlock],
        *,
        external_document_id: str,
        stage: WriterStage = 'final',
        title: str = '',
        uri: str | None = None,
        revision: str | None = None,
    ) -> WriterDocument:
        if isinstance(blocks, str):
            html = blocks
        elif isinstance(blocks, list):
            html = ''.join(
                str(block.get('raw_html') or block.get('html') or '')
                for block in blocks if isinstance(block, dict)
            )
        else:
            raise TypeError(f'blocks must be a list or HTML string, got {type(blocks).__name__}.')
        return self.html_to_ir(
            html,
            external_document_id=external_document_id,
            stage=stage,
            title=title,
            uri=uri,
            revision=revision,
        )

    def html_to_ir(
        self,
        html: str,
        *,
        external_document_id: str,
        stage: WriterStage = 'final',
        title: str = '',
        uri: str | None = None,
        revision: str | None = None,
    ) -> WriterDocument:
        '''Parse WeChat article HTML into Writer IR.'''
        source = str(html or '')
        parser = _WeChatHTMLParser(source)
        parser.feed(source)
        parser.close()
        blocks = self._parse_root_blocks(
            parser.root.children,
            source,
            external_document_id,
            stage,
            path=(),
        )
        binding: dict[str, Any] = {
            'provider': self.provider,
            'document_id': external_document_id,
        }
        if uri is not None:
            binding['uri'] = uri
        if revision is not None:
            binding['revision'] = revision
        document = WriterDocument(
            document_id=self.make_document_id(external_document_id),
            stage=stage,
            title=title,
            blocks=blocks,
            revision=revision,
            metadata={
                'source_block_count': len(blocks),
                'wechat_html_source': source,
                'wechat_heading_styles': _heading_style_map(blocks),
                'wechat_caption_style': _caption_style(blocks),
            },
            provider_binding=binding,
            ui_editable=False,
        )
        validate_writer_tables(document)
        document.metadata['wechat_html_snapshot'] = _document_snapshot(document)
        return document

    def _parse_root_blocks(
        self,
        nodes: Iterable[_HtmlNode],
        source: str,
        external_document_id: str,
        stage: WriterStage,
        *,
        path: tuple[int, ...],
    ) -> list[WriterBlock]:
        blocks: list[WriterBlock] = []
        for index, node in enumerate(nodes):
            if node.tag == '#text' and not node.text.strip():
                continue
            blocks.append(self._parse_block(
                node, source, external_document_id, stage,
                path=(*path, index),
            ))
        return blocks

    def _parse_block(  # noqa: C901
        self,
        node: _HtmlNode,
        source: str,
        external_document_id: str,
        stage: WriterStage,
        *,
        path: tuple[int, ...],
    ) -> WriterBlock:
        node_id = self.make_node_id(external_document_id, 'html-' + '-'.join(map(str, path)))
        raw_html = _html_raw(source, node)
        semantic, payload_node = self._semantic_node(node)
        source_number_label: str | None = None
        source_heading_style = ''
        source_heading_level: int | None = None
        source_image_attrs: dict[str, str] = {}
        source_image_wrapper: dict[str, Any] = {}
        source_image_caption: dict[str, Any] = {}
        if semantic == 'p':
            children = _element_children(payload_node)
            if len(children) == 1 and children[0].tag == 'img':
                semantic, payload_node = 'img', children[0]
        if semantic in {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'}:
            content, spans, references = _inline_content(payload_node)
            numbering: dict[str, Any] = {}
            block_type = 'quote' if semantic == 'blockquote' else 'heading' if semantic.startswith('h') else 'paragraph'
            if block_type == 'heading':
                source_content = content.strip()
                content = strip_heading_numbering(content)
                if source_content != content:
                    prefix_end = len(source_content) - len(content) if content else len(source_content)
                    source_number_label = source_content[:prefix_end].strip()
                else:
                    source_number_label = ''
                if spans:
                    spans[0].text = strip_heading_numbering(spans[0].text)
                source_heading_level = max(1, min(5, int(semantic[1:]) - 1))
                numbering['level'] = source_heading_level
                source_heading_style = _sanitize_css(payload_node.attrs.get('style', ''))
            block = WriterBlock(
                node_id=node_id,
                type=block_type,
                content=content,
                spans=spans,
                references=references,
                numbering=numbering,
                stage=stage,
            )
        elif semantic == 'img':
            image_url = (
                payload_node.attrs.get('src')
                or payload_node.attrs.get('data-src')
                or payload_node.attrs.get('data-original')
                or ''
            )
            caption = ''
            if node.tag in {'section', 'div', 'article', 'figure'}:
                children = _element_children(node)
                if len(children) == 2 and children[0].tag == 'img':
                    caption = _node_text(children[1])
                    source_image_caption = {
                        'tag': children[1].tag,
                        'attrs': dict(children[1].attrs),
                    }
            content = caption
            block = WriterBlock(
                node_id=node_id,
                type='image',
                content=content,
                references=[{
                    'type': 'wechat_image',
                    'url': image_url,
                    'alt': payload_node.attrs.get('alt', ''),
                }],
                stage=stage,
            )
            source_image_attrs = dict(payload_node.attrs)
            if node.tag in {'section', 'div', 'article', 'figure'}:
                source_image_wrapper = {
                    'tag': node.tag,
                    'attrs': dict(node.attrs),
                }
        elif semantic == 'pre':
            code_node = next((child for child in payload_node.children if child.tag == 'code'), None)
            content = _node_text(code_node or payload_node)
            language = ''
            code_class = (code_node or payload_node).attrs.get('class', '')
            for value in code_class.split():
                if value.startswith('language-'):
                    language = value.removeprefix('language-')
                    break
            block = WriterBlock(
                node_id=node_id,
                type='code',
                content=content,
                stage=stage,
                editable=True,
                **({'language': language} if language else {}),
            )
        elif semantic == 'table':
            rows: list[WriterBlock] = []
            for row_index, row_node in enumerate(_descendants(payload_node, 'tr')):
                cells: list[WriterBlock] = []
                for cell_index, cell_node in enumerate(
                    cell for cell in row_node.children if cell.tag in {'th', 'td'}
                ):
                    content, spans, references = _inline_content(cell_node)
                    numbering: dict[str, Any] = {'header': cell_node.tag == 'th'}
                    for html_name, ir_name in (('rowspan', 'row_span'), ('colspan', 'column_span')):
                        if cell_node.attrs.get(html_name):
                            try:
                                numbering[ir_name] = int(cell_node.attrs[html_name])
                            except ValueError as exc:
                                raise ValueError(
                                    f'WeChat table cell has invalid {html_name}.') from exc
                    cells.append(WriterBlock(
                        node_id=self.make_node_id(
                            external_document_id,
                            f'html-{"-".join(map(str, path))}-row-{row_index}-cell-{cell_index}',
                        ),
                        type='table_cell', content=content, spans=spans,
                        references=references, stage=stage,
                        numbering=numbering,
                    ))
                if cells:
                    rows.append(WriterBlock(
                        node_id=self.make_node_id(
                            external_document_id,
                            f'html-{"-".join(map(str, path))}-row-{row_index}',
                        ),
                        type='table_row', children=cells, stage=stage,
                    ))
            block = WriterBlock(
                node_id=node_id,
                type='table',
                children=rows,
                stage=stage,
                editable=True,
            )
        elif semantic in {'ul', 'ol'}:
            children = self._parse_list_items(
                payload_node, source, external_document_id, stage, path=path,
            )
            block = WriterBlock(
                node_id=node_id,
                type='wechat_list',
                children=children,
                numbering={'ordered': semantic == 'ol'},
                stage=stage,
            )
        else:
            block = WriterBlock(
                node_id=node_id,
                type='wechat_opaque',
                content=_node_text(node),
                stage=stage,
                editable=False,
            )
        block.provider_payload = {
            'raw_html': raw_html,
            'source_snapshot': _source_snapshot(block),
            'source_path': list(path),
        }
        if source_number_label is not None:
            block.provider_payload['source_number_label'] = source_number_label
        if source_heading_style:
            block.provider_payload['wechat_heading_style'] = source_heading_style
        if source_heading_level is not None:
            block.provider_payload['wechat_heading_level'] = source_heading_level
        if source_image_attrs:
            block.provider_payload['wechat_image_attrs'] = source_image_attrs
        if source_image_wrapper:
            block.provider_payload['wechat_image_wrapper'] = source_image_wrapper
        if source_image_caption:
            block.provider_payload['wechat_image_caption'] = source_image_caption
        return block

    def _parse_list_items(
        self,
        node: _HtmlNode,
        source: str,
        external_document_id: str,
        stage: WriterStage,
        *,
        path: tuple[int, ...],
    ) -> list[WriterBlock]:
        items: list[WriterBlock] = []
        for index, child in enumerate(_element_children(node)):
            if child.tag != 'li':
                continue
            content_nodes = [item for item in child.children if item.tag not in {'ul', 'ol'}]
            content, spans, references = self._inline_nodes(content_nodes)
            nested = [item for item in child.children if item.tag in {'ul', 'ol'}]
            nested_blocks = [self._parse_block(
                item, source, external_document_id, stage,
                path=(*path, index, nested_index),
            ) for nested_index, item in enumerate(nested)]
            block = WriterBlock(
                node_id=self.make_node_id(external_document_id, 'html-' + '-'.join(map(str, (*path, index)))),
                type='list_item',
                content=content,
                spans=spans,
                references=references,
                numbering={'ordered': node.tag == 'ol'},
                children=nested_blocks,
                stage=stage,
            )
            block.provider_payload = {
                'raw_html': _html_raw(source, child),
                'source_snapshot': _source_snapshot(block),
                'source_path': [*path, index],
            }
            items.append(block)
        return items

    @staticmethod
    def _inline_nodes(nodes: Iterable[_HtmlNode]) -> tuple[str, list[WriterSpan], list[dict[str, Any]]]:
        wrapper = _HtmlNode('#inline', 0, 0)
        wrapper.children = list(nodes)
        return _inline_content(wrapper)

    @staticmethod
    def _semantic_node(node: _HtmlNode) -> tuple[str, _HtmlNode]:
        if node.tag in _KNOWN_BLOCK_TAGS:
            return node.tag, node
        if node.tag in {'section', 'div', 'article', 'figure'}:
            children = _element_children(node)
            if len(children) == 1 and children[0].tag in _KNOWN_BLOCK_TAGS:
                return children[0].tag, children[0]
            if len(children) == 2 and children[0].tag == 'img' and children[1].tag in {'p', 'figcaption'}:
                return 'img', children[0]
        if node.tag in _KNOWN_INLINE_TAGS:
            return 'p', node
        return '', node

    def ir_to_blocks(
        self,
        document: WriterDocument,
        media_assets: Any = None,
    ) -> list[NativeBlock]:
        raise NotImplementedError('WeChat drafts use HTML instead of native blocks.')

    def patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
        media_assets: Any = None,
    ) -> NativePatchOperation:
        raise NotImplementedError('WeChat drafts apply changes through full-document replacement.')

    def document_to_html(
        self,
        document: WriterDocument,
        image_urls: dict[str, str] | None = None,
        template: str | None = None,
    ) -> str:
        '''Render Writer IR as WeChat draft HTML using a named template.'''
        validate_writer_tables(document)
        selected_template = get_wechat_template(template)
        preserve_source = not str(template or '').strip()
        source = document.metadata.get('wechat_html_source')
        source_snapshot = document.metadata.get('wechat_html_snapshot')
        if (
            preserve_source
            and isinstance(source, str)
            and source_snapshot == _document_snapshot(document)
        ):
            return source
        images = image_urls or {}
        numbering = compute_numbering(build_numbering_view_from_ir(document))
        metadata_styles = document.metadata.get('wechat_heading_styles')
        if not isinstance(metadata_styles, Mapping):
            metadata_styles = {}
        heading_styles: dict[str, str] = {}
        if preserve_source:
            for key, value in metadata_styles.items():
                style = _style_text(value)
                if style:
                    heading_styles[str(key)] = style
            heading_styles.update(_heading_style_map(document.blocks))
        caption_style = ''
        if preserve_source:
            caption_style = _style_text(document.metadata.get('wechat_caption_style'))
        if not caption_style and preserve_source:
            caption_style = _caption_style(document.blocks)
        if not caption_style:
            caption_style = _style_text(selected_template.caption_style())
        body = ''.join(self._render_sequence(
            document.blocks, images, numbering, heading_styles, caption_style,
            selected_template, preserve_source,
        ))
        if preserve_source:
            return body
        style = _style_text(selected_template.paragraph_style())
        style_attr = f' style="{escape(style, quote=True)}"' if style else ''
        return f'<section{style_attr}>{body}</section>'

    def _render_sequence(
        self,
        blocks: list[WriterBlock],
        images: dict[str, str],
        numbering: dict[str, NumberingEntry],
        heading_styles: Mapping[str, str],
        caption_style: str,
        template: WeChatTemplate,
        preserve_source: bool,
    ) -> list[str]:
        rendered: list[str] = []
        index = 0
        while index < len(blocks):
            block = blocks[index]
            if block.type == 'list_item':
                ordered = bool(block.numbering.get('ordered'))
                items: list[WriterBlock] = []
                while index < len(blocks):
                    item = blocks[index]
                    if item.type != 'list_item' or bool(item.numbering.get('ordered')) != ordered:
                        break
                    items.append(item)
                    index += 1
                rendered.append(self._render_list_items(
                    items, ordered, images, numbering, heading_styles, caption_style,
                    template, preserve_source,
                ))
                continue
            rendered.append(self._render_block(
                block, images, numbering, heading_styles, caption_style, template,
                preserve_source,
            ))
            index += 1
        return rendered

    def _render_block(  # noqa: C901
        self,
        block: WriterBlock,
        images: dict[str, str],
        numbering: dict[str, NumberingEntry],
        heading_styles: Mapping[str, str],
        caption_style: str,
        template: WeChatTemplate,
        preserve_source: bool,
    ) -> str:
        entry = numbering.get(block.node_id)
        label = format_target_number(entry) if entry is not None else ''
        raw = _raw_if_unchanged(block)
        if raw is not None and (preserve_source or block.type == 'wechat_opaque'):
            if block.type != 'heading':
                return raw
            source_label = block.provider_payload.get('source_number_label')
            if source_label == label:
                return raw
            if isinstance(source_label, str):
                renumbered = _replace_raw_heading_number(raw, source_label, label)
                if renumbered is not None:
                    return renumbered
        if block.type == 'wechat_opaque':
            raise ValueError(
                f'Unsupported WeChat HTML block {block.node_id!r} cannot be modified.')
        if block.type == 'wechat_list':
            return self._render_list(
                block, images, numbering, heading_styles, caption_style, template,
                preserve_source,
            )
        body = self._render_inline(block)
        children = '' if block.type == 'table' else ''.join(self._render_sequence(
            block.children, images, numbering, heading_styles, caption_style, template,
            preserve_source,
        ))
        if block.type == 'heading':
            level = int(block.numbering.get('level') or 1)
            heading = min(max(level + 1, 2), 4)
            title = f'{escape(label)} {body}'.strip()
            style = ''
            if preserve_source:
                style = _style_text(block.provider_payload.get('wechat_heading_style'))
            source_level = block.provider_payload.get('wechat_heading_level')
            if (
                style
                and isinstance(source_level, int)
                and not isinstance(source_level, bool)
                and source_level != level
            ):
                style = ''
            if not style:
                style = heading_styles.get(str(max(1, min(5, level))), '')
            if not style:
                style = _style_text(template.heading_style(level))
            style_attr = f' style="{escape(style, quote=True)}"' if style else ''
            return f'<h{heading}{style_attr}>{title}</h{heading}>{children}'
        if block.type == 'image':
            asset_id = next((
                str(ref.get('id')) for ref in block.references
                if ref.get('type') == 'media_asset' and ref.get('id')
            ), '')
            url = images.get(asset_id, '') or next((
                str(ref.get('url') or '').strip() for ref in block.references
                if ref.get('type') == 'wechat_image' and ref.get('url')
            ), '')
            if not url:
                raise ValueError(f'Image block {block.node_id!r} media is unavailable.')
            caption_text = block.content.strip()
            caption_config = (
                block.provider_payload.get('wechat_image_caption')
                if preserve_source else None
            )
            caption_tag = 'p'
            caption_attrs: Mapping[str, Any] = {}
            if isinstance(caption_config, Mapping):
                candidate_tag = str(caption_config.get('tag') or '').lower()
                if candidate_tag in {'p', 'figcaption'}:
                    caption_tag = candidate_tag
                if isinstance(caption_config.get('attrs'), Mapping):
                    caption_attrs = caption_config['attrs']
            if not _style_text(caption_attrs.get('style')):
                caption_attrs = {**caption_attrs, 'style': caption_style}
            caption_attrs_html = _serialize_attrs(
                caption_attrs,
                allowed=_WECHAT_SAFE_WRAPPER_ATTRIBUTES,
            )
            caption = (
                f'<{caption_tag}{caption_attrs_html}>'
                f'{escape(caption_text)}</{caption_tag}>'
                if caption_text else ''
            )
            image_attrs = block.provider_payload.get('wechat_image_attrs')
            if not preserve_source and isinstance(image_attrs, Mapping):
                image_attrs = {key: value for key, value in image_attrs.items() if key != 'style'}
            image_attrs_html = _serialize_attrs(
                image_attrs,
                allowed=_WECHAT_SAFE_IMAGE_ATTRIBUTES,
                overrides={'src': url},
            )
            image_tag = f'<img{image_attrs_html} />'
            wrapper = block.provider_payload.get('wechat_image_wrapper')
            wrapper_tag = 'section'
            wrapper_attrs: Mapping[str, Any] = {}
            if isinstance(wrapper, Mapping):
                candidate_tag = str(wrapper.get('tag') or '').lower()
                if candidate_tag in {'section', 'div', 'article', 'figure'}:
                    wrapper_tag = candidate_tag
                if isinstance(wrapper.get('attrs'), Mapping):
                    wrapper_attrs = wrapper['attrs']
            if not preserve_source and isinstance(wrapper_attrs, Mapping):
                wrapper_attrs = {key: value for key, value in wrapper_attrs.items() if key != 'style'}
            wrapper_attrs_html = _serialize_attrs(
                wrapper_attrs,
                allowed=_WECHAT_SAFE_WRAPPER_ATTRIBUTES,
            )
            return (
                f'<{wrapper_tag}{wrapper_attrs_html}>'
                f'{image_tag}{caption}</{wrapper_tag}>{children}'
            )
        if block.type == 'table':
            caption_text = f'{escape(label)} {body}'.strip() if block.content.strip() else ''
            caption_attr = f' style="{escape(caption_style, quote=True)}"' if caption_style else ''
            caption = f'<p{caption_attr}>{caption_text}</p>' if caption_text else ''
            return f'{caption}{self._render_table(block)}'
        if block.type in {'quote', 'callout'}:
            style = _style_text(template.quote_style())
            style_attr = f' style="{escape(style, quote=True)}"' if style else ''
            return f'<blockquote{style_attr}>{body}</blockquote>{children}'
        if block.type in {'code', 'code_block'}:
            language = str(getattr(block, 'language', '') or '').strip()
            code_class = f' class="language-{escape(language, quote=True)}"' if language else ''
            style = _style_text(template.code_style())
            style_attr = f' style="{escape(style, quote=True)}"' if style else ''
            return f'<pre{style_attr}><code{code_class}>{escape(block.content)}</code></pre>{children}'
        if block.type == 'divider':
            style = _style_text(template.divider_style())
            style_attr = f' style="{escape(style, quote=True)}"' if style else ''
            return f'<hr{style_attr} />{children}'
        return (f'<p>{body}</p>' if body else '') + children

    def _render_list(
        self,
        block: WriterBlock,
        images: dict[str, str],
        numbering: dict[str, NumberingEntry],
        heading_styles: Mapping[str, str],
        caption_style: str,
        template: WeChatTemplate,
        preserve_source: bool,
    ) -> str:
        return self._render_list_items(
            block.children, bool(block.numbering.get('ordered')), images, numbering,
            heading_styles, caption_style, template, preserve_source,
        )

    def _render_list_items(
        self,
        items: list[WriterBlock],
        ordered: bool,
        images: dict[str, str],
        numbering: dict[str, NumberingEntry],
        heading_styles: Mapping[str, str],
        caption_style: str,
        template: WeChatTemplate,
        preserve_source: bool,
    ) -> str:
        list_style = _style_text(template.list_style())
        item_style = _style_text(template.list_item_style())
        marker_style = _style_text(template.list_marker_style(ordered))
        list_attr = f' style="{escape(list_style, quote=True)}"' if list_style else ''
        item_attr = f' style="{escape(item_style, quote=True)}"' if item_style else ''
        marker_attr = f' style="{escape(marker_style, quote=True)}"' if marker_style else ''
        rendered: list[str] = []
        for index, item in enumerate(items, start=1):
            if item.type != 'list_item':
                raise ValueError(f'WeChat list contains invalid child {item.type!r}.')
            number = item.numbering.get('number')
            marker = escape(str(
                number[-1] if isinstance(number, list) and number else index
            )) if ordered else '&#8203;'
            children = ''.join(self._render_sequence(
                item.children, images, numbering, heading_styles, caption_style, template,
                preserve_source,
            ))
            rendered.append(
                f'<p{item_attr}><span{marker_attr}>{marker}</span>'
                f'{self._render_inline(item)}</p>{children}'
            )
        return f'<section{list_attr}>{"".join(rendered)}</section>'

    def _render_table(self, block: WriterBlock) -> str:
        table_style = _style_text(_WECHAT_TABLE_STYLE)
        header_style = _style_text(_WECHAT_TABLE_HEADER_STYLE)
        cell_style = _style_text(_WECHAT_TABLE_CELL_STYLE)
        rows = []
        for row in block.children:
            if row.type != 'table_row':
                continue
            cells = []
            for cell in row.children:
                if cell.type != 'table_cell':
                    continue
                tag = 'th' if cell.numbering.get('header') else 'td'
                style = header_style if tag == 'th' else cell_style
                spans = ''.join(
                    f' {html_name}="{cell.numbering[ir_name]}"'
                    for html_name, ir_name in (('rowspan', 'row_span'), ('colspan', 'column_span'))
                    if cell.numbering.get(ir_name, 1) != 1
                )
                cells.append(
                    f'<{tag}{spans} style="{escape(style, quote=True)}">'
                    f'{self._render_inline(cell)}</{tag}>'
                )
            rows.append(f'<tr>{"".join(cells)}</tr>')
        return f'<table style="{escape(table_style, quote=True)}">{"".join(rows)}</table>'

    @staticmethod
    def _safe_link(value: Any) -> str:
        url = str(value or '').strip()
        return url if urlsplit(url).scheme.lower() in {'http', 'https', 'mailto'} else ''

    def _render_inline(self, block: WriterBlock) -> str:  # noqa: C901
        content = block.content or ''
        spans = block.spans if ''.join(span.text for span in block.spans) == content else []
        boundaries = {0, len(content)}
        span_ranges = []
        offset = 0
        for span in spans:
            end = offset + len(span.text)
            boundaries.update({offset, end})
            span_ranges.append((offset, end, span.style or {}))
            offset = end
        link_ranges = []
        for reference in block.references:
            start, end = reference.get('start'), reference.get('end')
            url = self._safe_link(reference.get('url'))
            if url and isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(content):
                boundaries.update({start, end})
                link_ranges.append((start, end, url))

        positions = sorted(boundaries)
        parts: list[str] = []
        for index, start in enumerate(positions[:-1]):
            end = positions[index + 1]
            if start == end:
                continue
            value = escape(content[start:end])
            style = next((item for left, right, item in span_ranges if left <= start < right), {})
            if style.get('inline_code') or style.get('code'):
                value = f'<code>{value}</code>'
            if style.get('bold') or style.get('strong'):
                value = f'<strong>{value}</strong>'
            if style.get('italic'):
                value = f'<em>{value}</em>'
            if style.get('strikethrough') or style.get('strike'):
                value = f'<del>{value}</del>'
            styled_link = style.get('link')
            link = ''
            if isinstance(styled_link, dict):
                if styled_link.get('type') != 'internal_ref':
                    link = self._safe_link(styled_link.get('url'))
            if not link:
                link = next((url for left, right, url in link_ranges if left <= start < right), '')
            if link:
                value = f'<a href="{escape(link, quote=True)}">{value}</a>'
            parts.append(value)
        return ''.join(parts)


__all__ = ['WeChatWriterAdapter']
