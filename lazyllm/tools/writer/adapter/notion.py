from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchHunk
from ..data_models.writer_ir import (
    WRITER_BLOCK_MUTABLE_FIELDS,
    WriterBlock,
    WriterDocument,
    WriterSpan,
    WriterStage,
)
from ..utils import (
    strip_caption_numbering, strip_heading_numbering, strip_math_delimiters,
    table_grid, validate_writer_tables,
)
from .base import NativeBlock, NativePatchOperation, WriterAdapterBase


_BLOCK_TYPE_NAMES: Dict[str, str] = {
    'paragraph': 'paragraph',
    **{f'heading_{level}': 'heading' for level in range(1, 5)},
    'bulleted_list_item': 'list_item', 'numbered_list_item': 'list_item',
    'quote': 'quote', 'code': 'code', 'to_do': 'todo', 'callout': 'callout',
    'divider': 'divider', 'image': 'image', 'table': 'table', 'equation': 'math',
    'table_row': 'table_row', 'link_preview': 'link_preview',
    'column_list': 'grid', 'column': 'grid_column',
}

_EDITABLE_TEXT_TYPES = {
    'paragraph', 'heading_1', 'heading_2', 'heading_3', 'heading_4',
    'bulleted_list_item', 'numbered_list_item',
    'to_do', 'quote', 'callout', 'code',
}

_RICH_TEXT_BLOCK_TYPES = _EDITABLE_TEXT_TYPES
_CAPTION_BLOCK_TYPES = {'image'}
# Notion code.language values accepted for writes. Keep this in sync with the
# documented enum and fall back conservatively when reading newer values.
# https://developers.notion.com/reference/block#code
_WRITABLE_CODE_LANGUAGES = frozenset({
    'abap', 'arduino', 'bash', 'basic', 'c', 'c#', 'c++', 'clojure',
    'coffeescript', 'css', 'dart', 'diff', 'docker', 'elixir', 'elm', 'erlang',
    'f#', 'flow', 'fortran', 'gherkin', 'glsl', 'go', 'graphql', 'groovy',
    'haskell', 'html', 'java', 'javascript', 'json', 'julia', 'kotlin', 'latex',
    'less', 'lisp', 'livescript', 'lua', 'makefile', 'markdown', 'markup',
    'matlab', 'mermaid', 'nix', 'objective-c', 'ocaml', 'pascal', 'perl', 'php',
    'plain text', 'powershell', 'prolog', 'protobuf', 'python', 'r', 'reason',
    'ruby', 'rust', 'sass', 'scala', 'scheme', 'scss', 'shell', 'sql', 'swift',
    'typescript', 'vb.net', 'verilog', 'vhdl', 'visual basic', 'webassembly',
    'xml', 'yaml', 'java/c/c++/c#',
})
_IR_TO_BLOCK_TYPE = {
    'paragraph': 'paragraph',
    'quote': 'quote',
    'code': 'code',
    'todo': 'to_do',
    'callout': 'callout',
    'divider': 'divider',
    'image': 'image',
    'math': 'equation',
    'table': 'table',
    'table_row': 'table_row',
    'link_preview': 'link_preview',
    'grid': 'column_list',
    'grid_column': 'column',
}
_ANNOTATION_STYLES = {
    'bold': 'bold',
    'italic': 'italic',
    'strikethrough': 'strikethrough',
    'underline': 'underline',
    'code': 'inline_code',
}
_UUID_FRAGMENT_RE = re.compile(
    r'(?<![0-9a-fA-F])('
    r'[0-9a-fA-F]{32}|'
    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    r')(?![0-9a-fA-F])'
)


class NotionWriterAdapter(WriterAdapterBase):
    provider = 'notion'
    materializes_table_captions = True

    def _table_patch_to_operation(
        self, patch: PatchHunk, document: WriterDocument, media_assets: Any = None,
    ) -> NativePatchOperation:
        if not isinstance(patch, PatchHunk) or not isinstance(document, WriterDocument):
            raise TypeError('patch and document must be PatchHunk and WriterDocument instances.')
        current = document.block_by_id(patch.target_node_id)
        if patch.modify_type == 'update' and current is not None and current.type == 'table':
            return self._table_caption_update(patch, document, current, media_assets)
        operation = self._patch_to_operation(patch, document, media_assets)
        params = operation.params
        if operation.operation == 'replace':
            _, parent, index = self._block_location(document, patch.target_node_id)
            params['source_index'] = self._physical_index(parent.children if parent else document.blocks, index)
        if patch.modify_type == 'create':
            parent = document.block_by_id(patch.parent_node_id) if patch.parent_node_id else None
            siblings = parent.children if parent else document.blocks
            params['index'] = self._physical_index(siblings, patch.index)
        elif patch.modify_type in {'delete', 'move'}:
            current, parent, index = self._block_location(document, patch.target_node_id)
            siblings = parent.children if parent else document.blocks
            source_index = self._physical_index(siblings, index)
            caption_binding = self._caption_binding(current)
            caption_id = caption_binding.get('block_id')
            if patch.modify_type == 'delete':
                if caption_id:
                    params['_caption_operation'] = NativePatchOperation('delete', {'block_id': caption_id})
            else:
                target_parent = document.block_by_id(patch.parent_node_id) if patch.parent_node_id else None
                target_siblings = target_parent.children if target_parent else document.blocks
                target_siblings = [block for block in target_siblings if block.node_id != current.node_id]
                target_index = self._physical_index(target_siblings, patch.index)
                same_parent = params['source_parent_block_id'] == params['target_parent_block_id']
                params.update(source_index=source_index + bool(caption_id), target_index=target_index)
                if caption_id:
                    # Move the grid first, then place its caption immediately before it.
                    params['target_index'] += int(same_parent and source_index < target_index)
                    caption_params = {
                        'source_block_id': caption_id,
                        'source_parent_block_id': params['source_parent_block_id'],
                        'source_index': source_index + int(same_parent and target_index <= source_index),
                        'target_parent_block_id': params['target_parent_block_id'],
                        'target_index': target_index,
                    }
                    caption = self._bound_caption_block(current)
                    raw = self.ir_to_blocks(WriterDocument(
                        document_id=document.document_id, blocks=[caption],
                        provider_binding=deepcopy(document.provider_binding),
                    ))[0]
                    raw['_temporary_node_id'] = caption.node_id
                    caption_params['block'] = raw
                    params['_caption_operation'] = NativePatchOperation('move', caption_params)
        return operation

    def blocks_to_ir(self, blocks: List[NativeBlock], *, external_document_id: str,
                     stage: WriterStage = 'final', title: str = '', uri: Optional[str] = None,
                     revision: Optional[str] = None) -> WriterDocument:
        external_document_id = self._require_identifier(external_document_id, 'external_document_id')
        if not isinstance(blocks, list):
            raise TypeError(f'blocks must be a list, got {type(blocks).__name__}.')

        raw_by_id, source_order = self._index_raw_blocks(blocks)
        child_ids = self._build_child_relations(raw_by_id, source_order)
        self._validate_relations(child_ids, source_order)
        writer_by_id = {
            block_id: self._raw_block_to_ir(raw_by_id[block_id], source_index=index,
                                            external_document_id=external_document_id,
                                            stage=stage, revision=revision)
            for index, block_id in enumerate(source_order)
        }
        self._restore_internal_references(writer_by_id)

        for parent_id, children in child_ids.items():
            writer_by_id[parent_id].children = [
                writer_by_id[child_id] for child_id in children
            ] + writer_by_id[parent_id].children
        for table in writer_by_id.values():
            if table.type != 'table':
                continue
            table_payload = (table.provider_payload.get('raw_block') or {}).get('table') or {}
            has_column_header = bool(table_payload.get('has_column_header'))
            has_row_header = bool(table_payload.get('has_row_header'))
            for row_index, row in enumerate(table.children):
                if row.type != 'table_row':
                    continue
                native_cells = row.provider_payload.get('table_cells') or []
                row.content = ''
                row.children = []
                for cell_index, native_cell in enumerate(native_cells):
                    spans = self._rich_text_to_spans(
                        native_cell if isinstance(native_cell, list) else [],
                    )
                    row.children.append(WriterBlock(
                        node_id=f'{row.node_id}-cell-{cell_index + 1}',
                        type='table_cell',
                        content=''.join(span.text for span in spans),
                        spans=spans,
                        stage=stage,
                        numbering={'header': (
                            has_column_header and row_index == 0
                        ) or (has_row_header and cell_index == 0)},
                    ))
        nested_ids = {child_id for children in child_ids.values() for child_id in children}
        root_blocks = [writer_by_id[block_id] for block_id in source_order if block_id not in nested_ids]

        binding: Dict[str, Any] = {
            'provider': self.provider,
            'document_id': external_document_id,
        }
        if uri is not None:
            binding['uri'] = uri
        if revision is not None:
            binding['revision'] = revision
        document = WriterDocument(document_id=self.make_document_id(external_document_id), stage=stage, title=title,
                                  blocks=root_blocks, revision=revision,
                                  metadata={'source_block_count': len(blocks)}, provider_binding=binding,
                                  ui_editable=False)
        validate_writer_tables(document)
        return document

    def ir_to_blocks(self, document: WriterDocument, media_assets: Any = None) -> List[NativeBlock]:
        if not isinstance(document, WriterDocument):
            raise TypeError(f'document must be a WriterDocument, got {type(document).__name__}.')
        provider = document.provider_binding.get('provider')
        if provider and str(provider).lower() != self.provider:
            raise ValueError(f'document provider must be {self.provider!r}, got {provider!r}.')
        validate_writer_tables(document)

        blocks_by_node_id = {block.node_id: block for block in document.iter_blocks()}
        if len(blocks_by_node_id) != sum(1 for _ in document.iter_blocks()):
            raise ValueError('document contains duplicate Writer node_id values.')
        document_id = str(document.provider_binding.get('document_id') or '')
        document_uri = str(document.provider_binding.get('uri') or '')
        media_library = None if media_assets is None else MediaAssetLibrary.model_validate(media_assets)
        track_internal_refs = any(
            block.type == 'table' and block.content.strip() for block in document.iter_blocks()
        ) or any(
            isinstance(span.style.get('link'), dict)
            and span.style['link'].get('type') == 'internal_ref'
            for block in document.iter_blocks()
            for span in block.spans
        )

        def resolve_internal_ref(span: WriterSpan) -> Optional[str]:
            link = span.style.get('link')
            if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                return None
            target_node_id = link.get('target_node_id')
            target = blocks_by_node_id.get(target_node_id)
            if target is None:
                raise ValueError(f'internal reference target does not exist: {target_node_id!r}.')
            block_id = target.provider_binding.get('block_id')
            if not isinstance(block_id, str) or not block_id:
                block_id = document_id
            return self._notion_block_url(document_uri, document_id, block_id)

        return self._ir_blocks_to_raw(
            document.blocks,
            resolve_internal_ref,
            media_library=media_library,
            track_internal_refs=track_internal_refs,
        )

    @classmethod
    def materialize_internal_links(
        cls, blocks: List[NativeBlock], *, document_uri: str, document_id: str,
    ) -> List[NativeBlock]:
        output = deepcopy(blocks)
        target_url = cls._notion_block_url(document_uri, document_id, document_id)

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return
            text = value.get('text')
            link = text.get('link') if isinstance(text, dict) else None
            if isinstance(link, dict) and link.get('_target_node_id'):
                link['url'] = target_url
            for item in value.values():
                visit(item)

        visit(output)
        return output

    def _ir_blocks_to_raw(self, blocks: List[WriterBlock], resolve_internal_ref: Any, *,
                          media_library: Optional[MediaAssetLibrary] = None,
                          track_internal_refs: bool = False) -> List[NativeBlock]:
        '''Serialize logical Writer siblings using Notion's physical hierarchy.

        Writer headings own their section body logically. Ordinary Notion headings are
        not containers, so their logical children must follow them as physical siblings.
        Children of real container blocks remain nested; headings inside such containers
        flatten only within that container's children list.
        '''
        output: List[NativeBlock] = []
        for block in blocks:
            caption_block = self._table_caption_block(block) if block.type == 'table' else None
            if caption_block is not None:
                output.append(self._ir_block_to_raw(
                    caption_block,
                    resolve_internal_ref,
                    media_library=media_library,
                    track_internal_refs=track_internal_refs,
                ))
            output.append(self._ir_block_to_raw(
                block,
                resolve_internal_ref,
                media_library=media_library,
                track_internal_refs=track_internal_refs,
            ))
            if block.type == 'heading':
                output.extend(self._ir_blocks_to_raw(
                    block.children,
                    resolve_internal_ref,
                    media_library=media_library,
                    track_internal_refs=track_internal_refs,
                ))
        return output

    @staticmethod
    def has_reusable_image_payload(block: WriterBlock) -> bool:
        if block.type != 'image':
            return False
        if str(block.provider_binding.get('provider') or '').lower() != 'notion':
            return False
        raw = block.provider_payload.get('raw_block')
        if not isinstance(raw, dict) \
                or str(raw.get('type') or raw.get('block_type') or '') != 'image':
            return False
        image = raw.get('image')
        if not isinstance(image, dict):
            return False
        image_type = image.get('type')
        file_object = image.get(image_type) if image_type in {'file', 'external'} else None
        return isinstance(file_object, dict) and bool(file_object.get('url'))

    def _ir_block_to_raw(  # noqa: C901
            self, block: WriterBlock, resolve_internal_ref: Any, *,
            media_library: Optional[MediaAssetLibrary] = None,
            track_internal_refs: bool = False) -> NativeBlock:
        raw = self._raw_payload(block)
        original_type = str(raw.get('type') or raw.get('block_type') or '')
        block_type = self._notion_type_for_ir(block, original_type)

        if block.type == 'notion_unknown':
            if block.editable:
                raise ValueError('notion_unknown blocks must remain read-only.')
            if not block_type:
                raise ValueError('notion_unknown block does not preserve its Notion type.')
            output = deepcopy(raw)
        else:
            payload = deepcopy(
                raw.get(block_type)
                or (raw.get(original_type) if block_type in _RICH_TEXT_BLOCK_TYPES
                    and original_type in _RICH_TEXT_BLOCK_TYPES else {})
                or {})
            if not isinstance(payload, dict):
                payload = {}
            if block_type in _RICH_TEXT_BLOCK_TYPES:
                payload['rich_text'] = self._spans_to_rich_text(block, resolve_internal_ref)
                if block_type == 'code':
                    payload['language'] = self._code_language(block, payload)
            elif block_type == 'equation':
                payload['expression'] = strip_math_delimiters(block.content)
            elif block_type in _CAPTION_BLOCK_TYPES:
                payload['caption'] = self._spans_to_rich_text(block, resolve_internal_ref)
            elif block_type == 'link_preview':
                payload['url'] = block.content
            elif block_type == 'table':
                grid = table_grid(block)
                rows = block.children
                payload.update(
                    table_width=len(grid[0]),
                    has_column_header=bool(rows and any(
                        cell.numbering.get('header') for cell in rows[0].children
                    )),
                    has_row_header=bool(rows and all(
                        row.children and row.children[0].numbering.get('header')
                        for row in rows
                    )),
                )
            elif block_type == 'table_row':
                cells = [
                    self._spans_to_rich_text(cell, resolve_internal_ref)
                    for cell in block.children
                    if cell.type == 'table_cell'
                ]
                if isinstance(cells, list):
                    payload['cells'] = deepcopy(cells)
            output = {'object': 'block', 'type': block_type, block_type: payload}

        if block_type == 'image':
            self._attach_image_media(output, block, media_library)
        if track_internal_refs:
            output['_temporary_node_id'] = block.node_id

        nested_children = [] if block.type == 'table_row' else block.children
        if block.type == 'table':
            nested_children = []
            for row_index, (row, cells) in enumerate(zip(block.children, table_grid(block))):
                normalized_row = row.model_copy(deep=True)
                normalized_row.children = [
                    cell or WriterBlock(
                        node_id=f'{row.node_id}-covered-{row_index}-{cell_index}',
                        type='table_cell', stage=row.stage,
                    )
                    for cell_index, cell in enumerate(cells)
                ]
                nested_children.append(normalized_row)
        if nested_children and block.type != 'heading':
            output[block_type]['children'] = self._ir_blocks_to_raw(
                nested_children,
                resolve_internal_ref,
                media_library=media_library,
                track_internal_refs=track_internal_refs,
            )
        return output

    @staticmethod
    def _code_language(block: WriterBlock, payload: Dict[str, Any]) -> str:
        '''Resolve an edited IR language without losing the native round-trip value.'''
        raw_language = str(payload.get('language') or '').strip()
        provider_has_language = 'code_language' in block.provider_payload
        provider_language = str(block.provider_payload.get('code_language') or '').strip()
        ir_language = str(getattr(block, 'language', '') or '').strip()

        # A value changed away from raw_block is an explicit IR edit.  Notion's
        # provider field wins when both representations were edited differently.
        if provider_has_language and provider_language != raw_language:
            language = provider_language
        elif ir_language and ir_language != raw_language:
            language = ir_language
        else:
            language = provider_language or ir_language or raw_language
        return language if language in _WRITABLE_CODE_LANGUAGES else 'plain text'

    @staticmethod
    def _attach_image_media(output: NativeBlock, block: WriterBlock,
                            media_library: Optional[MediaAssetLibrary]) -> None:
        references = [
            reference for reference in block.references
            if reference.get('type') == 'media_asset' and reference.get('id')
        ]
        if references:
            if len(references) != 1:
                raise ValueError('A Notion image requires exactly one media_asset reference.')
            asset_id = str(references[0]['id'])
            asset = media_library.assets.get(asset_id) if media_library else None
            local_path = Path(asset.local_path) if asset and asset.local_path else None
            if asset is None or (not asset.uri and (local_path is None or not local_path.is_file())):
                raise ValueError(f'Image media asset {asset_id!r} is unavailable.')
            image = output.setdefault('image', {})
            for field in ('external', 'file', 'file_upload', 'type'):
                image.pop(field, None)
            if asset.uri:
                image.update(type='external', external={'url': asset.uri})
            media = {'media_asset_id': asset_id}
            if asset.uri:
                media['uri'] = asset.uri
            if local_path is not None and local_path.is_file():
                media.update(local_path=str(local_path), file_name=local_path.name)
            output['_media'] = media

    @staticmethod
    def _raw_payload(block: WriterBlock) -> NativeBlock:
        raw = block.provider_payload.get('raw_block')
        return deepcopy(raw) if isinstance(raw, dict) else {}

    @staticmethod
    def _notion_type_for_ir(block: WriterBlock, original_type: str) -> str:
        if block.type == 'heading':
            level = block.numbering.get('level', 1)
            if not isinstance(level, int) or level not in range(1, 5):
                raise ValueError(f'Notion heading level must be between 1 and 4, got {level!r}.')
            return f'heading_{level}'
        if block.type == 'list_item':
            return 'numbered_list_item' if block.numbering.get('ordered') else 'bulleted_list_item'
        if block.type == 'notion_unknown':
            return original_type
        block_type = _IR_TO_BLOCK_TYPE.get(block.type)
        if block_type is None:
            raise ValueError(f'Writer block type {block.type!r} cannot be written to Notion.')
        return block_type

    @classmethod
    def _spans_to_rich_text(cls, block: WriterBlock, resolve_internal_ref: Any) -> List[Dict[str, Any]]:
        spans = block.spans or ([WriterSpan(text=block.content)] if block.content else [])
        output: List[Dict[str, Any]] = []
        for span in spans:
            style = span.style
            rich_type = 'equation' if style.get('math_source') else style.get('notion:rich_text_type', 'text')
            annotations = {
                notion_style: style.get(writer_style) is True
                for notion_style, writer_style in _ANNOTATION_STYLES.items()
            }
            annotations['color'] = str(style.get('background_color') or style.get('text_color') or 'default')
            item: Dict[str, Any] = {
                'type': rich_type,
                'annotations': annotations,
            }
            if rich_type == 'equation':
                item['equation'] = {'expression': strip_math_delimiters(span.text)}
            elif rich_type == 'mention':
                preserved = style.get(f'notion:{rich_type}')
                if not isinstance(preserved, dict):
                    raise ValueError(f'Notion {rich_type} span is missing its native payload.')
                item[rich_type] = deepcopy(preserved)
            else:
                url = resolve_internal_ref(span)
                if url is None:
                    link = style.get('link')
                    url = link.get('url') if isinstance(link, dict) else None
                link_payload = {'url': url} if isinstance(url, str) and url else None
                link = style.get('link')
                if isinstance(link, dict) and link.get('type') == 'internal_ref':
                    if link_payload is None:
                        raise ValueError('Notion internal reference requires a placeholder URL.')
                    target_node_id = link.get('target_node_id')
                    if not isinstance(target_node_id, str) or not target_node_id:
                        raise ValueError('Notion internal reference requires a target node ID.')
                    link_payload['_target_node_id'] = target_node_id
                item['type'] = 'text'
                for start in range(0, max(1, len(span.text)), 2000):
                    text_item = deepcopy(item)
                    text_item['text'] = {
                        'content': span.text[start:start + 2000],
                        'link': deepcopy(link_payload),
                    }
                    output.append(text_item)
                continue
            output.append(item)
        return output

    @classmethod
    def _notion_block_url(cls, document_uri: str, document_id: str, block_id: str) -> str:
        fragment = cls._canonical_notion_id(block_id) or block_id.replace('-', '')
        if document_uri:
            return f'{document_uri.split("#", 1)[0]}#{fragment}'
        page = cls._canonical_notion_id(document_id) or document_id.replace('-', '')
        return f'https://www.notion.so/{page}#{fragment}'

    def patch_to_operation(
        self, patch: PatchHunk, document: WriterDocument, media_assets: Any = None,
    ) -> NativePatchOperation:
        return self._table_patch_to_operation(patch, document, media_assets)

    def _patch_to_operation(  # noqa: C901
            self, patch: PatchHunk, document: WriterDocument,
            media_assets: Any = None) -> NativePatchOperation:
        if not isinstance(patch, PatchHunk):
            raise TypeError(f'patch must be a PatchHunk, got {type(patch).__name__}.')
        if not isinstance(document, WriterDocument):
            raise TypeError(f'document must be a WriterDocument, got {type(document).__name__}.')
        if patch.modify_type == 'create':
            return self._create_patch_to_operation(
                patch, document, media_assets=media_assets)
        if patch.modify_type == 'delete':
            return self._delete_patch_to_operation(patch, document)
        if patch.modify_type == 'move':
            return self._move_patch_to_operation(
                patch, document, media_assets=media_assets)
        if patch.modify_type != 'update':
            raise NotImplementedError(
                f'NotionWriterAdapter does not support {patch.modify_type!r} patches yet.')
        current = document.block_by_id(patch.target_node_id)
        if current is None:
            raise ValueError(f'patch target node does not exist: {patch.target_node_id!r}.')
        if patch.block is None:
            raise ValueError('update patch must provide block.')
        if current.type == 'table_cell':
            if patch.block.type != 'table_cell' or patch.block.children:
                raise ValueError('Notion table cell updates must keep the table_cell structure.')
            _, row, _ = self._block_location(document, current.node_id)
            if row is None or row.type != 'table_row':
                raise ValueError('Notion table cell is missing its table_row parent.')
            desired_row = row.model_copy(deep=True)
            desired_cell = desired_row.children[next(
                index for index, cell in enumerate(row.children)
                if cell.node_id == current.node_id
            )]
            for field in WRITER_BLOCK_MUTABLE_FIELDS:
                setattr(desired_cell, field, deepcopy(getattr(patch.block, field)))
            raw = self._native_update_block(desired_row, document)
            raw.get('table_row', {}).pop('children', None)
            block_id = row.provider_binding.get('block_id')
            if not isinstance(block_id, str) or not block_id:
                raise ValueError('Notion table row is missing its block_id binding.')
            return NativePatchOperation(
                operation='update', params={'block_id': block_id, 'block': raw})
        if current.type == 'image' \
                and patch.meta.get('source') == 'system_numbering' \
                and patch.meta.get('update_scope') == 'caption':
            if patch.block.type != 'image':
                raise ValueError('system image numbering cannot change the block type.')
            block_id = current.provider_binding.get('block_id')
            if not isinstance(block_id, str) or not block_id:
                raise ValueError('Notion image update target is missing its block_id binding.')
            desired = current.model_copy(deep=True)
            desired.content = patch.block.content
            desired.spans = deepcopy(patch.block.spans)
            return NativePatchOperation(
                operation='update',
                params={
                    'block_id': block_id,
                    'block': {
                        'object': 'block',
                        'type': 'image',
                        'image': {
                            'caption': self._spans_to_rich_text(desired, lambda _span: None),
                        },
                    },
                },
            )
        if current.type == 'code' \
                and patch.meta.get('source') == 'system_numbering' \
                and patch.meta.get('update_scope') == 'caption':
            if patch.block.type != 'code':
                raise ValueError('system code numbering cannot change the block type.')
            block_id = current.provider_binding.get('block_id')
            if not isinstance(block_id, str) or not block_id:
                raise ValueError('Notion code update target is missing its block_id binding.')
            caption = str(
                patch.block.provider_payload.get('numbering_caption') or '').strip()
            caption_block = WriterBlock(
                node_id=patch.block.node_id,
                type='paragraph',
                content=caption,
                spans=[WriterSpan(text=caption)] if caption else [],
            )
            return NativePatchOperation(
                operation='update',
                params={
                    'block_id': block_id,
                    'block': {
                        'object': 'block',
                        'type': 'code',
                        'code': {
                            'caption': self._spans_to_rich_text(
                                caption_block, lambda _span: None),
                        },
                    },
                },
            )
        if not current.editable:
            raise ValueError(f'Notion block {patch.target_node_id!r} does not support updates.')

        desired = current.model_copy(deep=True)
        for field in WRITER_BLOCK_MUTABLE_FIELDS:
            setattr(desired, field, deepcopy(getattr(patch.block, field)))
        raw = self._native_update_block(desired, document)
        original_type = str((current.provider_payload.get('raw_block') or {}).get('type') or '')
        if raw.get('type') != original_type:
            raise ValueError(
                f'Notion cannot update block type {original_type!r} in place to {raw.get("type")!r}.')
        block_id = current.provider_binding.get('block_id')
        if not isinstance(block_id, str) or not block_id:
            raise ValueError('Notion update target is missing its block_id binding.')
        raw.get(raw['type'], {}).pop('children', None)
        return NativePatchOperation(
            operation='update', params={'block_id': block_id, 'block': raw})

    def merge_refreshed_document(  # noqa: C901
        self,
        previous_document: WriterDocument,
        refreshed_document: WriterDocument,
        patch: Optional[PatchHunk] = None,
        operation: Optional[NativePatchOperation] = None,
        operation_result: Optional[Dict[str, Any]] = None,
    ) -> WriterDocument:
        previous_by_binding: Dict[Tuple[str, Optional[int]], WriterBlock] = {}
        for block in previous_document.iter_blocks():
            block_id = block.provider_binding.get('block_id')
            if isinstance(block_id, str):
                previous_by_binding[(self._canonical_notion_id(block_id), None)] = block
        for block in refreshed_document.iter_blocks():
            block_id = block.provider_binding.get('block_id')
            key = (self._canonical_notion_id(block_id), None)
            previous = previous_by_binding.get(key)
            if previous is not None:
                block.node_id = previous.node_id
                block.references = deepcopy(previous.references)
        if operation is not None and operation.operation in {'create', 'move'}:
            bindings = operation_result.get('node_id_bindings') \
                if isinstance(operation_result, dict) else None
            if not isinstance(bindings, dict) or not bindings:
                raise ValueError(
                    f'{operation.operation} operation did not return Notion node ID bindings.')
            refreshed_by_block_id = {
                self._canonical_notion_id(block.provider_binding.get('block_id')): block
                for block in refreshed_document.iter_blocks()
                if isinstance(block.provider_binding.get('block_id'), str)
            }
            created_by_node_id: Dict[str, WriterBlock] = {}
            for node_id, provider_id in bindings.items():
                created_id = self._canonical_notion_id(provider_id)
                created = refreshed_by_block_id.get(created_id)
                if isinstance(node_id, str) and created is not None:
                    created.node_id = node_id
                    created_by_node_id[node_id] = created
            desired_root = None
            if patch is not None:
                desired_root = patch.block if patch.block is not None \
                    else previous_document.block_by_id(patch.target_node_id)
            if desired_root is not None:
                desired_by_id = {
                    block.node_id: block for block in desired_root.iter_blocks()
                }
                for node_id, created in created_by_node_id.items():
                    desired = desired_by_id.get(node_id)
                    if desired is not None:
                        created.references = deepcopy(desired.references)
        refreshed_document = self._restore_table_state(previous_document, refreshed_document)
        return WriterDocument.model_validate(refreshed_document.model_dump())

    def _native_update_block(
        self, block: WriterBlock, document: WriterDocument,
    ) -> NativeBlock:
        blocks_by_node_id = {item.node_id: item for item in document.iter_blocks()}
        document_id = str(document.provider_binding.get('document_id') or '')
        document_uri = str(document.provider_binding.get('uri') or '')

        def resolve_internal_ref(span: WriterSpan) -> Optional[str]:
            link = span.style.get('link')
            if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                return None
            target = blocks_by_node_id.get(link.get('target_node_id'))
            if target is None:
                raise ValueError(
                    f'internal reference target does not exist: {link.get("target_node_id")!r}.')
            target_id = target.provider_binding.get('block_id') or document_id
            return self._notion_block_url(document_uri, document_id, str(target_id))

        return self._ir_block_to_raw(block, resolve_internal_ref)

    def _create_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
        media_assets: Any = None,
    ) -> NativePatchOperation:
        if patch.block is None or patch.index is None:
            raise ValueError('create patch requires block and index.')
        parent_block_id = document.provider_binding.get('document_id')
        if patch.parent_node_id is not None:
            parent = document.block_by_id(patch.parent_node_id)
            if parent is None:
                raise ValueError(
                    f'create parent {patch.parent_node_id!r} is absent from document.')
            parent_block_id = parent.provider_binding.get('block_id')
            if not isinstance(parent_block_id, str) or not parent_block_id:
                raise ValueError('Notion create parent must be a physical block.')
        if not isinstance(parent_block_id, str) or not parent_block_id:
            raise ValueError('create patch does not have a Notion parent binding.')

        blocks_by_node_id = {block.node_id: block for block in document.iter_blocks()}
        blocks_by_node_id.update({block.node_id: block for block in patch.block.iter_blocks()})
        document_id = str(document.provider_binding.get('document_id') or '')
        document_uri = str(document.provider_binding.get('uri') or '')

        def resolve_internal_ref(span: WriterSpan) -> Optional[str]:
            link = span.style.get('link')
            if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                return None
            target_node_id = link.get('target_node_id')
            target = blocks_by_node_id.get(target_node_id)
            if target is None:
                raise ValueError(
                    f'internal reference target does not exist: {target_node_id!r}.')
            block_id = target.provider_binding.get('block_id') or document_id
            return self._notion_block_url(document_uri, document_id, str(block_id))

        media_library = None if media_assets is None \
            else MediaAssetLibrary.model_validate(media_assets)
        native_blocks = self._ir_blocks_to_raw(
            [patch.block.model_copy(deep=True)],
            resolve_internal_ref,
            media_library=media_library,
            track_internal_refs=True,
        )
        return NativePatchOperation(
            operation='create',
            params={
                'parent_block_id': parent_block_id,
                'blocks': native_blocks,
                'index': patch.index,
            },
        )

    def _move_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
        media_assets: Any = None,
    ) -> NativePatchOperation:
        if patch.index is None:
            raise ValueError('move patch requires index.')
        source, source_parent, source_index = self._block_location(
            document, patch.target_node_id)
        source_block_id = source.provider_binding.get('block_id')
        if not isinstance(source_block_id, str) or not source_block_id:
            raise ValueError('Notion move source must be a physical block.')
        if patch.parent_node_id and any(
                block.node_id == patch.parent_node_id for block in source.iter_blocks()):
            raise ValueError('move target parent cannot be inside the source subtree.')

        target_parent_block_id = document.provider_binding.get('document_id')
        if patch.parent_node_id is not None:
            target_parent = document.block_by_id(patch.parent_node_id)
            if target_parent is None:
                raise ValueError(
                    f'move parent {patch.parent_node_id!r} is absent from document.')
            target_parent_block_id = target_parent.provider_binding.get('block_id')
            if not isinstance(target_parent_block_id, str) or not target_parent_block_id:
                raise ValueError('Notion move parent must be a physical block.')
        source_parent_block_id = document.provider_binding.get('document_id') \
            if source_parent is None else source_parent.provider_binding.get('block_id')
        if not isinstance(source_parent_block_id, str) or not source_parent_block_id:
            raise ValueError('Notion move source parent is missing its block_id binding.')
        if not isinstance(target_parent_block_id, str) or not target_parent_block_id:
            raise ValueError('Notion move target parent is missing its block_id binding.')

        resolve_internal_ref = self._move_link_resolver(document)

        media_library = None if media_assets is None \
            else MediaAssetLibrary.model_validate(media_assets)
        moved_source = source.model_copy(deep=True)
        for table in moved_source.iter_blocks():
            if self._caption_binding(table).get('block_id'):
                caption = self._bound_caption_block(table)
                table.content, table.spans = caption.content, caption.spans
        native = self._ir_block_to_raw(
            moved_source,
            resolve_internal_ref,
            media_library=media_library,
            track_internal_refs=True,
        )
        return NativePatchOperation(
            operation='move',
            params={
                'source_block_id': source_block_id,
                'source_parent_block_id': source_parent_block_id,
                'source_index': source_index,
                'target_parent_block_id': target_parent_block_id,
                'target_index': patch.index,
                'block': native,
            },
        )

    @staticmethod
    def _block_location(
        document: WriterDocument, node_id: str,
    ) -> Tuple[WriterBlock, Optional[WriterBlock], int]:
        def walk(
            blocks: List[WriterBlock], parent: Optional[WriterBlock],
        ) -> Optional[Tuple[WriterBlock, Optional[WriterBlock], int]]:
            for index, block in enumerate(blocks):
                if block.node_id == node_id:
                    return block, parent, index
                found = walk(block.children, block)
                if found is not None:
                    return found
            return None

        location = walk(document.blocks, None)
        if location is None:
            raise ValueError(f'patch target node does not exist: {node_id!r}.')
        return location

    @staticmethod
    def _delete_patch_to_operation(
        patch: PatchHunk, document: WriterDocument,
    ) -> NativePatchOperation:
        current = document.block_by_id(patch.target_node_id)
        if current is None:
            raise ValueError(f'patch target node does not exist: {patch.target_node_id!r}.')
        block_id = current.provider_binding.get('block_id')
        if not isinstance(block_id, str) or not block_id:
            raise ValueError('Notion delete target is missing its block_id binding.')
        return NativePatchOperation(
            operation='delete', params={'block_id': block_id})

    @staticmethod
    def _index_raw_blocks(blocks: List[NativeBlock]) -> Tuple[Dict[str, NativeBlock], List[str]]:
        raw_by_id: Dict[str, NativeBlock] = {}
        canonical_ids: Set[str] = set()
        source_order: List[str] = []
        for index, raw in enumerate(blocks):
            if not isinstance(raw, dict):
                raise TypeError(f'blocks[{index}] must be a dict, got {type(raw).__name__}.')
            block_id = raw.get('block_id') or raw.get('id')
            if not isinstance(block_id, str) or not block_id.strip():
                raise ValueError(f'blocks[{index}] must have a non-empty block_id or id.')
            block_id = block_id.strip()
            canonical_id = NotionWriterAdapter._canonical_notion_id(block_id)
            duplicate = block_id in raw_by_id or canonical_id and canonical_id in canonical_ids
            if duplicate:
                raise ValueError(f'duplicate Notion block id: {block_id!r}.')
            raw_by_id[block_id] = raw
            source_order.append(block_id)
            if canonical_id:
                canonical_ids.add(canonical_id)
        return raw_by_id, source_order

    @classmethod
    def _build_child_relations(cls, raw_by_id: Dict[str, NativeBlock],
                               source_order: List[str]) -> Dict[str, List[str]]:
        relations: Dict[str, List[str]] = {block_id: [] for block_id in source_order}
        canonical_to_id = {
            canonical: block_id
            for block_id in source_order
            if (canonical := cls._canonical_notion_id(block_id))
        }
        for child_id in source_order:
            parent_id = cls._parent_block_id(raw_by_id[child_id])
            if not parent_id:
                continue
            resolved_parent_id = parent_id if parent_id in raw_by_id else canonical_to_id.get(
                cls._canonical_notion_id(parent_id))
            if resolved_parent_id is not None:
                relations[resolved_parent_id].append(child_id)
        return relations

    @staticmethod
    def _parent_block_id(raw: NativeBlock) -> str:
        parent_id = raw.get('parent_id')
        if isinstance(parent_id, str) and parent_id.strip():
            return parent_id.strip()
        parent = raw.get('parent') or {}
        if not isinstance(parent, dict):
            return ''
        for field in ('block_id', 'page_id', 'data_source_id', 'database_id'):
            value = parent.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ''

    @staticmethod
    def _validate_relations(relations: Dict[str, List[str]], source_order: List[str]) -> None:
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(block_id: str) -> None:
            if block_id in visiting:
                raise ValueError(f'cycle detected in Notion block hierarchy at {block_id!r}.')
            if block_id in visited:
                return
            visiting.add(block_id)
            for child_id in relations.get(block_id, []):
                visit(child_id)
            visiting.remove(block_id)
            visited.add(block_id)

        for block_id in source_order:
            visit(block_id)

    def _raw_block_to_ir(  # noqa: C901
            self, raw: NativeBlock, *, source_index: int, external_document_id: str,
            stage: WriterStage, revision: Optional[str]) -> WriterBlock:
        block_id = str(raw.get('block_id') or raw.get('id') or '').strip()
        block_type = str(raw.get('block_type') or raw.get('type') or '').strip()
        ir_type = _BLOCK_TYPE_NAMES.get(block_type, 'notion_unknown')
        content, spans = self._content_and_spans(raw, block_type)
        numbering: Dict[str, Any] = {}
        if block_type.startswith('heading_') and block_type[-1:].isdigit():
            numbering['level'] = int(block_type[-1])
            content = strip_heading_numbering(content)
            if spans and spans[0].text:
                spans[0].text = strip_heading_numbering(spans[0].text)
        elif block_type in {'bulleted_list_item', 'numbered_list_item'}:
            numbering['ordered'] = block_type == 'numbered_list_item'
        if block_type == 'image':
            content = strip_caption_numbering(content)
            if spans and spans[0].text:
                spans[0].text = strip_caption_numbering(spans[0].text)

        binding: Dict[str, Any] = {
            'provider': self.provider,
            'document_id': external_document_id,
            'block_id': block_id,
        }
        parent_id = self._parent_block_id(raw)
        if parent_id:
            binding['parent_block_id'] = parent_id
        if revision is not None:
            binding['revision'] = revision
        payload: Dict[str, Any] = {'raw_block': deepcopy(raw), 'source_index': source_index}
        if block_type == 'code':
            payload['code_language'] = str((raw.get('code') or {}).get('language') or '')
        block = WriterBlock(node_id=self.make_node_id(external_document_id, block_id), type=ir_type,
                            content=content, spans=spans, stage=stage, numbering=numbering,
                            provider_binding=binding, provider_payload=payload,
                            editable=block_type in _EDITABLE_TEXT_TYPES)
        if block_type == 'image':
            image = raw.get('image') or {}
            image_type = image.get('type') if isinstance(image, dict) else None
            image_source = image.get(image_type) if image_type in {'file', 'external'} else None
            url = image_source.get('url') if isinstance(image_source, dict) else None
            if isinstance(url, str) and url:
                preview = {
                    'type': 'preview_asset',
                    'provider': self.provider,
                    'url': url,
                }
                expiry_time = image_source.get('expiry_time')
                if isinstance(expiry_time, str) and expiry_time:
                    preview['expires_at'] = expiry_time
                block.references.append(preview)
        if block_type == 'table_row':
            cells = (raw.get('table_row') or {}).get('cells') or []
            block.content = self._table_row_text(cells)
            block.provider_payload['table_cells'] = deepcopy(cells)
        return block

    @classmethod
    def _table_row_text(cls, cells: Any) -> str:
        if not isinstance(cells, list):
            return ''
        values: List[str] = []
        for cell in cells:
            rich_text = cell if isinstance(cell, list) else []
            values.append(''.join(span.text for span in cls._rich_text_to_spans(rich_text)))
        return ' | '.join(values)

    @classmethod
    def _content_and_spans(cls, raw: NativeBlock, block_type: str) -> Tuple[str, List[WriterSpan]]:
        payload = raw.get(block_type) or {}
        if not isinstance(payload, dict):
            payload = {}
        rich_text: Any = None
        if block_type in _RICH_TEXT_BLOCK_TYPES:
            rich_text = payload.get('rich_text')
        elif block_type in _CAPTION_BLOCK_TYPES:
            rich_text = payload.get('caption')
        if isinstance(rich_text, list):
            spans = cls._rich_text_to_spans(rich_text)
            if spans:
                return ''.join(span.text for span in spans), spans
        if block_type == 'equation':
            return '$$\n' + str(payload.get('expression') or '') + '\n$$', []
        if block_type == 'link_preview':
            url = payload.get('url')
            if isinstance(url, str):
                return url, []
        if block_type == 'table_row':
            return '', []
        plain_text = raw.get('plain_text')
        return (plain_text if isinstance(plain_text, str) else ''), []

    @classmethod
    def _rich_text_to_spans(cls, rich_text: List[Dict[str, Any]]) -> List[WriterSpan]:
        spans: List[WriterSpan] = []
        for item in rich_text:
            if not isinstance(item, dict):
                continue
            text = cls._rich_text_item_text(item)
            annotations = item.get('annotations') or {}
            style: Dict[str, Any] = {
                writer_style: True
                for notion_style, writer_style in _ANNOTATION_STYLES.items()
                if annotations.get(notion_style) is True
            }
            color = annotations.get('color')
            if isinstance(color, str) and color and color != 'default':
                field = 'background_color' if color.endswith('_background') else 'text_color'
                style[field] = color
            text_payload = item.get('text') or {}
            link = item.get('href') or ((text_payload.get('link') or {}).get('url')
                                        if isinstance(text_payload, dict) else None)
            if isinstance(link, str) and link:
                style['link'] = {'url': link}
            item_type = item.get('type')
            if item_type in {'mention', 'equation'}:
                style['notion:rich_text_type'] = item_type
                value = item.get(item_type)
                if isinstance(value, dict):
                    style[f'notion:{item_type}'] = deepcopy(value)
            spans.append(WriterSpan(text=text, style=style))
        return spans

    @staticmethod
    def _rich_text_item_text(item: Dict[str, Any]) -> str:
        plain_text = item.get('plain_text')
        if isinstance(plain_text, str) and plain_text:
            return plain_text
        text = item.get('text') or {}
        if isinstance(text, dict) and isinstance(text.get('content'), str):
            return text['content']
        equation = item.get('equation') or {}
        if isinstance(equation, dict) and isinstance(equation.get('expression'), str):
            return equation['expression']
        mention = item.get('mention') or {}
        if isinstance(mention, dict):
            mention_type = mention.get('type')
            value = mention.get(mention_type) if isinstance(mention_type, str) else None
            if isinstance(value, dict):
                for field in ('name', 'id', 'url'):
                    if isinstance(value.get(field), str) and value[field]:
                        return value[field]
            if isinstance(value, str):
                return value
        return ''

    @staticmethod
    def _canonical_notion_id(value: Any) -> str:
        if not isinstance(value, str):
            return ''
        compact = value.strip().replace('-', '').lower()
        return compact if re.fullmatch(r'[0-9a-f]{32}', compact) else ''

    @classmethod
    def _restore_internal_references(cls, writer_by_id: Dict[str, WriterBlock]) -> None:
        all_blocks: List[WriterBlock] = []

        def collect(block: WriterBlock) -> None:
            all_blocks.append(block)
            for child in block.children:
                collect(child)

        for block in writer_by_id.values():
            collect(block)
        targets = {canonical: block for block in all_blocks
                   if (canonical := cls._canonical_notion_id(block.provider_binding.get('block_id')))}
        for block in all_blocks:
            for span in block.spans:
                link = span.style.get('link')
                if not isinstance(link, dict) or not isinstance(link.get('url'), str):
                    continue
                url = unquote(link['url'])
                fragment = urlparse(url).fragment
                match = _UUID_FRAGMENT_RE.search(fragment)
                target = targets.get(cls._canonical_notion_id(match.group(1))) if match else None
                if target is not None:
                    span.style['link'] = {'type': 'internal_ref', 'target_node_id': target.node_id}

    def _move_link_resolver(self, document):
        blocks_by_node_id = {block.node_id: block for block in document.iter_blocks()}
        document_id = str(document.provider_binding.get('document_id') or '')
        document_uri = str(document.provider_binding.get('uri') or '')

        def resolve_internal_ref(span: WriterSpan) -> Optional[str]:
            link = span.style.get('link')
            if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                return None
            target_node_id = link.get('target_node_id')
            target = blocks_by_node_id.get(target_node_id)
            if target is None:
                raise ValueError(
                    f'internal reference target does not exist: {target_node_id!r}.')
            target_id = target.provider_binding.get('block_id') or document_id
            return self._notion_block_url(document_uri, document_id, str(target_id))
        return resolve_internal_ref


__all__ = ['NotionWriterAdapter']
