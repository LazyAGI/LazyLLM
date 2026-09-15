from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from ..utils import (
    strip_caption_numbering, strip_heading_numbering, strip_math_delimiters,
    table_grid, validate_writer_tables,
)
from ..data_models.revision import PatchHunk
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.writer_ir import (
    WRITER_BLOCK_MUTABLE_FIELDS,
    WriterBlock,
    WriterDocument,
    WriterSpan,
    WriterStage,
)
from .base import NativeBlock, NativePatchOperation, WriterAdapterBase


_BLOCK_TYPE_FIELDS: Dict[int, str] = {
    1: 'page', 2: 'text', 3: 'heading1', 4: 'heading2', 5: 'heading3', 6: 'heading4',
    7: 'heading5', 8: 'heading6', 9: 'heading7', 10: 'heading8', 11: 'heading9',
    12: 'bullet', 13: 'ordered', 14: 'code', 15: 'quote', 17: 'todo',
    19: 'callout', 22: 'divider', 24: 'grid', 25: 'grid_column', 27: 'image',
    31: 'table', 32: 'table_cell', 34: 'quote_container',
}

_BLOCK_TYPE_NAMES: Dict[int, str] = {
    1: 'document', 2: 'paragraph',
    **{block_type: 'heading' for block_type in range(3, 12)},
    12: 'list_item', 13: 'list_item', 14: 'code', 15: 'quote', 17: 'todo',
    19: 'callout', 22: 'divider', 24: 'grid', 25: 'grid_column', 27: 'image',
    31: 'table', 32: 'table_cell', 34: 'quote_container', 48: 'link_preview',
}
_IR_BLOCK_TYPES: Dict[str, int] = {
    ir_type: block_type
    for block_type, ir_type in _BLOCK_TYPE_NAMES.items()
    if ir_type not in {'heading', 'list_item'} and block_type in _BLOCK_TYPE_FIELDS
}

_TEXT_BLOCK_TYPES = frozenset(range(2, 16)) | {17}
_STYLE_TO_IR = {
    'bold': 'bold',
    'italic': 'italic',
    'underline': 'underline',
    'strikethrough': 'strikethrough',
    'inline_code': 'inline_code',
}
_VALUE_STYLE_FIELDS = {
    'text_color', 'background_color', 'font_size', 'font_family',
}


def feishu_block_url(document_uri: str, document_id: str, block_id: str) -> str:
    base = str(document_uri or '').split('#', 1)[0]
    if not base.startswith(('http://', 'https://')):
        base = f'https://feishu.cn/docx/{document_id}'
    return f'{base}#{block_id}'


# Feishu TextStyle.language enum; keep numeric IDs stable.
_CODE_LANGUAGES = dict(enumerate((
    'plaintext abap ada apache apex assembly bash csharp c++ c '
    'cobol css coffeescript d dart delphi django dockerfile erlang fortran '
    'foxpro go groovy html htmlbars http haskell json java javascript '
    'julia kotlin latex lisp logo lua matlab makefile markdown nginx '
    'objective-c openedgeabl php perl postscript powershell prolog protobuf python r '
    'rpg ruby rust sas scss sql scala scheme scratch shell '
    'swift thrift typescript vbscript visual-basic xml yaml cmake diff gherkin '
    'graphql glsl properties solidity toml '
).split(), start=1))
_CODE_LANGUAGE_IDS = {name: value for value, name in _CODE_LANGUAGES.items()}
_CODE_LANGUAGE_ALIASES = {
    '': 'plaintext', 'plain text': 'plaintext', 'text': 'plaintext',
    'py': 'python', 'js': 'javascript', 'ts': 'typescript', 'sh': 'shell',
    'c#': 'csharp', 'cs': 'csharp', 'cpp': 'c++', 'yml': 'yaml',
    'objectivec': 'objective-c', 'visual basic': 'visual-basic',
}


_ELEMENT_TEXT_FIELDS = ('content', 'title', 'name', 'text')


class FeishuWriterAdapter(WriterAdapterBase):
    '''Convert between Feishu Docx blocks and Writer IR.'''

    provider = 'feishu'
    materializes_table_captions = True

    @staticmethod
    def _written_temporary_id(block: WriterBlock) -> str:
        return block.provider_binding.get('block_id') or block.node_id

    @staticmethod
    def _written_caption_id(temporary_id: str) -> str:
        return f'{temporary_id}-caption'

    @staticmethod
    def _new_caption_node_id(table_node_id: str) -> str:
        return f'{table_node_id}-caption'

    @staticmethod
    def _prepare_caption_delete(
        operation: NativePatchOperation, native_index: int,
    ) -> NativePatchOperation:
        if operation.operation == 'delete':
            operation.params.update(start_index=native_index, end_index=native_index + 1)
        return operation

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
                params.update(start_index=source_index, end_index=source_index + 1 + bool(caption_id))
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
                    params['_caption_operation'] = NativePatchOperation('move', caption_params)
        return operation

    def blocks_to_ir(  # noqa: C901
        self,
        blocks: List[NativeBlock],
        *,
        external_document_id: str,
        stage: WriterStage = 'final',
        title: str = '',
        uri: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> WriterDocument:
        external_document_id = self._require_identifier(
            external_document_id, 'external_document_id')
        if not isinstance(blocks, list):
            raise TypeError(f'blocks must be a list, got {type(blocks).__name__}.')

        raw_by_id, source_order = self._index_raw_blocks(blocks)

        child_ids = self._build_child_relations(raw_by_id, source_order)
        self._validate_relations(child_ids, source_order)

        writer_by_id = {
            block_id: self._raw_block_to_ir(
                raw_by_id[block_id],
                source_index=index,
                external_document_id=external_document_id,
                stage=stage,
                revision=revision,
            )
            for index, block_id in enumerate(source_order)
        }
        self._restore_internal_references(writer_by_id)
        page_ids = {
            block_id
            for block_id in source_order
            if raw_by_id[block_id].get('block_type') == 1
        }

        def visible_children(block_ids: List[str]) -> List[WriterBlock]:
            visible: List[WriterBlock] = []
            for block_id in block_ids:
                if block_id in page_ids:
                    visible.extend(visible_children(child_ids.get(block_id, [])))
                else:
                    visible.append(writer_by_id[block_id])
            return visible

        for parent_id, children in child_ids.items():
            if parent_id not in page_ids:
                writer_by_id[parent_id].children = visible_children(children)

        for table in writer_by_id.values():
            if table.type != 'table':
                continue
            native_cells = [cell for cell in table.children if cell.type == 'table_cell']
            columns = int(
                (((table.provider_payload.get('raw_block') or {}).get('table') or {})
                 .get('property') or {}).get('column_size') or len(native_cells) or 1
            )
            rows: List[WriterBlock] = []
            for offset in range(0, len(native_cells), columns):
                cells = native_cells[offset:offset + columns]
                for cell in cells:
                    cell.provider_payload['table_content_blocks'] = [
                        deepcopy(child.provider_payload.get('raw_block') or {})
                        for child in cell.children
                    ]
                    content = '\n'.join(child.content for child in cell.children)
                    spans: List[WriterSpan] = []
                    for index, child in enumerate(cell.children):
                        if index:
                            spans.append(WriterSpan(text='\n'))
                        spans.extend(deepcopy(child.spans) or [WriterSpan(text=child.content)])
                    cell.content = content
                    cell.spans = spans
                    cell.children = []
                    cell.editable = True
                rows.append(WriterBlock(
                    node_id=f'{table.node_id}-row-{len(rows) + 1}',
                    type='table_row', children=cells, stage=stage,
                ))
            table.children = rows

        nested_ids = {child_id for children in child_ids.values() for child_id in children}
        root_ids = [block_id for block_id in source_order if block_id not in nested_ids]
        # A Feishu Page block is the provider's document container. WriterDocument
        # already represents that container, so expose only its content blocks in IR.
        root_blocks = visible_children(root_ids)
        resolved_title = title
        if not resolved_title:
            document_block = next((
                writer_by_id[block_id]
                for block_id in source_order
                if raw_by_id[block_id].get('block_type') == 1
            ), None)
            if document_block is not None:
                resolved_title = document_block.content
        binding: Dict[str, Any] = {
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
            title=resolved_title,
            blocks=root_blocks,
            revision=revision,
            metadata={'source_block_count': len(blocks)},
            provider_binding=binding,
            ui_editable=False,
        )
        validate_writer_tables(document)
        return document

    @staticmethod
    def _restore_internal_references(writer_by_id: Dict[str, WriterBlock]) -> None:
        for block in writer_by_id.values():
            for span in block.spans:
                link = span.style.get('link')
                if not isinstance(link, dict):
                    continue
                url = unquote(str(link.get('url') or ''))
                block_id = url.rsplit('#', 1)[-1] if '#' in url else ''
                target = writer_by_id.get(block_id)
                if target is None:
                    continue
                span.style['link'] = {
                    'type': 'internal_ref',
                    'target_node_id': target.node_id,
                }

    @staticmethod
    def _index_raw_blocks(
        blocks: List[NativeBlock],
    ) -> Tuple[Dict[str, NativeBlock], List[str]]:
        raw_by_id: Dict[str, NativeBlock] = {}
        source_order: List[str] = []
        for index, raw in enumerate(blocks):
            if not isinstance(raw, dict):
                raise TypeError(f'blocks[{index}] must be a dict, got {type(raw).__name__}.')
            block_id = raw.get('block_id')
            if not isinstance(block_id, str) or not block_id.strip():
                raise ValueError(f'blocks[{index}].block_id must be a non-empty string.')
            block_id = block_id.strip()
            if block_id in raw_by_id:
                raise ValueError(f'duplicate Feishu block_id: {block_id!r}.')
            raw_by_id[block_id] = raw
            source_order.append(block_id)
        return raw_by_id, source_order

    def ir_to_blocks(  # noqa: C901
        self, document: WriterDocument, media_assets: Any = None,
    ) -> List[NativeBlock]:
        if not isinstance(document, WriterDocument):
            raise TypeError(
                f'document must be a WriterDocument, got {type(document).__name__}.')
        provider = document.provider_binding.get('provider')
        if provider and str(provider).lower() != self.provider:
            raise ValueError(
                f'document provider must be {self.provider!r}, got {provider!r}.')
        validate_writer_tables(document)
        media_library = None if media_assets is None else MediaAssetLibrary.model_validate(media_assets)

        flat_blocks: List[Tuple[WriterBlock, Optional[WriterBlock]]] = []

        def walk(items: List[WriterBlock], parent: Optional[WriterBlock] = None) -> None:
            for block in items:
                flat_blocks.append((block, parent))
                if block.type != 'table':
                    walk(block.children, block)

        walk(document.blocks)

        output_ids: Dict[str, str] = {}
        used_output_ids: Set[str] = set()
        for block, _ in flat_blocks:
            output_id = self._output_block_id(block)
            if output_id in used_output_ids:
                raise ValueError(f'duplicate output Feishu block_id: {output_id!r}.')
            output_ids[block.node_id] = output_id
            used_output_ids.add(output_id)

        external_document_id = document.provider_binding.get('document_id') or ''
        document_uri = document.provider_binding.get('uri') or ''

        def resolve_internal_ref(span: WriterSpan) -> str | None:
            link = span.style.get('link')
            if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                return None
            target_id = link.get('target_node_id')
            target_block_id = output_ids.get(target_id) if isinstance(target_id, str) else None
            if not target_block_id:
                return None
            return feishu_block_url(document_uri, external_document_id, target_block_id)

        output: List[NativeBlock] = []
        for block, parent in flat_blocks:
            caption = self._table_caption_block(block) if block.type == 'table' else None
            if caption is not None:
                caption_raw = self._ir_block_to_raw(
                    caption, media_library, resolve_internal_ref,
                )
                caption_raw['block_id'] = f'{output_ids[block.node_id]}-caption'
                if parent is not None and parent.type != 'heading':
                    caption_raw['parent_id'] = output_ids[parent.node_id]
                output.append(caption_raw)
            raw = self._ir_block_to_raw(block, media_library, resolve_internal_ref)
            raw['block_id'] = output_ids[block.node_id]
            raw.pop('children', None)
            # Headings are sections, not visual containers.
            if parent is not None and parent.type != 'heading':
                raw['parent_id'] = output_ids[parent.node_id]
            else:
                raw.pop('parent_id', None)
            output.append(raw)
        return output

    @staticmethod
    def materialize_internal_links(
        blocks: List[NativeBlock], *, document_uri: str, document_id: str,
    ) -> List[NativeBlock]:
        output = deepcopy(blocks)
        temporary_ids = {
            str(block.get('block_id') or '')
            for block in output
            if isinstance(block, dict) and block.get('block_id')
        }

        def visit(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit(item)
                return
            if not isinstance(value, dict):
                return
            link = value.get('link')
            url = link.get('url') if isinstance(link, dict) else None
            fragment = urlparse(url).fragment if isinstance(url, str) else ''
            if fragment in temporary_ids:
                link['url'] = feishu_block_url(document_uri, document_id, fragment)
            for item in value.values():
                visit(item)

        visit(output)
        return output

    def patch_to_operation(
        self, patch: PatchHunk, document: WriterDocument, media_assets: Any = None,
    ) -> NativePatchOperation:
        return self._table_patch_to_operation(patch, document, media_assets)

    def _patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
        media_assets: Any = None,
    ) -> NativePatchOperation:
        if not isinstance(patch, PatchHunk):
            raise TypeError(f'patch must be a PatchHunk, got {type(patch).__name__}.')
        if not isinstance(document, WriterDocument):
            raise TypeError(
                f'document must be a WriterDocument, got {type(document).__name__}.')

        handlers = {
            'update': self._update_patch_to_operation,
            'create': self._create_patch_to_operation,
            'delete': self._delete_patch_to_operation,
            'move': self._move_patch_to_operation,
        }
        if patch.modify_type == 'create':
            return self._create_patch_to_operation(patch, document, media_assets)
        return handlers[patch.modify_type](patch, document)

    def merge_refreshed_document(  # noqa: C901
        self,
        previous_document: WriterDocument,
        refreshed_document: WriterDocument,
        patch: Optional[PatchHunk] = None,
        operation: Optional[NativePatchOperation] = None,
        operation_result: Optional[Dict[str, Any]] = None,
    ) -> WriterDocument:
        refreshed_ids = {
            block.provider_binding.get('block_id'): block.node_id
            for block in refreshed_document.iter_blocks()
            if isinstance(block.provider_binding.get('block_id'), str)
        }
        previous_ids = {
            block.provider_binding.get('block_id'): block.node_id
            for block in previous_document.iter_blocks()
            if isinstance(block.provider_binding.get('block_id'), str)
        }
        for block in refreshed_document.iter_blocks():
            node_id = previous_ids.get(block.provider_binding.get('block_id'))
            if node_id is not None:
                block.node_id = node_id

        if operation is not None and operation.operation == 'create':
            relations = (
                operation_result.get('node_id_bindings')
                if isinstance(operation_result, dict) else None
            )
            if not isinstance(relations, dict) or not relations:
                raise ValueError('create operation did not return Feishu node ID bindings.')
            refreshed_by_block_id = {
                block.provider_binding.get('block_id'): block
                for block in refreshed_document.iter_blocks()
                if isinstance(block.provider_binding.get('block_id'), str)
            }
            for temporary_id, created_id in relations.items():
                refreshed = refreshed_by_block_id.get(created_id)
                if isinstance(temporary_id, str) and refreshed is not None:
                    refreshed.node_id = temporary_id
                    if patch is not None and patch.block is not None \
                            and patch.block.node_id == temporary_id:
                        refreshed.references = deepcopy(patch.block.references)

        if operation is not None and operation.operation in {'move', 'replace'}:
            provider_id_remap = (
                operation_result.get('provider_id_remap')
                if isinstance(operation_result, dict) else None
            )
            if not isinstance(provider_id_remap, dict) or not provider_id_remap:
                raise ValueError(
                    f'{operation.operation} operation did not return Feishu provider ID remap.')
            refreshed_by_block_id = {
                block.provider_binding.get('block_id'): block
                for block in refreshed_document.iter_blocks()
                if isinstance(block.provider_binding.get('block_id'), str)
            }
            for source_block_id, created_block_id in provider_id_remap.items():
                node_id = previous_ids.get(source_block_id)
                refreshed = refreshed_by_block_id.get(created_block_id)
                if node_id is None or refreshed is None:
                    raise ValueError(
                        'provider ID remap does not match the refreshed document.')
                refreshed.node_id = node_id

        self._rebase_internal_reference_targets(refreshed_document, refreshed_ids)

        previous_blocks = {block.node_id: block for block in previous_document.iter_blocks()}
        for block in refreshed_document.iter_blocks():
            previous = previous_blocks.get(block.node_id)
            if previous is None:
                continue
            block.references = deepcopy(previous.references)
            if block.type == previous.type == 'heading':
                block.numbering = deepcopy(previous.numbering)
        refreshed_document = self._restore_table_state(previous_document, refreshed_document)
        return WriterDocument.model_validate(refreshed_document.model_dump())

    @staticmethod
    def _rebase_internal_reference_targets(
        document: WriterDocument,
        refreshed_ids: Dict[str, str],
    ) -> None:
        node_id_remap = {}
        for block in document.iter_blocks():
            block_id = block.provider_binding.get('block_id')
            refreshed_id = refreshed_ids.get(block_id)
            if isinstance(refreshed_id, str) and refreshed_id != block.node_id:
                node_id_remap[refreshed_id] = block.node_id
        if not node_id_remap:
            return
        for block in document.iter_blocks():
            for span in block.spans:
                link = span.style.get('link')
                if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                    continue
                target_id = link.get('target_node_id')
                if isinstance(target_id, str) and target_id in node_id_remap:
                    link['target_node_id'] = node_id_remap[target_id]

    def _update_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        '''Convert a semantic block update into a Feishu update or replacement.'''
        block = document.block_by_id(patch.target_node_id)
        if block is None:
            raise ValueError(f'patch target node does not exist: {patch.target_node_id!r}.')
        if patch.block is None:
            raise ValueError('update patch must provide block.')
        if block.type == 'table_cell':
            if patch.block.type != 'table_cell' or patch.block.children:
                raise ValueError('Feishu table cell updates must keep the table_cell structure.')
            content_blocks = block.provider_payload.get('table_content_blocks') or []
            text_blocks = [
                raw for raw in content_blocks
                if isinstance(raw, dict) and raw.get('block_type') in _TEXT_BLOCK_TYPES
                and isinstance(raw.get('block_id'), str) and raw['block_id']
            ]
            if not text_blocks:
                raise ValueError('Feishu table cell is missing its writable text block binding.')
            desired = block.model_copy(deep=True)
            for field in WRITER_BLOCK_MUTABLE_FIELDS:
                setattr(desired, field, deepcopy(getattr(patch.block, field)))
            return NativePatchOperation(
                operation='update',
                params={'requests': [
                    {
                        'block_id': raw['block_id'],
                        'update_text_elements': {
                            'elements': self._spans_to_elements(desired) if index == 0 else [],
                        },
                    }
                    for index, raw in enumerate(text_blocks)
                ]},
            )
        block_id = self._require_feishu_binding(block, 'update target')
        raw_block = self._raw_payload(block)
        original_type = raw_block.get('block_type')
        if original_type not in _TEXT_BLOCK_TYPES or not block.editable:
            raise ValueError(
                f'Feishu block type {original_type!r} does not support updates.')

        desired = block.model_copy(deep=True)
        for field in WRITER_BLOCK_MUTABLE_FIELDS:
            setattr(desired, field, deepcopy(getattr(patch.block, field)))
        desired_raw = self._ir_block_to_raw(desired)
        desired_type = desired_raw.get('block_type')

        if desired_type != original_type:
            _, parent, index = self._block_location(document, block.node_id)
            content_field = _BLOCK_TYPE_FIELDS.get(desired_type)
            if content_field is None:
                raise ValueError(
                    f'Feishu block type {desired_type!r} cannot replace a block.')
            replacement_block = {
                'block_type': desired_type,
                content_field: deepcopy(desired_raw.get(content_field) or {}),
            }
            return NativePatchOperation(
                operation='replace',
                params={
                    'parent_block_id': self._parent_block_id(document, block, parent),
                    'source_block_id': block_id,
                    'source_index': index,
                    'replacement_block': replacement_block,
                },
            )

        request = {
            'block_id': block_id,
            'update_text_elements': {
                'elements': desired_raw[_BLOCK_TYPE_FIELDS[desired_type]]['elements'],
            },
        }
        return NativePatchOperation(operation='update', params={'requests': [request]})

    def _create_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
        media_assets: Any = None,
    ) -> NativePatchOperation:
        '''Convert a semantic block creation into native Feishu blocks.'''
        if patch.block is None or patch.index is None:
            raise ValueError('create patch requires block and index.')
        parent_block_id = document.provider_binding.get('document_id')
        if patch.parent_node_id is not None:
            parent = document.block_by_id(patch.parent_node_id)
            if parent is None:
                raise ValueError(
                    f'create parent {patch.parent_node_id!r} is absent from document.')
            parent_block_id = self._require_feishu_binding(parent, 'create parent')
        if not isinstance(parent_block_id, str) or not parent_block_id:
            raise ValueError('create patch does not have a Feishu parent binding.')

        inserted_document = WriterDocument(
            document_id=f'{document.document_id}::create::{patch.target_node_id}',
            stage=document.stage,
            blocks=[patch.block.model_copy(deep=True)],
            provider_binding={
                'provider': self.provider,
                'document_id': document.provider_binding.get('document_id', ''),
            },
        )
        native_blocks = self.ir_to_blocks(inserted_document, media_assets=media_assets)
        return NativePatchOperation(
            operation='create',
            params={
                'parent_block_id': parent_block_id,
                'index': patch.index,
                'blocks': native_blocks,
            },
        )

    def _delete_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        '''Convert delete into Feishu delete_block parameters.'''
        block, parent, index = self._block_location(document, patch.target_node_id)
        self._require_feishu_binding(block, 'delete target')
        if block.type == 'document':
            raise ValueError('delete patch cannot remove the Feishu document block.')
        return NativePatchOperation(
            operation='delete',
            params={
                'parent_block_id': self._parent_block_id(document, block, parent),
                'start_index': index,
                'end_index': index + 1,
            },
        )

    def _move_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        '''Convert move into Feishu move_block parameters.'''
        source, source_parent, source_index = self._block_location(
            document, patch.target_node_id)
        source_block_id = self._require_feishu_binding(source, 'move source')
        if source.type == 'document':
            raise ValueError('move patch cannot move the Feishu document block.')
        if patch.index is None:
            raise ValueError('move patch requires index.')
        if patch.parent_node_id and self._subtree_contains(source, patch.parent_node_id):
            raise ValueError('move target parent cannot be inside the source subtree.')

        source_parent_block_id = self._parent_block_id(document, source, source_parent)
        target_parent_block_id = document.provider_binding.get('document_id')
        if patch.parent_node_id is not None:
            target_parent = document.block_by_id(patch.parent_node_id)
            if target_parent is None:
                raise ValueError(
                    f'move parent {patch.parent_node_id!r} is absent from document.')
            target_parent_block_id = self._require_feishu_binding(
                target_parent, 'move parent')
        if not isinstance(target_parent_block_id, str) or not target_parent_block_id:
            raise ValueError('move patch does not have a Feishu target parent binding.')

        return NativePatchOperation(
            operation='move',
            params={
                'source_parent_block_id': source_parent_block_id,
                'source_block_id': source_block_id,
                'source_index': source_index,
                'target_parent_block_id': target_parent_block_id,
                'target_index': patch.index,
            },
        )

    @staticmethod
    def _block_location(
        document: WriterDocument,
        node_id: str,
    ) -> Tuple[WriterBlock, Optional[WriterBlock], int]:
        def find(
            blocks: List[WriterBlock],
            parent: Optional[WriterBlock],
        ) -> Optional[Tuple[WriterBlock, Optional[WriterBlock], int]]:
            for index, block in enumerate(blocks):
                if block.node_id == node_id:
                    return block, parent, index
                nested = find(block.children, block)
                if nested is not None:
                    return nested
            return None

        location = find(document.blocks, None)
        if location is None:
            raise ValueError(f'patch target node does not exist: {node_id!r}.')
        return location

    def _require_feishu_binding(self, block: WriterBlock, label: str) -> str:
        provider = block.provider_binding.get('provider')
        if provider != self.provider:
            raise ValueError(
                f'{label} provider must be {self.provider!r}, got {provider!r}.')
        block_id = block.provider_binding.get('block_id')
        if not isinstance(block_id, str) or not block_id.strip():
            raise ValueError(f'{label} does not have a Feishu block_id binding.')
        return block_id.strip()

    def _parent_block_id(
        self,
        document: WriterDocument,
        block: WriterBlock,
        parent: Optional[WriterBlock],
    ) -> str:
        if parent is not None:
            return self._require_feishu_binding(parent, 'parent block')
        parent_block_id = block.provider_binding.get('parent_block_id')
        if not isinstance(parent_block_id, str) or not parent_block_id.strip():
            parent_block_id = document.provider_binding.get('document_id')
        if not isinstance(parent_block_id, str) or not parent_block_id.strip():
            raise ValueError('patch target does not have a Feishu parent block binding.')
        return parent_block_id.strip()

    @staticmethod
    def _subtree_contains(root: WriterBlock, node_id: str) -> bool:
        return root.node_id == node_id or any(
            FeishuWriterAdapter._subtree_contains(child, node_id)
            for child in root.children
        )

    def _raw_block_to_ir(
        self,
        raw: NativeBlock,
        *,
        source_index: int,
        external_document_id: str,
        stage: WriterStage,
        revision: Optional[str],
    ) -> WriterBlock:
        block_id = raw['block_id'].strip()
        block_type = raw.get('block_type')
        ir_type = _BLOCK_TYPE_NAMES.get(block_type, 'feishu_unknown')
        content, spans = self._content_and_spans(raw)
        if block_type == 27:
            caption = ((raw.get('image') or {}).get('caption') or {}).get('content')
            content = caption if isinstance(caption, str) else ''
            content = strip_caption_numbering(content)
            spans = []
        numbering: Dict[str, Any] = {}
        if isinstance(block_type, int) and 3 <= block_type <= 11:
            numbering['level'] = block_type - 2
            content = strip_heading_numbering(content)
            if spans and spans[0].text:
                spans[0].text = strip_heading_numbering(spans[0].text)
        elif block_type in (12, 13):
            numbering['ordered'] = block_type == 13

        binding: Dict[str, Any] = {
            'provider': self.provider,
            'document_id': external_document_id,
            'block_id': block_id,
        }
        parent_id = raw.get('parent_id')
        if parent_id is not None:
            binding['parent_block_id'] = parent_id
        if revision is not None:
            binding['revision'] = revision

        return WriterBlock(
            node_id=self.make_node_id(external_document_id, block_id),
            type=ir_type,
            content=content,
            spans=spans,
            stage=stage,
            numbering=numbering,
            provider_binding=binding,
            provider_payload={
                'raw_block': deepcopy(raw),
                'source_index': source_index,
                **({'code_language': _CODE_LANGUAGES.get(
                    ((raw.get('code') or {}).get('style') or {}).get('language'), '')}
                   if block_type == 14 else {}),
            },
            editable=block_type in _TEXT_BLOCK_TYPES,
        )

    def _ir_block_to_raw(
        self,
        block: WriterBlock,
        media_assets: Optional[MediaAssetLibrary] = None,
        internal_ref_resolver: Any = None,
    ) -> NativeBlock:
        if block.type == 'image':
            return self._ir_image_block_to_raw(block, media_assets)
        return self._ir_non_image_block_to_raw(block, internal_ref_resolver)

    def _ir_non_image_block_to_raw(
        self,
        block: WriterBlock,
        internal_ref_resolver: Any = None,
    ) -> NativeBlock:
        original = self._raw_payload(block)
        raw = deepcopy(original)
        original_type = original.get('block_type')
        block_type = self._block_type_from_ir(block, original_type)

        if block.type == 'math':
            expression = strip_math_delimiters(block.content)
            if not expression.startswith(r'\begin{'):
                expression = r'\begin{equation}' + expression + r'\end{equation}'
            return {'block_type': 2, 'text': {'elements': [
                {'equation': {'content': expression}},
            ]}}
        if block.type == 'table':
            grid = table_grid(block)
            columns = len(grid[0])
            raw['block_type'] = 31
            table_payload = deepcopy(raw.get('table') or {})
            table_payload.pop('cells', None)
            table_payload.pop('merge_info', None)
            table_property = deepcopy(table_payload.get('property') or {})
            table_property.pop('merge_info', None)
            raw['table'] = {
                **table_payload,
                'property': {
                    **table_property,
                    'row_size': len(grid),
                    'column_size': columns,
                },
            }
            raw['_table_cells'] = [
                [
                    self._spans_to_elements(cell, internal_ref_resolver)
                    if cell is not None and cell.content else None
                    for cell in row
                ]
                for row in grid
            ]
            raw['_table_cell_ids'] = [
                [cell.node_id if cell is not None else '' for cell in row]
                for row in grid
            ]
            return raw

        if not block.editable and original:
            original_content, original_spans = self._content_and_spans(original)
            if (
                block.type != _BLOCK_TYPE_NAMES.get(original_type, 'feishu_unknown')
                or block.content != original_content
                or block.spans != original_spans
            ):
                raise ValueError(
                    f'non-editable Feishu block {block.node_id!r} was modified.')

        raw['block_type'] = block_type
        content_field = _BLOCK_TYPE_FIELDS.get(block_type)
        if content_field is None:
            if original:
                return raw
            raise ValueError(
                f'Writer block type {block.type!r} cannot be converted to a new Feishu block.')

        original_content, original_spans = self._content_and_spans(original)
        same_visible_content = (
            original
            and block_type == original_type
            and block.content == original_content
            and block.spans == original_spans
        )
        if block_type == 14:
            code = raw.setdefault('code', {})
            style = code.get('style') or {}
            if original_type == 14:
                language_id = style.get('language')
                if language_id not in _CODE_LANGUAGES:
                    language_id = 1
            else:
                language = str(block.provider_payload.get('code_language') or '').strip().lower()
                name = _CODE_LANGUAGE_ALIASES.get(language, language)
                language_id = _CODE_LANGUAGE_IDS.get(name, 1)
            code['style'] = {**style, 'language': language_id}
        if same_visible_content:
            return raw

        for field in _BLOCK_TYPE_FIELDS.values():
            if field != content_field:
                raw.pop(field, None)
        content_payload = deepcopy(raw.get(content_field) or {})
        if block_type == 22:
            content_payload = {}
        elif block_type in _TEXT_BLOCK_TYPES:
            content_payload['elements'] = self._spans_to_elements(
                block, internal_ref_resolver)
        raw[content_field] = content_payload
        raw['plain_text'] = block.content
        return raw

    def _ir_image_block_to_raw(
        self,
        block: WriterBlock,
        media_assets: Optional[MediaAssetLibrary],
    ) -> NativeBlock:
        original = self._raw_payload(block)
        if original:
            caption = ((original.get('image') or {}).get('caption') or {}).get('content') or ''
            if block.content not in {caption, strip_caption_numbering(caption)}:
                raise ValueError('Updating an existing Feishu image is not supported.')
            return deepcopy(original)
        return self._image_block_to_raw(block, media_assets)

    @staticmethod
    def _image_block_to_raw(
        block: WriterBlock,
        media_assets: Optional[MediaAssetLibrary],
    ) -> NativeBlock:
        references = [
            reference for reference in block.references
            if reference.get('type') == 'media_asset' and reference.get('id')
        ]
        if len(references) != 1:
            raise ValueError('A new image block requires exactly one media_asset reference.')
        asset_id = str(references[0]['id'])
        asset = media_assets.assets.get(asset_id) if media_assets else None
        local_path = Path(asset.local_path) if asset and asset.local_path else None
        if asset is None or (not asset.uri and (local_path is None or not local_path.is_file())):
            raise ValueError(f'Image media asset {asset_id!r} is unavailable.')
        media = {'media_asset_id': asset_id}
        if asset.uri:
            media['uri'] = asset.uri
        if local_path is not None and local_path.is_file():
            media.update(local_path=str(local_path), file_name=local_path.name)
        return {
            'block_type': 27,
            'image': {
                'align': 2,
                'caption': {'content': block.content},
            },
            '_media': media,
        }

    def _block_type_from_ir(self, block: WriterBlock, original_type: Any) -> int:
        if block.type == 'math':
            return 2
        if block.type == 'heading':
            level = block.numbering.get('level')
            if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 9:
                raise ValueError('heading blocks require numbering.level from 1 to 9.')
            return level + 2
        if block.type == 'list_item':
            ordered = block.numbering.get('ordered')
            if not isinstance(ordered, bool):
                raise ValueError('list_item blocks require boolean numbering.ordered.')
            return 13 if ordered else 12
        mapped = _IR_BLOCK_TYPES.get(block.type)
        if mapped is not None:
            return mapped
        if original_type is not None:
            return original_type
        raise ValueError(
            f'Writer block type {block.type!r} cannot be converted to a Feishu block.')

    def _build_child_relations(
        self,
        raw_by_id: Dict[str, NativeBlock],
        source_order: List[str],
    ) -> Dict[str, List[str]]:
        relations: Dict[str, List[str]] = {block_id: [] for block_id in source_order}
        owner: Dict[str, str] = {}

        for parent_id in source_order:
            children = raw_by_id[parent_id].get('children') or []
            if not isinstance(children, list):
                raise TypeError(f'Feishu block {parent_id!r}.children must be a list.')
            for child_id in children:
                if child_id not in raw_by_id:
                    continue
                previous_owner = owner.get(child_id)
                if previous_owner and previous_owner != parent_id:
                    raise ValueError(
                        f'Feishu block {child_id!r} belongs to multiple parents.')
                if child_id not in relations[parent_id]:
                    relations[parent_id].append(child_id)
                    owner[child_id] = parent_id

        for child_id in source_order:
            parent_id = raw_by_id[child_id].get('parent_id')
            if parent_id not in raw_by_id or child_id in owner:
                continue
            relations[parent_id].append(child_id)
            owner[child_id] = parent_id
        return relations

    @staticmethod
    def _validate_relations(relations: Dict[str, List[str]], source_order: List[str]) -> None:
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(block_id: str) -> None:
            if block_id in visiting:
                raise ValueError(f'cycle detected in Feishu block hierarchy at {block_id!r}.')
            if block_id in visited:
                return
            visiting.add(block_id)
            for child_id in relations[block_id]:
                visit(child_id)
            visiting.remove(block_id)
            visited.add(block_id)

        for block_id in source_order:
            visit(block_id)

    @classmethod
    def _content_and_spans(cls, raw: NativeBlock) -> Tuple[str, List[WriterSpan]]:
        block_type = raw.get('block_type')
        content_field = _BLOCK_TYPE_FIELDS.get(block_type)
        elements = ((raw.get(content_field) or {}).get('elements') or []) if content_field else []
        spans: List[WriterSpan] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            equation = element.get('equation')
            if isinstance(equation, dict):
                raw_style = equation.get('text_element_style') or {}
                styles = {'math_source': True}
                styles.update({ir_style: True for field, ir_style in _STYLE_TO_IR.items()
                               if raw_style.get(field) is True})
                styles.update({field: deepcopy(raw_style[field]) for field in _VALUE_STYLE_FIELDS
                               if field in raw_style})
                spans.append(WriterSpan(text='$$' + str(equation.get('content') or '') + '$$',
                                        style=styles))
                continue
            text_run = element.get('text_run')
            if isinstance(text_run, dict):
                text = text_run.get('content')
                if not isinstance(text, str):
                    text = ''
                raw_style = text_run.get('text_element_style') or {}
                styles: Dict[str, Any] = {
                    ir_style: True
                    for feishu_style, ir_style in _STYLE_TO_IR.items()
                    if raw_style.get(feishu_style) is True
                }
                styles.update({
                    field: deepcopy(raw_style[field])
                    for field in _VALUE_STYLE_FIELDS
                    if field in raw_style
                })
                if isinstance(raw_style.get('link'), dict) and raw_style['link'].get('url'):
                    styles['link'] = {'url': raw_style['link']['url']}
                spans.append(WriterSpan(text=text, style=styles))
                continue

            for element_type, value in element.items():
                text = cls._element_plain_text(value)
                if text:
                    spans.append(WriterSpan(
                        text=text,
                        style={'feishu:element_type': element_type},
                    ))
                break

        if spans:
            return ''.join(span.text for span in spans), spans
        plain_text = raw.get('plain_text')
        return (plain_text if isinstance(plain_text, str) else ''), []

    @staticmethod
    def _element_plain_text(value: Any) -> str:
        if not isinstance(value, dict):
            return ''
        for field in _ELEMENT_TEXT_FIELDS:
            text = value.get(field)
            if isinstance(text, str):
                return text
        return ''

    @classmethod
    def _spans_to_elements(
        cls,
        block: WriterBlock,
        internal_ref_resolver: Any = None,
    ) -> List[Dict[str, Any]]:
        if not block.spans:
            return cls._plain_text_elements(block.content)
        elements: List[Dict[str, Any]] = []
        for span in block.spans:
            provider_element = span.style.get('feishu:element_type')
            if provider_element:
                raise ValueError(
                    f'cannot reconstruct provider element {provider_element!r} from WriterSpan.')
            raw_style = {
                feishu_style: True
                for feishu_style, ir_style in _STYLE_TO_IR.items()
                if span.style.get(ir_style) is True
            }
            raw_style.update({
                field: deepcopy(span.style[field])
                for field in _VALUE_STYLE_FIELDS
                if field in span.style
            })
            if span.style.get('math_source'):
                equation = {'content': strip_math_delimiters(span.text)}
                if raw_style:
                    equation['text_element_style'] = raw_style
                elements.append({'equation': equation})
                continue
            link = span.style.get('link')
            link_url: Optional[str] = None
            if isinstance(link, dict) and isinstance(link.get('url'), str):
                link_url = link['url']
            elif internal_ref_resolver is not None:
                link_url = internal_ref_resolver(span)
            if link_url:
                raw_style['link'] = {'url': link_url}
            text_run: Dict[str, Any] = {'content': span.text}
            if raw_style:
                text_run['text_element_style'] = raw_style
            elements.append({'text_run': text_run})
        return elements

    @classmethod
    def _replace_text_elements(
        cls,
        raw: NativeBlock,
        original_text: str,
        replacement_text: str,
    ) -> List[Dict[str, Any]]:
        '''Replace visible text while preserving unaffected Feishu rich-text elements.'''
        content_field = _BLOCK_TYPE_FIELDS.get(raw.get('block_type'))
        elements = deepcopy(
            ((raw.get(content_field) or {}).get('elements') or [])
            if content_field else []
        )
        if not elements:
            return cls._plain_text_elements(replacement_text)

        element_text = ''.join(cls._element_text(element) for element in elements)
        if element_text != original_text:
            raise ValueError(
                'Feishu raw elements do not match the current Writer block content.')
        if original_text == replacement_text:
            return elements

        start = 0
        common_limit = min(len(original_text), len(replacement_text))
        while start < common_limit and original_text[start] == replacement_text[start]:
            start += 1

        suffix = 0
        suffix_limit = common_limit - start
        while (
            suffix < suffix_limit
            and original_text[-suffix - 1] == replacement_text[-suffix - 1]
        ):
            suffix += 1

        original_end = len(original_text) - suffix
        replacement_end = len(replacement_text) - suffix
        inserted_text = replacement_text[start:replacement_end]
        prefix = cls._slice_elements(elements, 0, start)
        tail = cls._slice_elements(elements, original_end, len(original_text))

        inserted: List[Dict[str, Any]] = []
        if inserted_text:
            template = cls._replacement_text_run_template(elements, start, original_end)
            text_run = deepcopy(template) if template is not None else {}
            text_run['content'] = inserted_text
            inserted.append({'text_run': text_run})
        return cls._merge_adjacent_text_runs(prefix + inserted + tail)

    @classmethod
    def _slice_elements(
        cls,
        elements: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> List[Dict[str, Any]]:
        sliced: List[Dict[str, Any]] = []
        offset = 0
        for element in elements:
            text = cls._element_text(element)
            element_start, element_end = offset, offset + len(text)
            offset = element_end
            overlap_start = max(start, element_start)
            overlap_end = min(end, element_end)
            if overlap_start >= overlap_end:
                continue
            if isinstance(element.get('text_run'), dict):
                copied = deepcopy(element)
                copied['text_run']['content'] = text[
                    overlap_start - element_start:overlap_end - element_start]
                sliced.append(copied)
            elif overlap_start == element_start and overlap_end == element_end:
                sliced.append(deepcopy(element))
            else:
                raise ValueError('replace patch cannot split a non-text Feishu element.')
        return sliced

    @classmethod
    def _replacement_text_run_template(
        cls,
        elements: List[Dict[str, Any]],
        start: int,
        end: int,
    ) -> Optional[Dict[str, Any]]:
        offset = 0
        previous: Optional[Dict[str, Any]] = None
        following: Optional[Dict[str, Any]] = None
        for element in elements:
            text = cls._element_text(element)
            element_start, element_end = offset, offset + len(text)
            offset = element_end
            text_run = element.get('text_run')
            if not isinstance(text_run, dict):
                continue
            if element_end <= start:
                previous = text_run
                continue
            if following is None:
                following = text_run
            if element_start < end and element_end > start:
                return text_run
            if start == end and element_start <= start <= element_end:
                return text_run
        return previous or following

    @staticmethod
    def _merge_adjacent_text_runs(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for element in elements:
            current = element.get('text_run')
            previous = merged[-1].get('text_run') if merged else None
            if isinstance(current, dict) and isinstance(previous, dict):
                current_format = {key: value for key, value in current.items() if key != 'content'}
                previous_format = {key: value for key, value in previous.items() if key != 'content'}
                if current_format == previous_format:
                    previous['content'] = previous.get('content', '') + current.get('content', '')
                    continue
            merged.append(element)
        return merged

    @classmethod
    def _element_text(cls, element: Dict[str, Any]) -> str:
        text_run = element.get('text_run')
        if isinstance(text_run, dict):
            content = text_run.get('content')
            return content if isinstance(content, str) else ''
        for value in element.values():
            return cls._element_plain_text(value)
        return ''

    @staticmethod
    def _plain_text_elements(text: str) -> List[Dict[str, Any]]:
        return [{'text_run': {'content': text}}]

    @staticmethod
    def _raw_payload(block: WriterBlock) -> NativeBlock:
        raw = block.provider_payload.get('raw_block')
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _output_block_id(block: WriterBlock) -> str:
        raw = FeishuWriterAdapter._raw_payload(block)
        candidates = (
            raw.get('block_id'),
            block.provider_binding.get('block_id'),
            block.node_id,
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        raise ValueError(f'Writer block {block.node_id!r} does not have a usable block ID.')


__all__ = ['FeishuWriterAdapter']
