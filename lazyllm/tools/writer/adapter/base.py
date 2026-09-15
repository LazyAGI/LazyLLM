from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from uuid import NAMESPACE_URL, uuid5

from ..data_models.revision import PatchHunk
from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterSpan, WriterStage
from ..utils.serialization import strip_caption_numbering


NativeBlock = Dict[str, Any]
NativePatchOperationType = Literal['create', 'update', 'replace', 'delete', 'move']


@dataclass(frozen=True)
class NativePatchOperation:
    '''One provider-native operation produced from a Writer patch hunk.'''

    operation: NativePatchOperationType
    params: Dict[str, Any]

_WRITER_ID_NAMESPACE = uuid5(NAMESPACE_URL, 'https://lazyllm.ai/writer-ir')


class WriterAdapterBase(ABC):
    '''Convert between provider-native document blocks and Writer IR.'''

    provider: str = ''
    materializes_table_captions: bool = False

    @classmethod
    def _table_caption_block(cls, table: WriterBlock) -> Optional[WriterBlock]:
        if not cls.materializes_table_captions or table.type != 'table' or not table.content.strip():
            return None
        return WriterBlock(
            node_id=f'{table.node_id}::caption', type='paragraph',
            content=table.content, spans=deepcopy(table.spans) or [WriterSpan(text=table.content)],
            stage=table.stage,
        )

    @classmethod
    def _caption_binding(cls, table: WriterBlock) -> Dict[str, Any]:
        if not cls.materializes_table_captions:
            return {}
        caption = table.provider_payload.get('table_caption') if table.type == 'table' else None
        binding = caption.get('provider_binding') if isinstance(caption, dict) else None
        return binding if isinstance(binding, dict) else {}

    @classmethod
    def _physical_index(cls, blocks: List[WriterBlock], index: int) -> int:
        return index + sum(bool(cls._caption_binding(block).get('block_id')) for block in blocks[:index])

    def _bound_caption_block(self, table: WriterBlock) -> WriterBlock:
        if not self.materializes_table_captions:
            raise NotImplementedError(
                f'{type(self).__name__} does not materialize table captions.')
        saved = table.provider_payload.get('table_caption') or {}
        caption_spans = [WriterSpan.model_validate(span) for span in saved['spans']] \
            if 'spans' in saved else deepcopy(table.spans)
        content = saved.get('content', table.content)
        if ''.join(span.text for span in caption_spans) != content:
            caption_spans = [WriterSpan(text=content)] if content else []
        return WriterBlock(
            node_id=f'{table.node_id}::caption', type='paragraph',
            content=content, stage=table.stage, spans=caption_spans,
            provider_binding=deepcopy(self._caption_binding(table)),
            provider_payload=deepcopy(saved.get('provider_payload') or {
                'raw_block': saved.get('raw_block') or {},
            }),
        )

    def _table_caption_update(
        self, patch: PatchHunk, document: WriterDocument, current: WriterBlock, media_assets: Any,
    ) -> NativePatchOperation:
        if not self.materializes_table_captions:
            raise NotImplementedError(
                f'{type(self).__name__} does not materialize table captions.')
        desired = patch.block

        def grid_snapshot(block: WriterBlock) -> List[Dict[str, Any]]:
            return [{
                **child.model_dump(exclude={'provider_binding', 'provider_payload', 'editable', 'children'}),
                'children': grid_snapshot(child),
            } for child in block.children]

        if desired is None or desired.type != 'table' or grid_snapshot(desired) != grid_snapshot(current):
            raise ValueError('Table updates must keep the grid; update individual table cells separately.')
        caption = self._bound_caption_block(current)
        _, parent, index = self._block_location(document, current.node_id)
        siblings = parent.children if parent else document.blocks
        native_index = self._physical_index(siblings, index)
        caption_document = document.model_copy(deep=True)
        caption_document.blocks = [caption]
        if self._caption_binding(current).get('block_id'):
            caption_patch = PatchHunk(
                target_node_id=caption.node_id, modify_type='update' if desired.content.strip() else 'delete',
                block=caption.model_copy(update={'content': desired.content, 'spans': deepcopy(desired.spans)})
                if desired.content.strip() else None,
            )
            operation = self._patch_to_operation(caption_patch, caption_document, media_assets)
            return self._prepare_caption_delete(operation, native_index)
        if not desired.content.strip():
            raise ValueError('Table has no bound caption to update.')
        caption = self._table_caption_block(desired)
        caption.node_id = self._new_caption_node_id(current.node_id)
        operation = self._patch_to_operation(PatchHunk(
            target_node_id=caption.node_id, modify_type='create', block=caption,
            parent_node_id=patch.parent_node_id, index=0,
        ), document, media_assets)
        operation.params['parent_block_id'] = current.provider_binding.get('parent_block_id') or \
            document.provider_binding['document_id']
        operation.params['index'] = native_index
        return operation

    @staticmethod
    def _new_caption_node_id(table_node_id: str) -> str:
        return f'{table_node_id}::caption'

    @staticmethod
    def _prepare_caption_delete(operation: NativePatchOperation, native_index: int) -> NativePatchOperation:
        return operation

    @classmethod
    def _restore_table_state(
        cls, previous: WriterDocument, refreshed: WriterDocument,
    ) -> WriterDocument:
        if not cls.materializes_table_captions:
            return refreshed
        previous_by_node_id = {block.node_id: block for block in previous.iter_blocks()}
        for table in refreshed.iter_blocks():
            old_table = previous_by_node_id.get(table.node_id)
            if table.type != 'table' or old_table is None or old_table.type != 'table':
                continue
            if len(table.children) != len(old_table.children):
                continue
            for row, old_row in zip(table.children, old_table.children):
                if len(row.children) != len(old_row.children):
                    continue
                if not row.provider_binding.get('block_id'):
                    if [cell.node_id for cell in row.children] == [cell.node_id for cell in old_row.children]:
                        row.node_id = old_row.node_id
                elif row.node_id == old_row.node_id:
                    for cell, old_cell in zip(row.children, old_row.children):
                        if not cell.provider_binding.get('block_id'):
                            cell.node_id = old_cell.node_id
                            cell.references = deepcopy(old_cell.references)

        if previous.provider_binding.get('document_id') != refreshed.provider_binding.get('document_id'):
            return refreshed
        known_captions = {
            block.provider_binding.get('block_id'): cls._caption_binding(block).get('block_id')
            for block in previous.iter_blocks() if block.type == 'table'
        }

        def restore(blocks: List[WriterBlock]) -> List[WriterBlock]:
            output: List[WriterBlock] = []
            for block in blocks:
                block.children = restore(block.children)
                if block.type == 'table':
                    block.provider_payload.pop('table_caption', None)
                    expected = known_captions.get(block.provider_binding.get('block_id'))
                    caption = output[-1] if output else None
                    if expected and caption is not None and caption.type == 'paragraph' \
                            and caption.provider_binding.get('block_id') == expected \
                            and caption.provider_binding.get('parent_block_id') == \
                            block.provider_binding.get('parent_block_id'):
                        output.pop()
                        block.editable = True
                        block.content = strip_caption_numbering(caption.content)
                        block.spans = deepcopy(caption.spans)
                        prefix = caption.content.find(block.content) if block.content else len(caption.content)
                        while prefix > 0 and block.spans:
                            span = block.spans[0]
                            removed = min(prefix, len(span.text))
                            span.text = span.text[removed:]
                            prefix -= removed
                            if not span.text:
                                block.spans.pop(0)
                        if block.spans:
                            block.spans[-1].text = block.spans[-1].text.rstrip()
                        if ''.join(span.text for span in block.spans) != block.content:
                            block.spans = [WriterSpan(text=block.content)] if block.content else []
                        block.provider_payload['table_caption'] = {
                            'content': caption.content,
                            'spans': [span.model_dump() for span in caption.spans],
                            'provider_binding': deepcopy(caption.provider_binding),
                            'provider_payload': deepcopy(caption.provider_payload),
                        }
                output.append(block)
            return output

        refreshed.blocks = restore(refreshed.blocks)
        return refreshed

    @classmethod
    def _update_local_caption(cls, document: WriterDocument, patch: PatchHunk) -> None:
        if not cls.materializes_table_captions:
            return
        if patch.modify_type != 'update' or patch.block is None or patch.block.type != 'table':
            return
        table = document.block_by_id(patch.target_node_id)
        if table is None:
            return
        if not patch.block.content.strip():
            table.provider_payload.pop('table_caption', None)
        elif 'table_caption' in table.provider_payload:
            table.provider_payload['table_caption']['content'] = patch.block.content
            table.provider_payload['table_caption']['spans'] = [span.model_dump() for span in patch.block.spans]

    def bind_written_document(
        self, source: WriterDocument, relations: List[Dict[str, str]], refreshed: WriterDocument,
    ) -> WriterDocument:
        source = source.model_copy(deep=True)
        mapping = {item['temporary_block_id']: item['block_id'] for item in relations}
        for block in source.iter_blocks():
            temporary_id = self._written_temporary_id(block)
            block.provider_binding = {'provider': self.provider}
            if mapping.get(temporary_id):
                block.provider_binding['block_id'] = mapping[temporary_id]
            if self.materializes_table_captions and block.type == 'table':
                block.provider_payload.pop('table_caption', None)
                caption_id = mapping.get(self._written_caption_id(temporary_id))
                if caption_id and mapping.get(temporary_id):
                    block.provider_payload['table_caption'] = {
                        'provider_binding': {'provider': self.provider, 'block_id': caption_id},
                    }
        return self.merge_refreshed_document(source, refreshed)

    @staticmethod
    def _written_temporary_id(block: WriterBlock) -> str:
        return block.node_id

    @staticmethod
    def _written_caption_id(temporary_id: str) -> str:
        return f'{temporary_id}::caption'

    @classmethod
    def make_document_id(cls, external_document_id: str) -> str:
        '''Return a stable internal document ID for one provider document.'''
        provider = cls._provider_key()
        external_id = cls._require_identifier(external_document_id, 'external_document_id')
        value = uuid5(_WRITER_ID_NAMESPACE, f'document:{provider}:{external_id}')
        return f'writer-doc-{value}'

    @classmethod
    def make_node_id(cls, external_document_id: str, external_block_id: str) -> str:
        '''Return a stable internal node ID for one provider block.'''
        provider = cls._provider_key()
        document_id = cls._require_identifier(external_document_id, 'external_document_id')
        block_id = cls._require_identifier(external_block_id, 'external_block_id')
        value = uuid5(
            _WRITER_ID_NAMESPACE,
            f'node:{provider}:{document_id}:{block_id}',
        )
        return f'writer-node-{value}'

    @classmethod
    def _provider_key(cls) -> str:
        provider = cls.provider
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f'{cls.__name__}.provider must be a non-empty string.')
        return provider.strip().lower()

    @staticmethod
    def _require_identifier(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a non-empty string.')
        return value.strip()

    @abstractmethod
    def blocks_to_ir(
        self,
        blocks: List[NativeBlock],
        *,
        external_document_id: str,
        stage: WriterStage = 'final',
        title: str = '',
        uri: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> WriterDocument:
        '''Convert provider-native blocks into a WriterDocument.'''
        raise NotImplementedError

    @abstractmethod
    def ir_to_blocks(self, document: WriterDocument, media_assets: Any = None) -> List[NativeBlock]:
        '''Convert a WriterDocument into provider-native blocks.'''
        raise NotImplementedError

    @abstractmethod
    def patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
        media_assets: Any = None,
    ) -> NativePatchOperation:
        '''Convert one Writer patch hunk into a classified provider operation.'''
        raise NotImplementedError

    def materialize_internal_links(
        self,
        blocks: List[NativeBlock],
        *,
        document_uri: str,
        document_id: str,
    ) -> List[NativeBlock]:
        return deepcopy(blocks)

    def merge_refreshed_document(
        self,
        previous_document: WriterDocument,
        refreshed_document: WriterDocument,
        patch: Optional[PatchHunk] = None,
        operation: Optional[NativePatchOperation] = None,
        operation_result: Optional[Dict[str, Any]] = None,
    ) -> WriterDocument:
        raise NotImplementedError(
            f'{type(self).__name__} does not support merging refreshed documents.')

__all__ = [
    'NativeBlock',
    'NativePatchOperation',
    'NativePatchOperationType',
    'WriterAdapterBase',
]
