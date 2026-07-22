from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

from ..data_models.revision import PatchHunk
from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterSpan, WriterStage
from .base import NativeBlock, NativePatchOperation, WriterAdapterBase


_BLOCK_TYPE_FIELDS: Dict[int, str] = {
    1: 'page', 2: 'text',
    **{block_type: f'heading{block_type - 2}' for block_type in range(3, 12)},
    12: 'bullet', 13: 'ordered', 14: 'code', 15: 'quote', 17: 'todo',
    19: 'callout', 22: 'divider', 24: 'grid', 25: 'grid_column',
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
    'bold': 'strong',
    'italic': 'italic',
    'underline': 'underline',
    'strikethrough': 'strikethrough',
    'inline_code': 'code',
}
_STYLE_FROM_IR = {value: key for key, value in _STYLE_TO_IR.items()}
_ELEMENT_TEXT_FIELDS = ('content', 'title', 'name', 'text')
class FeishuWriterAdapter(WriterAdapterBase):
    '''Convert between Feishu Docx blocks and Writer IR.'''

    provider = 'feishu'

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
        external_document_id = self._require_identifier(
            external_document_id, 'external_document_id')
        if not isinstance(blocks, list):
            raise TypeError(f'blocks must be a list, got {type(blocks).__name__}.')

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
        for parent_id, children in child_ids.items():
            writer_by_id[parent_id].children = [writer_by_id[child_id] for child_id in children]

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

        return WriterDocument(
            document_id=self.make_document_id(external_document_id),
            stage=stage,
            title=title,
            blocks=root_blocks,
            revision=revision,
            metadata={'source_block_count': len(blocks)},
            provider_binding=binding,
        )

    def ir_to_blocks(self, document: WriterDocument) -> List[NativeBlock]:
        if not isinstance(document, WriterDocument):
            raise TypeError(
                f'document must be a WriterDocument, got {type(document).__name__}.')
        provider = document.provider_binding.get('provider')
        if provider and str(provider).lower() != self.provider:
            raise ValueError(
                f'document provider must be {self.provider!r}, got {provider!r}.')

        flat_blocks: List[Tuple[WriterBlock, Optional[WriterBlock]]] = []

        def walk(items: List[WriterBlock], parent: Optional[WriterBlock] = None) -> None:
            for block in items:
                flat_blocks.append((block, parent))
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

        output: List[NativeBlock] = []
        for block, parent in flat_blocks:
            raw = self._ir_block_to_raw(block)
            raw['block_id'] = output_ids[block.node_id]
            raw.pop('children', None)
            if parent is not None:
                raw['parent_id'] = output_ids[parent.node_id]
            elif raw.get('parent_id') in used_output_ids:
                raw.pop('parent_id', None)
            output.append(raw)
        return output

    def patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        if not isinstance(patch, PatchHunk):
            raise TypeError(f'patch must be a PatchHunk, got {type(patch).__name__}.')
        if not isinstance(document, WriterDocument):
            raise TypeError(
                f'document must be a WriterDocument, got {type(document).__name__}.')

        handlers = {
            'replace': self._replace_patch_to_operation,
            'insert': self._insert_patch_to_operation,
            'delete': self._delete_patch_to_operation,
            'move': self._move_patch_to_operation,
        }
        return handlers[patch.modify_type](patch, document)

    def _replace_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        '''Convert replace into a Feishu batch-update operation.'''

        block = document.block_by_id(patch.target_node_id)
        if block is None:
            raise ValueError(f'patch target node does not exist: {patch.target_node_id!r}.')
        provider = block.provider_binding.get('provider')
        if provider != self.provider:
            raise ValueError(
                f'patch target provider must be {self.provider!r}, got {provider!r}.')
        block_id = block.provider_binding.get('block_id')
        if not isinstance(block_id, str) or not block_id.strip():
            raise ValueError('patch target does not have a Feishu block_id binding.')
        raw_block = self._raw_payload(block)
        block_type = raw_block.get('block_type')
        if block_type not in _TEXT_BLOCK_TYPES or not block.editable:
            raise ValueError(
                f'Feishu block type {block_type!r} does not support text replacement.')
        if patch.new_text is None:
            raise ValueError('replace patch must provide new_text.')

        replacement = patch.new_text
        if patch.text_range is None:
            if patch.old_text is not None and patch.old_text != block.content:
                raise ValueError('patch old_text does not match the current block content.')
        else:
            start, end = patch.text_range
            if start < 0 or end < start or end > len(block.content):
                raise ValueError('patch text_range is outside the current block content.')
            selected = block.content[start:end]
            if patch.old_text is not None and patch.old_text != selected:
                raise ValueError('patch old_text does not match the selected text range.')
            replacement = f'{block.content[:start]}{replacement}{block.content[end:]}'

        return NativePatchOperation(
            operation='update',
            params={
                'requests': [{
                    'block_id': block_id.strip(),
                    'update_text_elements': {
                        'elements': self._replace_text_elements(
                            raw_block, block.content, replacement),
                    },
                }],
            },
        )

    def _insert_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        '''Reserved for conversion to Feishu create_block parameters.'''
        raise NotImplementedError(
            'Feishu insert patch conversion to create_block is not implemented yet.')

    def _delete_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        '''Reserved for conversion to Feishu delete_block parameters.'''
        raise NotImplementedError(
            'Feishu delete patch conversion to delete_block is not implemented yet.')

    def _move_patch_to_operation(
        self,
        patch: PatchHunk,
        document: WriterDocument,
    ) -> NativePatchOperation:
        '''Reserved for conversion to Feishu move_block parameters.'''
        raise NotImplementedError(
            'Feishu move patch conversion to move_block is not implemented yet.')

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
        numbering: Dict[str, Any] = {}
        if isinstance(block_type, int) and 3 <= block_type <= 11:
            numbering['level'] = block_type - 2
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
            },
            editable=block_type in _TEXT_BLOCK_TYPES,
        )

    def _ir_block_to_raw(self, block: WriterBlock) -> NativeBlock:
        original = self._raw_payload(block)
        raw = deepcopy(original)
        original_type = original.get('block_type')
        block_type = self._block_type_from_ir(block, original_type)

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
        if same_visible_content:
            return raw

        for field in _BLOCK_TYPE_FIELDS.values():
            if field != content_field:
                raw.pop(field, None)
        content_payload = deepcopy(raw.get(content_field) or {})
        if block_type == 22:
            content_payload = {}
        elif block_type in _TEXT_BLOCK_TYPES:
            content_payload['elements'] = self._spans_to_elements(block)
        raw[content_field] = content_payload
        raw['plain_text'] = block.content
        return raw

    def _block_type_from_ir(self, block: WriterBlock, original_type: Any) -> int:
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
            text_run = element.get('text_run')
            if isinstance(text_run, dict):
                text = text_run.get('content')
                if not isinstance(text, str):
                    text = ''
                raw_style = text_run.get('text_element_style') or {}
                styles = [
                    ir_style for feishu_style, ir_style in _STYLE_TO_IR.items()
                    if raw_style.get(feishu_style) is True
                ]
                if isinstance(raw_style.get('link'), dict) and raw_style['link'].get('url'):
                    styles.append('link')
                spans.append(WriterSpan(text=text, style=styles))
                continue

            for element_type, value in element.items():
                text = cls._element_plain_text(value)
                if text:
                    spans.append(WriterSpan(text=text, style=[f'feishu:{element_type}']))
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
    def _spans_to_elements(cls, block: WriterBlock) -> List[Dict[str, Any]]:
        if not block.spans:
            return cls._plain_text_elements(block.content)
        elements: List[Dict[str, Any]] = []
        for span in block.spans:
            unsupported = [style for style in span.style if style.startswith('feishu:')]
            if unsupported:
                raise ValueError(
                    f'cannot reconstruct provider elements from span styles: {unsupported}.')
            raw_style = {
                _STYLE_FROM_IR[style]: True
                for style in span.style
                if style in _STYLE_FROM_IR
            }
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
