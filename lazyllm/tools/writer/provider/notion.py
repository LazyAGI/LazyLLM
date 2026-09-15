from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Tuple

from lazyllm import LOG

from .base import (
    WriterProviderBase,
    WriterProviderCapabilities,
    WriterProviderDocument,
    WriterProviderRevisionError,
    WriterProviderWriteMode,
)
from ..adapter.base import NativePatchOperation, WriterAdapterBase
from ..adapter.notion import NotionWriterAdapter
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchHunk, PatchResult, PatchSet
from ..data_models.task import TargetDocument
from ..data_models.writer_ir import WriterDocument, WriterStage
from ..numbering import build_numbering_view_from_ir, compute_numbering, format_target_number, materialize_ir
from ..tools.revision_tools import apply_patch_to_ir, apply_persisted_patch_hunk
from ..utils import strip_caption_numbering


_NOTION_URL_RE = re.compile(
    r'^https?://(?:[^/]+\.)?notion\.(?:so|site|com)(?:[/:?#]|$)',
    re.IGNORECASE,
)


class NotionWriterProvider(WriterProviderBase):
    '''Orchestrate structured Notion page IO through NotionFS and Writer IR.'''

    provider = 'notion'
    capabilities = WriterProviderCapabilities(
        load=True,
        create=True,
        replace=True,
        append=True,
        patch=True,
        revision_check=True,
        media=True,
    )

    @classmethod
    def matches(cls, locator: str) -> bool:
        value = str(locator or '').strip()
        return value.lower().startswith('notion:/') or bool(_NOTION_URL_RE.match(value))

    def resolve(self, locator: str) -> TargetDocument:
        value = str(locator or '').strip()
        if not value or not self.matches(value):
            raise ValueError(f'Invalid Notion document locator: {locator!r}.')
        return TargetDocument(uri=value, adapter=self.provider)

    def load_document(
        self,
        target: TargetDocument,
        *,
        stage: WriterStage = 'final',
        previous_document: Optional[WriterDocument] = None,
    ) -> dict:
        protocol, real_path, fs, adapter, locator, document_id = \
            self._resolve_document_target(target)
        metadata = self._document_metadata(fs, real_path)
        raw_blocks = fs.get_doc_blocks(real_path, with_descendants=True) or []
        title = str(target.title or metadata.get('title') or '')
        revision = str(metadata.get('last_edited_time') or '') or None
        document = adapter.blocks_to_ir(
            raw_blocks,
            external_document_id=document_id,
            stage=stage,
            title=title,
            uri=locator,
            revision=revision,
        )
        if previous_document is not None:
            document = adapter.merge_refreshed_document(previous_document, document)
        document.metadata.update({
            'block_count': len(raw_blocks),
            'source': target.model_dump(),
            'provider_metadata': metadata,
        })
        resolved_target = target.model_copy(deep=True)
        resolved_target.doc_id = document_id
        resolved_target.uri = str(metadata.get('browser_url') or locator)
        resolved_target.adapter = protocol
        resolved_target.title = title or None
        resolved_target.meta = {
            **resolved_target.meta,
            'internal_uri': str(metadata.get('internal_uri') or f'notion:/~page/{document_id}'),
            'browser_url': str(metadata.get('browser_url') or locator),
            'last_edited_time': revision or '',
        }
        return {
            'representation': 'ir',
            'source_document': document,
            'target_document': resolved_target,
            'provider': protocol,
            'block_count': len(raw_blocks),
        }

    def create_document(self, title: str, parent_uri: str = '') -> TargetDocument:
        title = str(title or '').strip()
        parent_uri = str(parent_uri or '').strip()
        if not title:
            raise ValueError('title is required')
        import lazyllm.tools.fs.client as _fs_client
        parent_locator = parent_uri or f'{self.provider}:/'
        protocol, space_id, real_path = _fs_client.FS._parse(parent_locator)
        if protocol != self.provider:
            raise ValueError(
                f'parent URI protocol {protocol!r} does not match adapter {self.provider!r}.')
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)
        create_document = getattr(fs, 'create_document', None)
        if not callable(create_document):
            raise TypeError(f'{type(fs).__name__} does not support create_document().')
        created = create_document(title, real_path)
        if not isinstance(created, dict):
            raise TypeError('NotionFS.create_document() must return a dict.')
        document_id = str(created.get('document_id') or '').strip()
        browser_url = str(created.get('browser_url') or '').strip()
        internal_uri = str(created.get('internal_uri') or '').strip()
        if not document_id or not (browser_url or internal_uri):
            raise ValueError('NotionFS returned an incomplete created document.')
        return TargetDocument(
            doc_id=document_id,
            uri=browser_url or internal_uri,
            adapter=protocol,
            title=str(created.get('title') or title),
            meta={
                'internal_uri': internal_uri or f'notion:/~page/{document_id}',
                'browser_url': browser_url,
                'parent_uri': parent_uri,
                'last_edited_time': str(created.get('last_edited_time') or ''),
            },
        )

    def _convert_native_document(
        self,
        content: WriterDocument | str,
        *,
        target: TargetDocument | None = None,
        media_assets: MediaAssetLibrary | None = None,
    ) -> WriterProviderDocument:
        source_document = self._writer_document(content, media_assets)
        document = source_document.model_copy(deep=True)
        if target is not None:
            if target.adapter and target.adapter != self.provider:
                raise ValueError(
                    f'target adapter {target.adapter!r} does not match provider {self.provider!r}.')
            document.provider_binding = {
                **document.provider_binding,
                'provider': self.provider,
                'document_id': str(target.doc_id or ''),
                'uri': str(target.uri or ''),
            }
        adapter = self._writer_adapter()
        numbering = compute_numbering(build_numbering_view_from_ir(document))
        document = materialize_ir(document, numbering)
        native_blocks = adapter.ir_to_blocks(document, media_assets=media_assets)
        return WriterProviderDocument(
            provider=self.provider,
            format='notion_blocks',
            content=self._copyable_media_content(native_blocks, media_assets),
            source_document=source_document,
            media_references={
                asset_id: str(asset.uri or asset.local_path or '')
                for asset_id, asset in (media_assets.assets.items() if media_assets else [])
                if asset.uri or asset.local_path
            },
        )

    def apply_patch_to_document(
        self,
        patch_set: PatchSet,
        source_document: WriterDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        if patch_set.target_doc_id != source_document.document_id:
            raise ValueError(
                f'patch target_doc_id {patch_set.target_doc_id!r} does not match '
                f'document_id {source_document.document_id!r}.')
        if not patch_set.hunks \
                and (patch_set.new_title is None or patch_set.new_title == source_document.title):
            raise ValueError('patch contains no document operations.')
        _, real_path, fs, adapter, _, document_id = self._resolve_document_target(
            target, source_document=source_document)
        revised_document, _ = apply_patch_to_ir(
            source_document, patch_set, media_assets=media_assets)
        final_numbering = compute_numbering(
            build_numbering_view_from_ir(revised_document))
        numbered_document = materialize_ir(revised_document, final_numbering)
        baseline = self._document_metadata(fs, real_path)
        current_revision = str(baseline.get('last_edited_time') or '') or None
        if source_document.revision and current_revision != source_document.revision:
            raise WriterProviderRevisionError(
                self.provider, source_document.revision, current_revision,
            )

        persisted = source_document
        applied_hunks: List[str] = []
        pending_rows: dict[str, NativePatchOperation] = {}
        pending_hunk_ids: List[str] = []

        def flush_cells() -> None:
            for operation in pending_rows.values():
                try:
                    self._execute_native_operation(fs, document_id, operation)
                except Exception as exc:
                    raise RuntimeError(
                        f'Notion cell batch failed for row {operation.params["block_id"]!r} '
                        f'and hunks {pending_hunk_ids!r}; earlier rows may have been applied.'
                    ) from exc
            applied_hunks.extend(pending_hunk_ids)
            pending_rows.clear()
            pending_hunk_ids.clear()

        for source_hunk in patch_set.hunks:
            current = persisted.block_by_id(source_hunk.target_node_id)
            is_cell_update = source_hunk.modify_type == 'update' and current is not None \
                and current.type == 'table_cell'
            if not is_cell_update:
                flush_cells()
            hunk = self._materialize_table_caption_hunk(
                source_hunk, final_numbering)
            operation = adapter.patch_to_operation(
                hunk, persisted, media_assets=media_assets)
            if is_cell_update:
                # Each row payload includes earlier edits from this uninterrupted cell run.
                pending_rows[operation.params['block_id']] = operation
                pending_hunk_ids.append(hunk.hunk_id or hunk.target_node_id)
                persisted = self._apply_operation_result_locally(
                    persisted, hunk, operation, {}, media_assets=media_assets)
                continue
            try:
                result = self._execute_native_operation(fs, document_id, operation)
            except Exception as exc:
                hunk_id = hunk.hunk_id or hunk.target_node_id
                LOG.error(
                    f'Notion Writer patch failed: operation={operation.operation} '
                    f'hunk_id={hunk_id} block_id={operation.params.get("block_id")} '
                    f'revision={persisted.revision} error={exc}')
                raise RuntimeError(
                    f'provider {operation.operation} failed for block '
                    f'{operation.params.get("block_id") or "unknown"!r} '
                    f'at revision {persisted.revision!r}: {exc}') from exc
            applied_hunks.append(hunk.hunk_id or hunk.target_node_id)
            persisted = self._apply_operation_result_locally(
                persisted, hunk, operation, result, media_assets=media_assets)

        flush_cells()

        title_updated = patch_set.new_title is not None \
            and patch_set.new_title != source_document.title
        if title_updated:
            update_title = getattr(fs, 'update_page_title', None)
            if not callable(update_title):
                raise TypeError(f'{type(fs).__name__} does not support update_page_title().')
            update_title(document_id, patch_set.new_title)
            persisted = persisted.model_copy(update={'title': patch_set.new_title})

        for sync_hunk in self._numbering_sync_hunks(numbered_document, persisted):
            operation = adapter.patch_to_operation(
                sync_hunk, persisted, media_assets=media_assets)
            try:
                sync_result = self._execute_native_operation(fs, document_id, operation)
            except Exception as exc:
                LOG.error(
                    'Notion numbering sync failed: hunk_id=%s block_id=%s error=%s',
                    sync_hunk.hunk_id,
                    operation.params.get('block_id'),
                    exc,
                )
                raise RuntimeError(
                    f'provider numbering sync failed for block '
                    f'{operation.params.get("block_id") or "unknown"!r}: {exc}'
                ) from exc
            applied_hunks.append(sync_hunk.hunk_id or sync_hunk.target_node_id)
            persisted = self._apply_operation_result_locally(
                persisted, sync_hunk, operation, sync_result,
                media_assets=media_assets)

        refreshed = self._read_persisted_document(
            fs=fs, adapter=adapter, real_path=real_path,
            document_id=document_id, source_document=persisted)
        persisted = adapter.merge_refreshed_document(persisted, refreshed)
        result = PatchResult(
            patch_id=patch_set.patch_id,
            success=True,
            applied_hunks=applied_hunks,
            failed_hunks=[],
            message='Patch written to document.',
            meta={
                'provider': self.provider,
                'external_document_id': document_id,
                'operation_count': len(applied_hunks) + int(title_updated),
                'title_updated': title_updated,
            },
        )
        return {
            'patch_result': result,
            'persisted_document': persisted,
            'provider': self.provider,
            'document_id': document_id,
        }

    @staticmethod
    def _apply_operation_result_locally(
        document: WriterDocument,
        patch: PatchHunk,
        operation: NativePatchOperation,
        operation_result: Any,
        *,
        media_assets: MediaAssetLibrary | None,
    ) -> WriterDocument:
        '''Advance Writer IR without rereading the entire Notion page.'''
        updated = apply_persisted_patch_hunk(document, patch)

        if operation.operation not in {'create', 'move'}:
            NotionWriterAdapter._update_local_caption(updated, patch)
            return updated

        bindings = operation_result.get('node_id_bindings') \
            if isinstance(operation_result, dict) else None
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError(
                f'{operation.operation} operation did not return Notion node ID bindings.')

        native_by_node_id: dict[str, dict[str, Any]] = {}

        def collect_native(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    collect_native(item)
                return
            if not isinstance(value, dict):
                return
            temporary_id = value.get('_temporary_node_id')
            if isinstance(temporary_id, str) and temporary_id:
                native_by_node_id[temporary_id] = value
            for item in value.values():
                collect_native(item)

        collect_native(operation.params.get('blocks'))
        collect_native(operation.params.get('block'))
        caption_operation = operation.params.get('_caption_operation')
        if caption_operation is not None:
            collect_native(caption_operation.params.get('block'))
        relation_map = bindings
        if any(not isinstance(relation_map.get(node_id), str) or not relation_map[node_id]
               for node_id in native_by_node_id):
            raise ValueError('Notion operation returned incomplete block ID relations.')
        physical_ids = [relation_map[node_id] for node_id in native_by_node_id]
        if len(physical_ids) != len(set(physical_ids)):
            raise ValueError('Notion operation returned duplicate block IDs.')
        for node_id, block_id in bindings.items():
            if not isinstance(node_id, str) or not node_id \
                    or not isinstance(block_id, str) or not block_id:
                continue
            block = updated.block_by_id(node_id)
            if block is None:
                if node_id.endswith('::caption'):
                    root = updated.block_by_id(node_id[:-9])
                    if root is not None and root.type == 'table':
                        root.provider_payload['table_caption'] = {
                            'provider_binding': {
                                'provider': 'notion',
                                'block_id': block_id,
                            },
                            'raw_block': deepcopy(native_by_node_id.get(node_id) or {}),
                        }
                    continue
                raise ValueError(
                    f'Notion returned a block relation for unknown node {node_id!r}.')
            block.provider_binding = {
                **block.provider_binding,
                'provider': 'notion',
                'block_id': block_id,
            }
            native = native_by_node_id.get(node_id)
            if native is not None:
                raw = deepcopy(native)
                raw.pop('_temporary_node_id', None)
                raw.pop('_media', None)
                raw['object'] = 'block'
                raw['id'] = block_id
                block.provider_payload = {
                    **block.provider_payload,
                    'raw_block': raw,
                }
        root = updated.block_by_id(patch.target_node_id)
        if root is not None:
            parent_key = 'target_parent_block_id' if operation.operation == 'move' else 'parent_block_id'

            def bind_parents(block: Any, parent_id: str) -> None:
                block_id = block.provider_binding.get('block_id')
                if block_id:
                    block.provider_binding['parent_block_id'] = parent_id
                caption_binding = NotionWriterAdapter._caption_binding(block)
                if caption_binding:
                    caption_binding['parent_block_id'] = parent_id
                for child in block.children:
                    bind_parents(child, block_id or parent_id)

            bind_parents(root, operation.params[parent_key])
        NotionWriterAdapter._update_local_caption(updated, patch)
        return WriterDocument.model_validate(updated.model_dump())

    @staticmethod
    def _materialize_table_caption_hunk(
        hunk: PatchHunk,
        numbering: dict[str, Any],
    ) -> PatchHunk:
        if hunk.block is None:
            return hunk
        hunk = hunk.model_copy(deep=True)
        for item in hunk.block.iter_blocks():
            if item.type != 'table' or not item.content.strip():
                continue
            entry = numbering.get(item.node_id)
            if entry is None:
                continue
            item.content = (
                f'{format_target_number(entry)} '
                f'{strip_caption_numbering(item.content)}'
            ).strip()
            item.spans = []
        return hunk

    @classmethod
    def _numbering_sync_hunks(
        cls,
        numbered_document: WriterDocument,
        persisted_document: WriterDocument,
    ) -> List[PatchHunk]:
        expected_by_id = {
            block.node_id: block for block in numbered_document.iter_blocks()
        }
        hunks = []
        for current in persisted_document.iter_blocks():
            # Notion headings have visible rich text and images/code blocks have
            # captions. Native table blocks do not expose a caption, so table
            # numbering cannot be synchronized without inserting a synthetic
            # paragraph next to the table.
            if current.type not in {'heading', 'image', 'code'}:
                continue
            expected = expected_by_id.get(current.node_id)
            if expected is None or expected.type != current.type:
                continue
            expected_text = cls._expected_numbering_text(expected)
            if cls._native_visible_text(current) == expected_text:
                continue
            meta = {'source': 'system_numbering'}
            if current.type in {'image', 'code'}:
                meta['update_scope'] = 'caption'
            hunks.append(PatchHunk(
                hunk_id=f'{current.type}-numbering-sync-{current.node_id}',
                target_node_id=current.node_id,
                modify_type='update',
                block=expected.model_copy(deep=True),
                meta=meta,
            ))
        return hunks

    @staticmethod
    def _expected_numbering_text(block: Any) -> str:
        if block.type == 'code':
            return str(block.provider_payload.get('numbering_caption') or '')
        return block.content

    @staticmethod
    def _native_visible_text(block: Any) -> str:
        raw = block.provider_payload.get('raw_block') or {}
        block_type = raw.get('type') or raw.get('block_type')
        payload = raw.get(block_type) if isinstance(block_type, str) else None
        if not isinstance(payload, dict):
            return block.content
        rich_text = payload.get(
            'caption' if block.type in {'image', 'code'} else 'rich_text')
        if not isinstance(rich_text, list):
            return block.content
        values: List[str] = []
        for item in rich_text:
            if not isinstance(item, dict):
                continue
            plain_text = item.get('plain_text')
            if isinstance(plain_text, str):
                values.append(plain_text)
                continue
            text = item.get('text') or {}
            if isinstance(text, dict) and isinstance(text.get('content'), str):
                values.append(text['content'])
        return ''.join(values)

    def write_document(
        self,
        converted: WriterProviderDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
        mode: WriterProviderWriteMode = 'replace',
    ) -> dict:
        if converted.provider != self.provider or converted.format != 'notion_blocks':
            raise ValueError('Notion write_document requires converted Notion blocks.')
        source_document = converted.source_document.model_copy(deep=True)
        protocol, real_path, fs, _, locator, document_id = \
            self._resolve_document_target(target, source_document=source_document)
        if source_document.revision is not None:
            current_revision = str(
                self._document_metadata(fs, real_path).get('last_edited_time') or '',
            ) or None
            if current_revision != source_document.revision:
                raise WriterProviderRevisionError(
                    self.provider, source_document.revision, current_revision,
                )
        source_document.provider_binding = {
            **source_document.provider_binding,
            'provider': protocol,
            'document_id': document_id,
            'uri': locator,
        }
        converted_content = self._writer_adapter().materialize_internal_links(
            converted.content, document_uri=locator, document_id=document_id,
        )
        native_blocks = self._writable_media_content(converted_content, media_assets)
        if not isinstance(native_blocks, list):
            raise TypeError('Converted Notion content must be a block list.')
        if source_document.title:
            update_title = getattr(fs, 'update_page_title', None)
            if not callable(update_title):
                raise TypeError(f'{type(fs).__name__} does not support update_page_title().')
            update_title(document_id, source_document.title)
        method_name = 'replace_doc_blocks' if mode == 'replace' else 'write_doc_blocks'
        write_blocks = getattr(fs, method_name, None)
        if not callable(write_blocks):
            raise TypeError(f'{type(fs).__name__} does not support {method_name}().')
        relations: List[dict[str, str]] = []
        if native_blocks:
            written_blocks = write_blocks(
                document_id, native_blocks, block_id_relations=relations)
        warnings: List[str] = []
        if not native_blocks:
            warnings.append('Document has no publishable blocks.')
        persisted_result = {}
        if relations:
            adapter = self._writer_adapter()
            refreshed = adapter.blocks_to_ir(
                written_blocks, external_document_id=document_id, stage='final',
                title=source_document.title or target.title or '', uri=locator,
            )
            refreshed.metadata.update({'block_count': len(written_blocks), 'source': target.model_dump()})
            metadata = self._document_metadata(fs, real_path)
            refreshed.title = str(metadata.get('title') or refreshed.title)
            refreshed.revision = str(metadata.get('last_edited_time') or '') or None
            refreshed.metadata['provider_metadata'] = metadata
            persisted_result = {
                'representation': 'ir',
                'persisted_document': adapter.bind_written_document(source_document, relations, refreshed),
            }
        return {
            **persisted_result,
            'doc_id': document_id,
            'adapter': protocol,
            'locator': locator,
            'block_count': len(native_blocks),
            'warnings': warnings,
        }

    def _resolve_document_target(
        self,
        target: TargetDocument,
        source_document: Optional[WriterDocument] = None,
        *,
        require_page: bool = True,
    ) -> Tuple[str, str, Any, WriterAdapterBase, str, str]:
        locator = self._target_locator(target, source_document)
        if not locator:
            raise ValueError(
                'target_document or source_document provider_binding must provide uri or doc_id.')
        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        if protocol != self.provider:
            raise ValueError(f'Notion provider cannot handle locator protocol {protocol!r}.')
        requested_adapter = target.adapter or (
            source_document.provider_binding.get('provider') if source_document else None)
        if requested_adapter and requested_adapter != protocol:
            raise ValueError(
                f'target adapter {requested_adapter!r} does not match locator protocol {protocol!r}.')
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)
        metadata = self._document_metadata(fs, real_path)
        object_type = str(metadata.get('object_type') or '').lower()
        if require_page and object_type not in {'page'}:
            raise ValueError(f'Notion Writer requires a page target, got {object_type or "unknown"!r}.')
        document_id = str(metadata.get('document_id') or '').strip()
        if not document_id:
            raise ValueError('NotionFS returned an empty document ID.')
        return protocol, real_path, fs, self._writer_adapter(), locator, document_id

    @staticmethod
    def _execute_native_operation(
        fs: Any, document_id: str, operation: NativePatchOperation,
    ) -> Any:
        if operation.operation not in {'create', 'update', 'delete', 'move'}:
            raise NotImplementedError(
                f'Notion provider does not support {operation.operation!r} patches yet.')
        method_name = f'{operation.operation}_block'
        method = getattr(fs, method_name, None)
        if not callable(method):
            raise TypeError(f'{type(fs).__name__} does not support {method_name}().')
        params = dict(operation.params)
        caption_operation = params.pop('_caption_operation', None)
        result = method(document_id=document_id, **params)
        temporary_bindings = result.pop('block_id_relations', None)
        if temporary_bindings is not None:
            result['node_id_bindings'] = {
                relation['temporary_block_id']: relation['block_id']
                for relation in temporary_bindings
                if isinstance(relation, dict)
                and isinstance(relation.get('temporary_block_id'), str)
                and relation.get('temporary_block_id')
                and isinstance(relation.get('block_id'), str)
                and relation.get('block_id')
            }
        if caption_operation is not None:
            try:
                extra = NotionWriterProvider._execute_native_operation(fs, document_id, caption_operation)
            except Exception as exc:
                raise RuntimeError(
                    'Notion operation partially applied: caption operation failed; reload the document.'
                ) from exc
            result = {
                **result,
                'node_id_bindings': {
                    **(result.get('node_id_bindings') or {}),
                    **(extra.get('node_id_bindings') or {}),
                },
            }
        return result

    @classmethod
    def _read_persisted_document(
        cls,
        *,
        fs: Any,
        adapter: WriterAdapterBase,
        real_path: str,
        document_id: str,
        source_document: WriterDocument,
    ) -> WriterDocument:
        metadata = cls._document_metadata(fs, real_path)
        raw_blocks = fs.get_doc_blocks(real_path, with_descendants=True) or []
        title = str(metadata.get('title') or source_document.title)
        revision = str(metadata.get('last_edited_time') or '') or None
        document = adapter.blocks_to_ir(
            raw_blocks,
            external_document_id=document_id,
            stage=source_document.stage,
            title=title,
            uri=str(source_document.provider_binding.get('uri') or ''),
            revision=revision,
        )
        document.metadata = {
            **deepcopy(source_document.metadata),
            'block_count': len(raw_blocks),
            'provider_metadata': metadata,
        }
        return document

    @classmethod
    def _target_locator(
        cls,
        target: TargetDocument,
        source_document: Optional[WriterDocument] = None,
    ) -> str:
        if target.uri:
            return target.uri
        if source_document:
            source_uri = source_document.provider_binding.get('uri')
            if isinstance(source_uri, str) and source_uri:
                return source_uri
        document_id = target.doc_id
        if not document_id and source_document:
            document_id = source_document.provider_binding.get('document_id')
        return f'{cls.provider}:/~page/{document_id}' if document_id else ''

    def _writer_adapter(self) -> WriterAdapterBase:
        configured = self.adapters.get(self.provider, NotionWriterAdapter)
        adapter = configured() if isinstance(configured, type) else configured
        if not isinstance(adapter, WriterAdapterBase):
            raise TypeError(
                f'Writer adapter for {self.provider!r} must inherit WriterAdapterBase, '
                f'got {type(adapter).__name__}.'
            )
        return adapter

    @staticmethod
    def _document_metadata(fs: Any, path: str) -> dict:
        method = getattr(fs, 'get_document_metadata', None)
        if not callable(method):
            raise TypeError(f'{type(fs).__name__} does not support get_document_metadata().')
        metadata = method(path)
        if not isinstance(metadata, dict):
            raise TypeError('NotionFS.get_document_metadata() must return a dict.')
        return metadata

    @staticmethod
    def _validate_available_images(
        document: WriterDocument,
        media_assets: Optional[MediaAssetLibrary],
        adapter: WriterAdapterBase,
    ) -> None:
        for block in document.iter_blocks():
            if block.type != 'image':
                continue
            references = [
                ref.get('id') for ref in block.references
                if ref.get('type') == 'media_asset' and ref.get('id')
            ]
            reusable = getattr(adapter, 'has_reusable_image_payload', None)
            if not references and callable(reusable) and reusable(block):
                continue
            if len(references) != 1:
                raise ValueError(
                    f'Image block {block.node_id!r} requires one media_asset reference '
                    'or a reusable Notion file payload.'
                )
            asset = media_assets.assets.get(references[0]) if media_assets else None
            if asset is None or not asset.local_path or not Path(asset.local_path).is_file():
                raise ValueError(f'Image block {block.node_id!r} media is unavailable.')


__all__ = ['NotionWriterProvider']
