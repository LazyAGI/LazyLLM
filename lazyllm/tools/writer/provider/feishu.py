from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from lazyllm import LOG

from .base import (
    WriterProviderBase,
    WriterProviderCapabilities,
    WriterProviderDocument,
    WriterProviderRevisionError,
    WriterProviderWriteMode,
)
from ..adapter.base import NativePatchOperation, WriterAdapterBase
from ..adapter.feishu import FeishuWriterAdapter, feishu_block_url
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchHunk, PatchResult, PatchSet
from ..data_models.task import InputResource, TargetDocument
from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterStage
from ..numbering import (
    build_numbering_view_from_ir,
    compute_numbering,
    format_target_number,
    materialize_ir,
)
from ..tools.revision_tools import apply_patch_to_ir, apply_persisted_patch_hunk
from ..utils import strip_caption_numbering, strip_heading_numbering


_FEISHU_URL_RE = re.compile(
    r'^https?://[^/]*(?:feishu\.(?:cn|com)|larksuite\.com)/',
    re.IGNORECASE,
)


class FeishuWriterProvider(WriterProviderBase):
    '''Orchestrate Feishu document IO and structured Writer conversion.'''

    provider = 'feishu'
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
        return value.lower().startswith('feishu:/') or bool(_FEISHU_URL_RE.match(value))

    def resolve(self, locator: str) -> TargetDocument:
        value = str(locator or '').strip()
        if not value or not self.matches(value):
            raise ValueError(f'Invalid Feishu document locator: {locator!r}.')
        return TargetDocument(uri=value, adapter=self.provider)

    def load_document(
        self,
        target: TargetDocument,
        *,
        stage: WriterStage = 'final',
        previous_document: Optional[WriterDocument] = None,
    ) -> dict:
        protocol, real_path, fs, adapter, locator, external_document_id = \
            self._resolve_document_target(target)
        metadata = self._document_metadata(fs, real_path)
        if not hasattr(fs, 'get_doc_blocks'):
            raise TypeError(f'{type(fs).__name__} does not support structured document reads.')
        raw_blocks = fs.get_doc_blocks(real_path, with_descendants=True) or []
        document = adapter.blocks_to_ir(
            raw_blocks,
            external_document_id=external_document_id,
            stage=stage,
            title=target.title or str(metadata.get('title') or ''),
            uri=locator,
            revision=str(metadata['revision_id']),
        )
        if previous_document is not None:
            document = adapter.merge_refreshed_document(previous_document, document)
        document.metadata.update({
            'block_count': len(raw_blocks),
            'source': target.model_dump(),
            'provider_metadata': metadata,
        })
        resolved_target = target.model_copy(deep=True)
        resolved_target.doc_id = external_document_id
        resolved_target.uri = locator
        resolved_target.adapter = protocol
        resolved_target.title = document.title or target.title
        resolved_target.meta = {
            **resolved_target.meta,
            'revision_id': metadata['revision_id'],
        }
        return {
            'representation': 'ir',
            'source_document': document,
            'target_document': resolved_target,
            'provider': protocol,
            'block_count': len(raw_blocks),
        }

    def create_document(self, title: str, parent_uri: str = '') -> TargetDocument:
        title = (title or '').strip()
        if not title:
            raise ValueError('title is required')

        import lazyllm.tools.fs.client as _fs_client
        parent_locator = (parent_uri or '').strip() or f'{self.provider}:/'
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
            raise TypeError('Document provider create_document() must return a dict.')

        document_id = str(created.get('document_id') or '').strip()
        created_path = str(created.get('path') or '').strip()
        if not document_id or not created_path:
            raise ValueError('Document provider returned an incomplete created document.')
        effective_space_id = str(created.get('space_id') or '').strip()
        internal_uri = (
            f'{protocol}@{effective_space_id}:{created_path}'
            if effective_space_id else f'{protocol}:{created_path}'
        )
        browser_url = str(created.get('browser_url') or '').strip()
        return TargetDocument(
            doc_id=document_id,
            uri=browser_url or internal_uri,
            adapter=protocol,
            title=str(created.get('title') or title),
            meta={
                'internal_uri': internal_uri,
                'browser_url': browser_url,
                'container': created.get('container') or '',
                'parent_uri': (parent_uri or '').strip(),
                'node_token': created.get('node_token') or '',
                'space_id': effective_space_id,
            },
        )

    def document_image_resources(
        self,
        document: WriterDocument,
    ) -> tuple[list[InputResource], list[str]]:
        locator = str(
            document.provider_binding.get('uri')
            or ((document.metadata.get('source') or {}).get('uri') if isinstance(
                document.metadata.get('source'), dict) else '')
            or f'feishu:/~docx/{document.provider_binding.get("document_id") or ""}'
        ).strip()
        resources: list[InputResource] = []
        warnings: list[str] = []
        for block in (item for item in document.iter_blocks() if item.type == 'image'):
            raw = block.provider_payload.get('raw_block') or {}
            image = raw.get('image') if isinstance(raw, dict) else None
            token = str((image or {}).get('token') or '').strip()
            block_id = str(block.provider_binding.get('block_id') or block.node_id)
            if not token:
                warnings.append(f'Feishu image block {block_id!r} has no media token.')
                continue
            resources.append(InputResource(
                resource_id=f'feishu-image-{block_id}',
                resource_type='image',
                uri=f'{locator}#image={block_id}',
                title=block.content or f'Feishu image {block_id}',
                summary=block.content or None,
                meta={
                    'provider': self.provider,
                    'provider_block_id': block_id,
                    'source_type': 'input_resource',
                    'origin': 'source_document',
                    'caption': block.content or None,
                },
            ))
        return resources, warnings

    def download_document_image(
        self,
        document: WriterDocument,
        resource: InputResource,
    ) -> bytes | None:
        block_id = str(resource.meta.get('provider_block_id') or '').strip()
        block = next((
            item for item in document.iter_blocks()
            if str(item.provider_binding.get('block_id') or item.node_id) == block_id
        ), None)
        raw = block.provider_payload.get('raw_block') if block else None
        image = raw.get('image') if isinstance(raw, dict) else None
        token = str((image or {}).get('token') or '').strip()
        if not token:
            raise ValueError(f'Feishu image block {block_id!r} has no media token.')
        locator = str(
            document.provider_binding.get('uri')
            or ((document.metadata.get('source') or {}).get('uri') if isinstance(
                document.metadata.get('source'), dict) else '')
            or f'feishu:/~docx/{document.provider_binding.get("document_id") or ""}'
        ).strip()
        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)
        download_media = getattr(fs, 'download_media', None)
        if not callable(download_media):
            raise TypeError(f'{type(fs).__name__} does not support Feishu media downloads.')
        return download_media(token)

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
        numbering = compute_numbering(build_numbering_view_from_ir(document))
        document = materialize_ir(document, numbering)
        native_blocks = self._writer_adapter().ir_to_blocks(
            document, media_assets=media_assets,
        )
        return WriterProviderDocument(
            provider=self.provider,
            format='feishu_blocks',
            content=self._copyable_media_content(native_blocks, media_assets),
            source_document=source_document,
            media_references={
                asset_id: str(asset.uri or asset.local_path or '')
                for asset_id, asset in (media_assets.assets.items() if media_assets else [])
                if asset.uri or asset.local_path
            },
        )

    def write_document(
        self,
        converted: WriterProviderDocument,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
        mode: WriterProviderWriteMode = 'replace',
    ) -> dict:
        if converted.provider != self.provider or converted.format != 'feishu_blocks':
            raise ValueError('Feishu write_document requires converted Feishu blocks.')
        source_document = converted.source_document.model_copy(deep=True)
        protocol, real_path, fs, _, locator, document_id = \
            self._resolve_document_target(target, source_document=source_document)
        if source_document.revision is not None:
            current_revision = str(self._document_metadata(fs, real_path).get('revision_id'))
            if current_revision != source_document.revision:
                raise WriterProviderRevisionError(
                    self.provider, source_document.revision, current_revision)
        source_document.provider_binding = {
            **source_document.provider_binding,
            'provider': protocol,
            'document_id': document_id,
            'uri': locator,
        }
        converted_content = self._writer_adapter().materialize_internal_links(
            converted.content, document_uri=locator, document_id=document_id,
        )
        warnings: List[str] = []
        method_name = 'replace_doc_blocks' if mode == 'replace' else 'write_doc_blocks'
        write_blocks = getattr(fs, method_name, None)
        if not callable(write_blocks):
            raise TypeError(f'{type(fs).__name__} does not support {method_name}().')
        native_blocks = self._writable_media_content(converted_content, media_assets)
        if not isinstance(native_blocks, list):
            raise TypeError('Converted Feishu content must be a block list.')
        if source_document.title:
            title_result = self._update_document_title(
                fs, document_id, source_document.title, source_document.revision,
            )
            source_document = self._with_operation_revision(source_document, title_result)
        relations: List[dict[str, str]] = []
        if not native_blocks:
            warnings.append('Document has no publishable blocks.')
        else:
            try:
                revision_id = int(source_document.revision) if source_document.revision is not None else -1
            except (TypeError, ValueError):
                revision_id = -1
            written_blocks = write_blocks(
                document_id, native_blocks, block_id_relations=relations,
                document_revision_id=revision_id)
        persisted_result = {}
        if relations:
            adapter = self._writer_adapter()
            metadata = self._document_metadata(fs, real_path)
            refreshed = adapter.blocks_to_ir(
                written_blocks, external_document_id=document_id, stage='final',
                title=source_document.title or target.title or '', uri=locator,
                revision=str(metadata['revision_id']),
            )
            refreshed.metadata.update({
                'block_count': len(written_blocks),
                'source': target.model_dump(),
                'provider_metadata': metadata,
            })
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

    def apply_patch_to_document(  # noqa: C901
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

        protocol, real_path, fs, adapter, locator, document_id = \
            self._resolve_document_target(target, source_document=source_document)
        metadata = self._document_metadata(fs, real_path)
        current_revision = str(metadata['revision_id'])
        if source_document.revision is not None and current_revision != source_document.revision:
            raise WriterProviderRevisionError(
                self.provider, source_document.revision, current_revision)
        revised_document, _ = apply_patch_to_ir(
            source_document, patch_set, media_assets=media_assets)
        final_numbering = compute_numbering(build_numbering_view_from_ir(revised_document))

        applied_hunks: List[str] = []
        persisted_document = source_document
        expected_title = (
            patch_set.new_title if patch_set.new_title is not None else source_document.title)
        title_updated = (
            patch_set.new_title is not None and patch_set.new_title != source_document.title)
        normalized_fields: Dict[str, List[str]] = {}
        pending_cells: Dict[str, Dict[str, Any]] = {}
        pending_hunk_ids: List[str] = []

        def flush_cells() -> None:
            nonlocal persisted_document
            requests = list(pending_cells.values())
            # Docx batch_update accepts at most 200 distinct block IDs per call.
            for offset in range(0, len(requests), 200):
                operation = NativePatchOperation('update', {'requests': requests[offset:offset + 200]})
                try:
                    result = self._execute_native_operation(
                        fs, document_id, operation, persisted_document.revision)
                except Exception as exc:
                    raise RuntimeError(
                        f'Feishu cell batch failed for hunks {pending_hunk_ids!r}; '
                        'earlier batches may have been applied.'
                    ) from exc
                persisted_document = self._with_operation_revision(persisted_document, result)
            applied_hunks.extend(pending_hunk_ids)
            pending_cells.clear()
            pending_hunk_ids.clear()

        for source_hunk in patch_set.hunks:
            current = persisted_document.block_by_id(source_hunk.target_node_id)
            is_cell_update = source_hunk.modify_type == 'update' and current is not None \
                and current.type == 'table_cell'
            if not is_cell_update:
                flush_cells()
            block_id_by_node_id = {
                block.node_id: block.provider_binding.get('block_id')
                for block in persisted_document.iter_blocks()
            }
            hunk = self._materialize_hunk_feishu_links(
                source_hunk,
                block_id_by_node_id=block_id_by_node_id,
                numbering=final_numbering,
                document_id=document_id,
                document_uri=locator,
            )
            local_hunk = self._materialize_hunk_feishu_links(
                source_hunk,
                block_id_by_node_id={},
                numbering=final_numbering,
                document_id=document_id,
                document_uri=locator,
            )
            operation = adapter.patch_to_operation(
                hunk, persisted_document, media_assets=media_assets)
            if is_cell_update:
                for request in operation.params['requests']:
                    pending_cells[request['block_id']] = request
                pending_hunk_ids.append(hunk.hunk_id or hunk.target_node_id)
                persisted_document = self._apply_operation_result_locally(
                    persisted_document, local_hunk, operation, {}, media_assets=media_assets)
                continue
            try:
                operation_result = self._execute_native_operation(
                    fs, document_id, operation, persisted_document.revision)
            except Exception as exc:
                block_id = operation.params.get('block_id')
                if block_id is None:
                    requests = operation.params.get('requests')
                    if isinstance(requests, list) and requests and isinstance(requests[0], dict):
                        block_id = requests[0].get('block_id')
                hunk_id = hunk.hunk_id or hunk.target_node_id
                LOG.error(
                    'Writer provider patch failed: operation=%s hunk_id=%s block_id=%s '
                    'revision=%s error=%s',
                    operation.operation,
                    hunk_id,
                    block_id,
                    persisted_document.revision,
                    exc,
                )
                raise RuntimeError(
                    f'provider {operation.operation} failed for block {block_id or "unknown"!r} '
                    f'at revision {persisted_document.revision!r}: {exc}'
                ) from exc
            if isinstance(operation_result, dict) \
                    and isinstance(operation_result.get('normalized_fields'), list):
                normalized_fields[hunk.hunk_id or hunk.target_node_id] = \
                    operation_result['normalized_fields']
            applied_hunks.append(hunk.hunk_id or hunk.target_node_id)
            persisted_document = self._apply_operation_result_locally(
                persisted_document, local_hunk, operation, operation_result,
                media_assets=media_assets,
            )

        flush_cells()

        if title_updated:
            title_result = self._update_document_title(
                fs, document_id, expected_title, persisted_document.revision)
            persisted_document = self._with_operation_revision(
                persisted_document.model_copy(update={'title': expected_title}),
                title_result,
            )

        for heading in revised_document.iter_blocks():
            if heading.type != 'heading':
                continue
            entry = final_numbering.get(heading.node_id)
            if entry is None:
                continue
            expected = (
                f'{format_target_number(entry)} '
                f'{strip_heading_numbering(heading.content)}'
            ).strip()
            current = persisted_document.block_by_id(heading.node_id)
            if current is None or current.content == expected:
                continue
            sync_hunk = PatchHunk(
                hunk_id=f'heading-sync-{heading.node_id}',
                target_node_id=heading.node_id,
                modify_type='update',
                block=WriterBlock(
                    node_id=heading.node_id,
                    type='heading',
                    content=expected,
                    stage='draft',
                    numbering={'level': current.numbering.get('level', 1)},
                ),
            )
            block_id_by_node_id = {
                block.node_id: block.provider_binding.get('block_id')
                for block in persisted_document.iter_blocks()
            }
            sync_hunk = self._materialize_hunk_feishu_links(
                sync_hunk,
                block_id_by_node_id=block_id_by_node_id,
                numbering=final_numbering,
                document_id=document_id,
                document_uri=locator,
            )
            operation = adapter.patch_to_operation(
                sync_hunk, persisted_document, media_assets=media_assets)
            operation_result = self._execute_native_operation(
                fs, document_id, operation, persisted_document.revision)
            applied_hunks.append(sync_hunk.hunk_id)
            persisted_document = self._apply_operation_result_locally(
                persisted_document, sync_hunk, operation, operation_result,
                media_assets=media_assets,
            )

        refreshed = self._read_persisted_document(
            fs=fs,
            adapter=adapter,
            real_path=real_path,
            locator=locator,
            document_id=document_id,
            source_document=persisted_document,
        )
        persisted_document = adapter.merge_refreshed_document(
            persisted_document, refreshed)

        patch_result = PatchResult(
            patch_id=patch_set.patch_id,
            success=True,
            applied_hunks=applied_hunks,
            failed_hunks=[],
            message='Patch written to document.',
            meta={
                'provider': protocol,
                'external_document_id': document_id,
                'operation_count': len(applied_hunks) + int(title_updated),
                'title_updated': title_updated,
                'normalized_fields': normalized_fields,
            },
        )
        return {
            'patch_result': patch_result,
            'persisted_document': persisted_document,
            'provider': protocol,
            'document_id': document_id,
        }

    @staticmethod
    def _with_operation_revision(
        document: WriterDocument,
        operation_result: Any,
    ) -> WriterDocument:
        revision = operation_result.get('document_revision_id') \
            if isinstance(operation_result, dict) else None
        if revision is None or isinstance(revision, bool):
            return document
        return document.model_copy(update={'revision': str(revision)})

    @classmethod
    def _apply_operation_result_locally(
        cls,
        document: WriterDocument,
        patch: PatchHunk,
        operation: NativePatchOperation,
        operation_result: Any,
        *,
        media_assets: MediaAssetLibrary | None,
    ) -> WriterDocument:
        '''Advance Writer IR and provider bindings without a full Feishu reread.'''
        updated = apply_persisted_patch_hunk(document, patch)

        updated = cls._with_operation_revision(updated, operation_result)

        if operation.operation == 'create':
            cls._apply_created_blocks(updated, patch, operation, operation_result)
        elif operation.operation in {'move', 'replace'}:
            cls._apply_remapped_blocks(document, updated, patch, operation, operation_result)
        FeishuWriterAdapter._update_local_caption(updated, patch)
        return WriterDocument.model_validate(updated.model_dump())

    @staticmethod
    def _bind_local_feishu_block(
        block: WriterBlock,
        block_id: str,
        native: Any,
        relation_map: Dict[str, str],
        native_by_temporary_id: Dict[str, dict[str, Any]],
    ) -> None:
        block.provider_binding = {
            **block.provider_binding,
            'provider': 'feishu',
            'block_id': block_id,
        }
        if not isinstance(native, dict):
            return
        raw = deepcopy(native)
        raw.pop('_media', None)
        raw['block_id'] = block_id
        if isinstance(raw.get('children'), list):
            raw['children'] = [relation_map.get(child_id, child_id)
                               for child_id in raw['children']]
        block.provider_payload = {
            **block.provider_payload,
            'raw_block': raw,
        }
        if block.type == 'table_cell':
            block.provider_payload['table_content_blocks'] = [
                {
                    **deepcopy(native_by_temporary_id[child_id]),
                    'block_id': relation_map.get(child_id, child_id),
                }
                for child_id in native.get('children') or []
                if child_id in native_by_temporary_id
            ]

    def _resolve_document_target(
        self,
        target: TargetDocument,
        source_document: Optional[WriterDocument] = None,
    ) -> Tuple[str, str, Any, WriterAdapterBase, str, str]:
        locator = self._target_locator(target, source_document)
        if not locator:
            raise ValueError(
                'target_document or source_document provider_binding must provide uri or doc_id.')

        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        requested_adapter = target.adapter or (
            source_document.provider_binding.get('provider') if source_document else None)
        if protocol != self.provider:
            raise ValueError(
                f'Feishu provider cannot handle locator protocol {protocol!r}.')
        if requested_adapter and requested_adapter != protocol:
            raise ValueError(
                f'target adapter {requested_adapter!r} does not match locator protocol {protocol!r}.')
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)
        get_document_id = getattr(fs, 'get_document_id', None)
        if not callable(get_document_id):
            raise TypeError(f'{type(fs).__name__} does not support get_document_id().')
        document_id = get_document_id(real_path)
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError('Document provider returned an empty document ID.')
        return (
            protocol,
            real_path,
            fs,
            self._writer_adapter(),
            locator,
            document_id.strip(),
        )

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
        provider = target.adapter or (
            source_document.provider_binding.get('provider') if source_document else None)
        if not document_id and source_document:
            document_id = source_document.provider_binding.get('document_id')
        if document_id and provider == cls.provider:
            return f'{cls.provider}:/~docx/{document_id}'
        return str(document_id or '')

    def _writer_adapter(self) -> WriterAdapterBase:
        configured = self.adapters.get(self.provider, FeishuWriterAdapter)
        adapter = configured() if isinstance(configured, type) else configured
        if not isinstance(adapter, WriterAdapterBase):
            raise TypeError(
                f'Writer adapter for {self.provider!r} must inherit WriterAdapterBase, '
                f'got {type(adapter).__name__}.')
        return adapter

    @staticmethod
    def _document_metadata(fs: Any, path: str) -> Dict[str, Any]:
        method = getattr(fs, 'get_document_metadata', None)
        if not callable(method):
            raise TypeError(f'{type(fs).__name__} does not support get_document_metadata().')
        metadata = method(path)
        if not isinstance(metadata, dict):
            raise TypeError('FeishuFS.get_document_metadata() must return a dict.')
        revision_id = metadata.get('revision_id')
        if not isinstance(revision_id, int) or isinstance(revision_id, bool):
            raise ValueError('Feishu document metadata must contain an integer revision_id.')
        return metadata

    @staticmethod
    def _validate_available_images(
        document: WriterDocument,
        media_assets: Optional[MediaAssetLibrary],
    ) -> None:
        for block in document.iter_blocks():
            if block.type != 'image':
                continue
            references = [
                ref.get('id') for ref in block.references
                if ref.get('type') == 'media_asset' and ref.get('id')
            ]
            if len(references) != 1:
                raise ValueError(
                    f'Image block {block.node_id!r} requires exactly one media_asset reference.')
            asset = media_assets.assets.get(references[0]) if media_assets else None
            if asset is None or not asset.local_path or not Path(asset.local_path).is_file():
                raise ValueError(f'Image block {block.node_id!r} media is unavailable.')

    @staticmethod
    def _materialize_hunk_feishu_links(
        hunk: PatchHunk,
        *,
        block_id_by_node_id: Dict[str, Any],
        numbering: Dict[str, Any],
        document_id: str,
        document_uri: str,
    ) -> PatchHunk:
        if hunk.block is None:
            return hunk
        hunk = hunk.model_copy(deep=True)
        for item in hunk.block.iter_blocks():
            if item.type == 'heading':
                entry = numbering.get(item.node_id)
                if entry is not None:
                    item.content = (
                        f'{format_target_number(entry)} '
                        f'{strip_heading_numbering(item.content)}'
                    ).strip()
                    item.spans = []
            elif item.type == 'table' and item.content.strip():
                entry = numbering.get(item.node_id)
                if entry is not None:
                    item.content = (
                        f'{format_target_number(entry)} '
                        f'{strip_caption_numbering(item.content)}'
                    ).strip()
                    item.spans = []
            for span in item.spans:
                link = span.style.get('link')
                if not isinstance(link, dict) or link.get('type') != 'internal_ref':
                    continue
                target_id = link.get('target_node_id')
                target_block_id = block_id_by_node_id.get(target_id)
                if not target_block_id:
                    continue
                span.style['link'] = {
                    'url': feishu_block_url(document_uri, document_id, target_block_id),
                }
        return hunk

    @staticmethod
    def _update_document_title(
        fs: Any,
        document_id: str,
        title: str,
        revision: Optional[str],
    ) -> Any:
        update_title = getattr(fs, 'update_document_title', None)
        if not callable(update_title):
            raise TypeError(f'{type(fs).__name__} does not support document title updates.')
        try:
            revision_id = int(revision) if revision is not None else -1
        except (TypeError, ValueError):
            revision_id = -1
        return update_title(document_id, title, document_revision_id=revision_id)

    @staticmethod
    def _execute_native_operation(
        fs: Any,
        document_id: str,
        operation: NativePatchOperation,
        revision: Optional[str],
    ) -> Any:
        method_name = f'{operation.operation}_block'
        method = getattr(fs, method_name, None)
        if not callable(method):
            raise TypeError(f'{type(fs).__name__} does not support {method_name}().')
        params = dict(operation.params)
        caption_operation = params.pop('_caption_operation', None)
        params.setdefault('document_id', document_id)
        if operation.operation in {'create', 'update', 'replace', 'delete', 'move'} \
                and 'document_revision_id' not in params:
            try:
                params['document_revision_id'] = int(revision) if revision is not None else -1
            except (TypeError, ValueError):
                params['document_revision_id'] = -1
        result = method(**params)
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
            next_revision = result.get('document_revision_id', revision)
            try:
                extra = FeishuWriterProvider._execute_native_operation(fs, document_id, caption_operation, next_revision)
            except Exception as exc:
                raise RuntimeError(
                    'Feishu operation partially applied: caption operation failed; reload the document.'
                ) from exc
            result = {
                **result, **extra,
                'node_id_bindings': {
                    **(result.get('node_id_bindings') or {}),
                    **(extra.get('node_id_bindings') or {}),
                },
                'provider_id_remap': {
                    **(result.get('provider_id_remap') or {}),
                    **(extra.get('provider_id_remap') or {}),
                },
            }
        return result

    @staticmethod
    def _read_persisted_document(
        *,
        fs: Any,
        adapter: WriterAdapterBase,
        real_path: str,
        locator: str,
        document_id: str,
        source_document: WriterDocument,
    ) -> WriterDocument:
        if not hasattr(fs, 'get_doc_blocks'):
            raise TypeError(f'{type(fs).__name__} does not support structured document reads.')
        latest_blocks = fs.get_doc_blocks(real_path, with_descendants=True) or []
        metadata = FeishuWriterProvider._document_metadata(fs, real_path)
        document = adapter.blocks_to_ir(
            latest_blocks,
            external_document_id=document_id,
            stage=source_document.stage,
            title=source_document.title,
            uri=locator,
            revision=str(metadata['revision_id']),
        )
        document.metadata = {
            **deepcopy(source_document.metadata),
            'block_count': len(latest_blocks),
            'source': source_document.metadata.get('source', {}),
            'provider_metadata': metadata,
        }
        return document

    @classmethod
    def _apply_created_blocks(cls, updated, patch, operation, operation_result) -> None:
        bindings = operation_result.get('node_id_bindings') \
            if isinstance(operation_result, dict) else None
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError('create operation did not return Feishu node ID bindings.')
        relation_map = bindings
        native_blocks = operation.params.get('blocks')
        native_by_temporary_id = {
            block.get('block_id'): block
            for block in native_blocks
            if isinstance(native_blocks, list) and isinstance(block, dict)
            and isinstance(block.get('block_id'), str)
        } if isinstance(native_blocks, list) else {}
        if any(not relation_map.get(node_id) for node_id in native_by_temporary_id):
            raise ValueError('Feishu create returned incomplete block ID relations.')
        if len(set(relation_map.values())) != len(relation_map):
            raise ValueError('Feishu create returned duplicate block IDs.')
        cls._bind_created_nodes(updated, relation_map, native_by_temporary_id)
        for temporary_id, native in native_by_temporary_id.items():
            for child_id in native.get('children') or []:
                child = updated.block_by_id(child_id)
                if child is not None:
                    child.provider_binding['parent_block_id'] = relation_map[temporary_id]
        root = updated.block_by_id(patch.target_node_id)
        if root is not None:
            root.provider_binding['parent_block_id'] = operation.params['parent_block_id']

    @classmethod
    def _apply_remapped_blocks(cls, document, updated, patch, operation, operation_result) -> None:
        previous_by_block_id = {
            block.provider_binding.get('block_id'): block.node_id
            for block in document.iter_blocks()
            if isinstance(block.provider_binding.get('block_id'), str)
        }
        provider_id_remap = operation_result.get('provider_id_remap') \
            if isinstance(operation_result, dict) else None
        if not isinstance(provider_id_remap, dict) or not provider_id_remap:
            raise ValueError(
                f'{operation.operation} operation did not return Feishu provider ID remap.')
        # Native cell text blocks are folded into payloads rather than IR children.
        physical_ids = set(previous_by_block_id)
        for item in document.iter_blocks():
            physical_ids.update(
                raw['block_id'] for raw in item.provider_payload.get('table_content_blocks', [])
                if isinstance(raw, dict) and isinstance(raw.get('block_id'), str)
            )
            caption_id = FeishuWriterAdapter._caption_binding(item).get('block_id')
            if caption_id:
                physical_ids.add(caption_id)
        if any(source_id not in physical_ids or not isinstance(created_id, str)
               for source_id, created_id in provider_id_remap.items()):
            raise ValueError(f'{operation.operation} provider ID remap does not match local IR.')
        previous_root = document.block_by_id(patch.target_node_id)
        required_ids = {
            item.provider_binding['block_id'] for item in previous_root.iter_blocks()
            if item.provider_binding.get('block_id')
        }
        if not required_ids.issubset(provider_id_remap) \
                or any(not value for value in provider_id_remap.values()):
            raise ValueError(f'{operation.operation} returned an incomplete Feishu provider ID remap.')
        if len(set(provider_id_remap.values())) != len(provider_id_remap):
            raise ValueError(f'{operation.operation} returned duplicate Feishu block IDs.')

        for block in updated.iter_blocks():
            block.provider_binding = cls._remap_provider_ids(block.provider_binding, provider_id_remap)
            block.provider_payload = cls._remap_provider_ids(block.provider_payload, provider_id_remap)
        cls._bind_replacement_block(updated, patch, operation, provider_id_remap)
        root = updated.block_by_id(patch.target_node_id)
        if root is not None:
            parent_key = 'target_parent_block_id' \
                if operation.operation == 'move' else 'parent_block_id'
            root.provider_binding['parent_block_id'] = operation.params[parent_key]
            caption_binding = FeishuWriterAdapter._caption_binding(root)
            if caption_binding:
                caption_binding['parent_block_id'] = operation.params[parent_key]

    @classmethod
    def _bind_created_nodes(cls, updated, relation_map, native_by_temporary_id) -> None:
        for node_id, block_id in relation_map.items():
            block = updated.block_by_id(node_id)
            if block is None:
                if node_id.endswith('-caption'):
                    root = updated.block_by_id(node_id[:-8])
                    if root is not None and root.type == 'table':
                        root.provider_payload['table_caption'] = {
                            'provider_binding': {
                                'provider': 'feishu',
                                'block_id': block_id,
                            },
                            'raw_block': deepcopy(
                                native_by_temporary_id.get(node_id) or {}),
                        }
                    continue
                if node_id.endswith(('::text', '_text')) or '-covered-' in node_id:
                    continue
                raise ValueError(
                    f'Feishu returned a block relation for unknown node {node_id!r}.')
            cls._bind_local_feishu_block(
                block, block_id, native_by_temporary_id.get(node_id), relation_map,
                native_by_temporary_id)

    @staticmethod
    def _remap_provider_ids(value: Any, provider_id_remap, key: str = '') -> Any:
        if isinstance(value, dict):
            return {field: FeishuWriterProvider._remap_provider_ids(item, provider_id_remap, field)
                    for field, item in value.items()}
        if isinstance(value, list):
            return [FeishuWriterProvider._remap_provider_ids(item, provider_id_remap, key) for item in value]
        if isinstance(value, str) and key in {
            'block_id', 'parent_block_id', 'parent_id', 'children', 'cells',
        }:
            return provider_id_remap.get(value, value)
        return value

    @classmethod
    def _bind_replacement_block(cls, updated, patch, operation, provider_id_remap) -> None:
        if operation.operation == 'replace':
            root = updated.block_by_id(patch.target_node_id)
            if root is not None:
                replacement = deepcopy(operation.params.get('replacement_block') or {})
                children = (root.provider_payload.get('raw_block') or {}).get('children')
                if children:
                    replacement['children'] = deepcopy(children)
                cls._bind_local_feishu_block(
                    root, root.provider_binding['block_id'],
                    replacement, provider_id_remap, {})


__all__ = ['FeishuWriterProvider']
