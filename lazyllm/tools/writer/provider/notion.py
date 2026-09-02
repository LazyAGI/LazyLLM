from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Tuple

from lazyllm import LOG

from .base import WriterProviderBase
from ..adapter.base import NativePatchOperation, WriterAdapterBase
from ..adapter.notion import NotionWriterAdapter
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchHunk, PatchResult, PatchSet
from ..data_models.task import TargetDocument
from ..data_models.writer_ir import WriterDocument, WriterStage
from ..numbering import build_numbering_view_from_ir, compute_numbering, materialize_ir
from ..tools.revision_tools import apply_patch_to_ir
from ..utils import parse_document_markdown


_NOTION_URL_RE = re.compile(
    r'^https?://(?:[^/]+\.)?notion\.(?:so|site|com)(?:[/:?#]|$)',
    re.IGNORECASE,
)


class NotionWriterProvider(WriterProviderBase):
    '''Orchestrate structured Notion page IO through NotionFS and Writer IR.'''

    provider = 'notion'

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

    def replace_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        return self._write_document(content, target, media_assets=media_assets, mode='replace')

    def append_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        return self._write_document(content, target, media_assets=media_assets, mode='append')

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
            raise RuntimeError(
                f'Notion document changed since it was loaded: expected '
                f'{source_document.revision!r}, got {current_revision!r}.')

        persisted = source_document
        applied_hunks: List[str] = []
        for hunk in patch_set.hunks:
            operation = adapter.patch_to_operation(
                hunk, persisted, media_assets=media_assets)
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
            refreshed = self._read_persisted_document(
                fs=fs, adapter=adapter, real_path=real_path,
                document_id=document_id, source_document=persisted)
            persisted = adapter.merge_refreshed_document(
                persisted, refreshed, patch=hunk, operation=operation,
                operation_result=result)

        title_updated = patch_set.new_title is not None \
            and patch_set.new_title != source_document.title
        if title_updated:
            update_title = getattr(fs, 'update_page_title', None)
            if not callable(update_title):
                raise TypeError(f'{type(fs).__name__} does not support update_page_title().')
            update_title(document_id, patch_set.new_title)
            persisted = self._read_persisted_document(
                fs=fs, adapter=adapter, real_path=real_path,
                document_id=document_id,
                source_document=persisted.model_copy(update={'title': patch_set.new_title}),
            )

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
            refreshed = self._read_persisted_document(
                fs=fs, adapter=adapter, real_path=real_path,
                document_id=document_id, source_document=persisted)
            persisted = adapter.merge_refreshed_document(
                persisted, refreshed, patch=sync_hunk, operation=operation,
                operation_result=sync_result)
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

    def _write_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None,
        mode: str,
    ) -> dict:
        source_document = content if isinstance(content, WriterDocument) else None
        protocol, _, fs, adapter, locator, document_id = \
            self._resolve_document_target(target, source_document=source_document)
        document = source_document.model_copy(deep=True) if source_document else \
            parse_document_markdown(
                content,
                document_id=adapter.make_document_id(document_id),
                stage='final',
                media_assets=media_assets,
            )
        document.provider_binding = {
            **(document.provider_binding or {}),
            'provider': protocol,
            'document_id': document_id,
            'uri': locator,
        }
        self._validate_available_images(document, media_assets, adapter)
        numbering = compute_numbering(build_numbering_view_from_ir(document))
        document = materialize_ir(document, numbering)
        native_blocks = adapter.ir_to_blocks(document, media_assets=media_assets)
        if document.title:
            update_title = getattr(fs, 'update_page_title', None)
            if not callable(update_title):
                raise TypeError(f'{type(fs).__name__} does not support update_page_title().')
            update_title(document_id, document.title)
        method_name = 'replace_doc_blocks' if mode == 'replace' else 'write_doc_blocks'
        write_blocks = getattr(fs, method_name, None)
        if not callable(write_blocks):
            raise TypeError(f'{type(fs).__name__} does not support {method_name}().')
        if native_blocks:
            write_blocks(document_id, native_blocks)
        warnings: List[str] = []
        if not native_blocks:
            warnings.append('Document has no publishable blocks.')
        return {
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
        return method(document_id=document_id, **operation.params)

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
