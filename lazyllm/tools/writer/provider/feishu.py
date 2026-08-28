from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from lazyllm import LOG

from .base import WriterProviderBase
from ..adapter.base import NativePatchOperation, WriterAdapterBase
from ..adapter.feishu import FeishuWriterAdapter, feishu_block_url
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchHunk, PatchResult, PatchSet
from ..data_models.task import TargetDocument
from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterStage
from ..numbering import (
    build_numbering_view_from_ir,
    compute_numbering,
    format_target_number,
    materialize_ir,
)
from ..tools.revision_tools import apply_patch_to_ir
from ..utils import parse_document_markdown, strip_heading_numbering


_FEISHU_URL_RE = re.compile(
    r'^https?://[^/]*(?:feishu\.(?:cn|com)|larksuite\.com)/',
    re.IGNORECASE,
)


class FeishuWriterProvider(WriterProviderBase):
    '''Orchestrate Feishu document IO and structured Writer conversion.'''

    provider = 'feishu'

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
    ) -> dict:
        protocol, real_path, fs, adapter, locator, external_document_id = \
            self._resolve_document_target(target)
        if not hasattr(fs, 'get_doc_blocks'):
            raise TypeError(f'{type(fs).__name__} does not support structured document reads.')
        raw_blocks = fs.get_doc_blocks(real_path, with_descendants=True) or []
        document = adapter.blocks_to_ir(
            raw_blocks,
            external_document_id=external_document_id,
            stage=stage,
            title=target.title or '',
            uri=locator,
            revision=None,
        )
        document.metadata.update({
            'block_count': len(raw_blocks),
            'source': target.model_dump(),
        })
        resolved_target = target.model_copy(deep=True)
        resolved_target.doc_id = external_document_id
        resolved_target.uri = locator
        resolved_target.adapter = protocol
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

    def replace_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        return self._write_document(
            content, target, media_assets=media_assets, mode='replace')

    def append_document(
        self,
        content: WriterDocument | str,
        target: TargetDocument,
        *,
        media_assets: MediaAssetLibrary | None = None,
    ) -> dict:
        return self._write_document(
            content, target, media_assets=media_assets, mode='append')

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
        document = source_document or parse_document_markdown(
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
        warnings: List[str] = []
        self._validate_available_images(document, media_assets)
        numbering = compute_numbering(build_numbering_view_from_ir(document))
        document = materialize_ir(document, numbering)
        method_name = 'replace_doc_blocks' if mode == 'replace' else 'write_doc_blocks'
        write_blocks = getattr(fs, method_name, None)
        if not callable(write_blocks):
            raise TypeError(f'{type(fs).__name__} does not support {method_name}().')
        native_blocks = adapter.ir_to_blocks(document, media_assets=media_assets)
        if document.title:
            self._update_document_title(fs, document_id, document.title, document.revision)
        if not native_blocks:
            warnings.append('Document has no publishable blocks.')
        else:
            write_blocks(document_id, native_blocks)
        return {
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
        revised_document, _ = apply_patch_to_ir(
            source_document, patch_set, media_assets=media_assets)
        final_numbering = compute_numbering(build_numbering_view_from_ir(revised_document))
        block_id_by_node_id = {
            block.node_id: block.provider_binding.get('block_id')
            for block in revised_document.iter_blocks()
        }

        def refresh(previous: WriterDocument, result: Any = None, **merge_kwargs) -> WriterDocument:
            revision = result.get('document_revision_id') if isinstance(result, dict) else None
            if revision is not None and not isinstance(revision, bool):
                previous = previous.model_copy(update={'revision': str(revision)})
            refreshed = self._read_persisted_document(
                fs=fs,
                adapter=adapter,
                real_path=real_path,
                locator=locator,
                document_id=document_id,
                source_document=previous,
            )
            merge = getattr(adapter, 'merge_refreshed_document', None)
            return merge(previous, refreshed, operation_result=result, **merge_kwargs) \
                if callable(merge) else refreshed

        applied_hunks: List[str] = []
        persisted_document = source_document
        expected_title = (
            patch_set.new_title if patch_set.new_title is not None else source_document.title)
        title_updated = (
            patch_set.new_title is not None and patch_set.new_title != source_document.title)
        normalized_fields: Dict[str, List[str]] = {}
        for hunk in patch_set.hunks:
            hunk = self._materialize_hunk_feishu_links(
                hunk,
                block_id_by_node_id=block_id_by_node_id,
                numbering=final_numbering,
                document_id=document_id,
                document_uri=locator,
            )
            operation = adapter.patch_to_operation(
                hunk, persisted_document, media_assets=media_assets)
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
            persisted_document = refresh(
                persisted_document.model_copy(update={'title': expected_title}),
                operation_result,
                patch=hunk,
                operation=operation,
            )

        if title_updated:
            title_result = self._update_document_title(
                fs, document_id, expected_title, persisted_document.revision)
            persisted_document = refresh(
                persisted_document.model_copy(update={'title': expected_title}),
                title_result,
            )
        elif not patch_set.hunks:
            persisted_document = refresh(persisted_document)

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
            persisted_document = refresh(
                persisted_document.model_copy(update={'title': expected_title}),
                operation_result,
                patch=sync_hunk,
                operation=operation,
            )

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
        params.setdefault('document_id', document_id)
        if operation.operation in {'create', 'update', 'replace', 'delete', 'move'} \
                and 'document_revision_id' not in params:
            try:
                params['document_revision_id'] = int(revision) if revision is not None else -1
            except (TypeError, ValueError):
                params['document_revision_id'] = -1
        return method(**params)

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
        document = adapter.blocks_to_ir(
            latest_blocks,
            external_document_id=document_id,
            stage=source_document.stage,
            title=source_document.title,
            uri=locator,
            revision=source_document.revision,
        )
        document.metadata = {
            **deepcopy(source_document.metadata),
            'block_count': len(latest_blocks),
            'source': source_document.metadata.get('source', {}),
        }
        return document


__all__ = ['FeishuWriterProvider']
