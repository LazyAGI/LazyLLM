from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lazyllm import LOG
from pydantic import TypeAdapter, ValidationError

from .base import WriterToolBase
from ..adapter.base import NativePatchOperation, WriterAdapterBase
from ..adapter.feishu import FeishuWriterAdapter
from ..data_models.resource import MaterialStyle, ResourceProfile
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchHunk, PatchResult, PatchSet
from ..data_models.task import InputResource, TargetDocument, WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterStage
from ..numbering import (
    build_numbering_view_from_ir,
    compute_numbering,
    format_target_number,
    materialize_ir,
)
from ..prompts.profile_resources import RESOURCE_PROFILE_PROMPT
from ..tools.revision_tools import apply_patch_to_ir
from ..utils import parse_document_markdown, strip_heading_numbering

_WRITER_STAGE_ADAPTER = TypeAdapter(WriterStage)


class WriterResourceTools(WriterToolBase):
    __public_apis__ = [
        'profile_resources',
        'document_to_docir',
        'create_document',
        'write_to_document',
        'append_to_document',
        'replace_document',
        'apply_patch_to_document',
    ]

    _DEFAULT_ADAPTERS = {
        'feishu': FeishuWriterAdapter,
    }

    def _read_resource_content(self, res: InputResource) -> str:
        if res.resource_type == 'text':
            return res.inline_text or ''

        if res.resource_type == 'document':
            import lazyllm.tools.fs.client as _fs_client
            protocol, space_id, real_path = _fs_client.FS._parse(res.uri or '')
            fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)
            return fs.read_bytes(real_path).decode('utf-8', errors='replace')

        if res.resource_type in ('file', 'table', 'slide'):
            if not res.uri:
                return res.summary or ''
            from lazyllm.tools.rag.dataReader import SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=[str(res.uri)])
            nodes = reader._load_data()
            content = '\n'.join(n.text for n in nodes if n.text)
            return content if content.strip() else ''

        if res.resource_type == 'image':
            return res.summary or ''

        # url / kb — no ready gateway yet
        return res.summary or ''

    def profile_resources(
        self,
        task: Any,
        input_resources: Any = None,
    ) -> dict:
        '''Profile input resources for the writing task.'''
        writing_task = self._unified_model(task, WritingTask)
        inputs = self._unified_models(input_resources, InputResource)

        profiles: List[ResourceProfile] = []
        for res in inputs:
            content = self._read_resource_content(res)

            resource_role = res.meta.get('role', 'background')
            template_usage = res.meta.get('template', 'none')
            summary = res.summary or (content[:500] if content else '')
            key_facts: List[str] = []
            style: Optional[MaterialStyle] = None
            confidence = 1.0
            extracted_constraints: Dict[str, Any] = {}
            extracted_outline = None

            if self.llm is not None and content.strip():
                try:
                    prompt = RESOURCE_PROFILE_PROMPT.format(
                        query=writing_task.query,
                        task_type=writing_task.task_type,
                        constraints=str(writing_task.constraints),
                        title=res.title or '',
                        summary=res.summary or '',
                        content=content,
                    )
                    llm_result = self._call_llm_structured(prompt, ResourceProfile)
                    resource_role = llm_result.resource_role or resource_role
                    template_usage = llm_result.template_usage or template_usage
                    summary = llm_result.summary or summary
                    key_facts = llm_result.key_facts or []
                    if llm_result.style is not None:
                        style = llm_result.style
                    confidence = llm_result.confidence or 1.0
                    extracted_constraints = llm_result.extracted_constraints or {}
                    extracted_outline = llm_result.extracted_outline or None
                    if extracted_outline is not None:
                        extracted_outline.ui_editable = False
                except Exception:
                    LOG.warning('profile_resources: LLM analysis failed, using rule-based fallback')

            profiles.append(ResourceProfile(
                resource_id=res.resource_id or f'res-{len(profiles)}',
                resource_role=resource_role,
                template_usage=template_usage,
                summary=summary,
                key_facts=key_facts,
                style=style,
                confidence=confidence,
                extracted_constraints=extracted_constraints,
                extracted_outline=extracted_outline,
                raw_content=content[:3000] if content else None,
            ))

        return self._save_artifacts(
            {'resource_profiles': profiles},
            step_name='profile_resources',
            primary_key='resource_profiles',
            context_key=None,
            summary=f'Profiled {len(profiles)} resources.',
            counts={'resource_profiles': len(profiles)},
        ).model_dump()

    def document_to_docir(self, target_document: Any, context: Any = None) -> dict:
        '''Convert a target document into a WriterDocument artifact.'''
        target = self._unified_model(target_document, TargetDocument)
        protocol, real_path, fs, adapter, locator, external_document_id = \
            self._resolve_document_target(target)
        if not hasattr(fs, 'get_doc_blocks'):
            raise TypeError(f'{type(fs).__name__} does not support structured document reads.')
        raw_blocks = fs.get_doc_blocks(real_path, with_descendants=True) or []

        try:
            stage = _WRITER_STAGE_ADAPTER.validate_python(target.meta.get('stage', 'final'))
        except ValidationError as exc:
            raise ValueError('target_document.meta.stage must be a valid WriterStage') from exc
        title = target.title or ''
        document = adapter.blocks_to_ir(
            raw_blocks,
            external_document_id=external_document_id,
            stage=stage,
            title=title,
            uri=locator,
            revision=None,
        )
        document.metadata.update({
            'block_count': len(raw_blocks),
            'source': target.model_dump(),
        })

        return self._save_artifacts(
            {'document': document},
            step_name='document_to_docir',
            primary_key='document',
            summary='Loaded target document into WriterDocument.',
            counts={'blocks': len(raw_blocks)},
            extra={
                'adapter': protocol,
                'document_id': document.document_id,
                'stage': document.stage,
            },
        ).model_dump()

    def create_document(
        self,
        title: str,
        parent_uri: str = '',
        adapter: str = 'feishu',
    ) -> dict:
        '''Create an empty provider document and return its normalized target artifact.'''
        title = (title or '').strip()
        if not title:
            raise ValueError('title is required')
        adapter = (adapter or '').strip().lower()
        if not adapter:
            raise ValueError('adapter is required')

        import lazyllm.tools.fs.client as _fs_client
        parent_locator = (parent_uri or '').strip() or f'{adapter}:/'
        protocol, space_id, real_path = _fs_client.FS._parse(parent_locator)
        if protocol != adapter:
            raise ValueError(
                f'parent URI protocol {protocol!r} does not match adapter {adapter!r}.')
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
        target = TargetDocument(
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
        return self._save_artifacts(
            {'target_document': target},
            step_name='create_document',
            primary_key='target_document',
            context_key=None,
            summary='Created an empty provider document.',
            counts={'documents': 1},
            extra={
                'adapter': protocol,
                'document_id': document_id,
                'uri': target.uri,
            },
        ).model_dump()

    def write_to_document(self, content: Any, target_document: Any, media_assets: Any = None) -> dict:
        '''Backward-compatible alias for append_to_document().'''
        return self.append_to_document(content, target_document, media_assets)

    def append_to_document(self, content: Any, target_document: Any, media_assets: Any = None) -> dict:
        '''Append Writer IR or Markdown to an existing provider document.'''
        return self._write_document(content, target_document, media_assets=media_assets, mode='append')

    def replace_document(self, content: Any, target_document: Any, media_assets: Any = None) -> dict:
        '''Replace an existing provider document with Writer IR or Markdown.'''
        return self._write_document(content, target_document, media_assets=media_assets, mode='replace')

    def _write_document(
        self,
        content: Any,
        target_document: Any,
        media_assets: Any = None,
        *,
        mode: str,
    ) -> dict:
        source = self._unified_document(content)
        source_document = source if isinstance(source, WriterDocument) else None
        target = self._unified_optional_model(target_document, TargetDocument) or TargetDocument()
        locator = self._target_locator(target, source_document)

        if not locator:
            LOG.warning(
                '%s_to_document: no target document URI or doc_id, '
                'content not written to any platform',
                mode,
            )
            return self._save_write_result('', '', '', 0)

        protocol, real_path, fs, adapter, locator, document_id = \
            self._resolve_document_target(target, source_document=source_document)
        media_library = self._unified_optional_model(media_assets, MediaAssetLibrary)
        document = source_document or parse_document_markdown(
            source, document_id=adapter.make_document_id(document_id), stage='final',
            media_assets=media_library,
        )
        if not document.provider_binding.get('document_id'):
            document.provider_binding = {
                **(document.provider_binding or {}),
                'provider': protocol,
                'document_id': document_id,
            }
        warnings: List[str] = []
        self._validate_available_images(document, media_library)
        numbering = compute_numbering(build_numbering_view_from_ir(document))
        document = materialize_ir(document, numbering)
        method_name = 'replace_doc_blocks' if mode == 'replace' else 'write_doc_blocks'
        write_blocks = getattr(fs, method_name, None)
        if not callable(write_blocks):
            raise TypeError(f'{type(fs).__name__} does not support {method_name}().')
        native_blocks = adapter.ir_to_blocks(document, media_assets=media_library)
        if document.title:
            self._update_document_title(fs, document_id, document.title, document.revision)
        if not native_blocks:
            warnings.append('Document has no publishable blocks.')
            return self._save_write_result(document_id, protocol, locator, 0, warnings)
        write_blocks(document_id, native_blocks)
        return self._save_write_result(document_id, protocol, locator, len(native_blocks), warnings)

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
                    f'Image block {block.node_id!r} requires exactly one media_asset reference.'
                )
            asset = media_assets.assets.get(references[0]) if media_assets else None
            if asset is None or not asset.local_path or not Path(asset.local_path).is_file():
                raise ValueError(f'Image block {block.node_id!r} media is unavailable.')

    def apply_patch_to_document(  # noqa: C901
        self,
        patch_set: Any,
        source_document: Any,
        target_document: Any = None,
        media_assets: Any = None,
    ) -> dict:
        '''Translate a PatchSet into native block operations and persist it.'''
        patch = self._unified_model(patch_set, PatchSet)
        source = self._unified_model(source_document, WriterDocument)
        media_library = self._unified_optional_model(media_assets, MediaAssetLibrary)
        if patch.target_doc_id != source.document_id:
            raise ValueError(
                f'patch target_doc_id {patch.target_doc_id!r} does not match '
                f'document_id {source.document_id!r}.'
            )
        if not patch.hunks and (patch.new_title is None or patch.new_title == source.title):
            raise ValueError('patch contains no document operations.')

        target = self._unified_optional_model(target_document, TargetDocument) or TargetDocument()
        protocol, real_path, fs, adapter, locator, document_id = \
            self._resolve_document_target(target, source_document=source)
        revised_document, _ = apply_patch_to_ir(source, patch, media_assets=media_library)
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
                fs=fs, adapter=adapter, real_path=real_path, locator=locator,
                document_id=document_id, source_document=previous,
            )
            merge = getattr(adapter, 'merge_refreshed_document', None)
            return merge(previous, refreshed, operation_result=result, **merge_kwargs) \
                if callable(merge) else refreshed

        applied_hunks: List[str] = []
        persisted_document = source
        expected_title = patch.new_title if patch.new_title is not None else source.title
        title_updated = patch.new_title is not None and patch.new_title != source.title
        normalized_fields: Dict[str, List[str]] = {}
        for hunk in patch.hunks:
            hunk = self._materialize_hunk_feishu_links(
                hunk,
                block_id_by_node_id=block_id_by_node_id,
                numbering=final_numbering,
                document_id=document_id,
            )
            operation = adapter.patch_to_operation(
                hunk, persisted_document, media_assets=media_library)
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
                    operation.operation, hunk_id, block_id,
                    persisted_document.revision, exc,
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
                operation_result, patch=hunk, operation=operation,
            )

        if title_updated:
            title_result = self._update_document_title(
                fs, document_id, expected_title, persisted_document.revision)
            persisted_document = refresh(
                persisted_document.model_copy(update={'title': expected_title}),
                title_result,
            )
        elif not patch.hunks:
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
            )
            operation = adapter.patch_to_operation(
                sync_hunk, persisted_document, media_assets=media_library)
            operation_result = self._execute_native_operation(
                fs, document_id, operation, persisted_document.revision)
            applied_hunks.append(sync_hunk.hunk_id)
            persisted_document = refresh(
                persisted_document.model_copy(update={'title': expected_title}),
                operation_result, patch=sync_hunk, operation=operation,
            )

        patch_result = PatchResult(
            patch_id=patch.patch_id,
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
        return self._save_artifacts(
            {
                'patch_result': patch_result,
                'persisted_document': persisted_document,
            },
            step_name='apply_patch_to_document',
            primary_key='patch_result',
            summary='Applied patch to provider document.',
            counts={'applied': len(applied_hunks), 'failed': 0},
            extra={
                'adapter': protocol,
                'document_id': document_id,
            },
        ).model_dump()

    @staticmethod
    def _materialize_hunk_feishu_links(
        hunk: PatchHunk,
        *,
        block_id_by_node_id: Dict[str, Any],
        numbering: Dict[str, Any],
        document_id: str,
    ) -> PatchHunk:
        if hunk.block is None:
            return hunk
        hunk = hunk.model_copy(deep=True)
        block = hunk.block
        for item in block.iter_blocks():
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
                    'url': f'https://feishu.cn/docx/{document_id}#{target_block_id}',
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

    def _resolve_document_target(
        self,
        target: TargetDocument,
        source_document: Optional[WriterDocument] = None,
    ) -> Tuple[str, str, Any, WriterAdapterBase, str, str]:
        locator = self._target_locator(target, source_document)
        if not locator:
            raise ValueError('target_document or source_document provider_binding must provide uri or doc_id.')

        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        requested_adapter = target.adapter or (
            source_document.provider_binding.get('provider') if source_document else None)
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
            self._writer_adapter(protocol),
            locator,
            document_id.strip(),
        )

    def _read_persisted_document(
        self,
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

    def _writer_adapter(self, protocol: str) -> WriterAdapterBase:
        configured = self.adapters.get(protocol)
        if configured is None:
            configured = self._DEFAULT_ADAPTERS.get(protocol)
        if configured is None:
            raise ValueError(f'No Writer adapter is configured for provider {protocol!r}.')
        adapter = configured() if isinstance(configured, type) else configured
        if not isinstance(adapter, WriterAdapterBase):
            raise TypeError(
                f'Writer adapter for {protocol!r} must inherit WriterAdapterBase, '
                f'got {type(adapter).__name__}.'
            )
        return adapter

    @staticmethod
    def _target_locator(
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
        if document_id and provider == 'feishu':
            return f'feishu:/~docx/{document_id}'
        return str(document_id or '')

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

    def _save_write_result(
        self,
        document_id: str,
        adapter: str,
        locator: str,
        block_count: int,
        warnings: Optional[List[str]] = None,
    ) -> dict:
        return self._save_artifacts(
            {'write_result': {
                'doc_id': document_id,
                'adapter': adapter,
                'locator': locator,
                'block_count': block_count,
            }},
            step_name='write_to_document',
            primary_key='write_result',
            summary='Wrote content to target document.' if document_id else 'No target document was provided.',
            counts={'blocks': block_count},
            warnings=warnings,
            extra={
                'adapter': adapter,
                'document_id': document_id,
            },
        ).model_dump()
