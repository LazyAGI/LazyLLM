from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from lazyllm import LOG
from pydantic import TypeAdapter, ValidationError

from .base import WriterToolBase
from ..adapter.base import NativePatchOperation, WriterAdapterBase
from ..adapter.feishu import FeishuWriterAdapter
from ..data_models.resource import MaterialStyle, ResourceProfile
from ..data_models.revision import PatchResult, PatchSet
from ..data_models.task import InputResource, TargetDocument, WritingTask
from ..data_models.writer_ir import WriterDocument, WriterStage
from ..prompts.profile_resources import RESOURCE_PROFILE_PROMPT

_WRITER_STAGE_ADAPTER = TypeAdapter(WriterStage)


class WriterResourceTools(WriterToolBase):
    __public_apis__ = [
        'profile_resources',
        'document_to_docir',
        'write_to_document',
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

    def write_to_document(self, content: Any, target_document: Any) -> dict:
        '''Convert a final WriterDocument into native blocks and write them.'''
        document = self._unified_model(content, WriterDocument)
        if document.stage != 'final':
            raise ValueError(f'content must have stage="final", got {document.stage!r}')
        target = self._unified_optional_model(target_document, TargetDocument) or TargetDocument()
        locator = self._target_locator(target, document)

        if not locator:
            LOG.warning('write_to_document: no target document URI or doc_id, content not written to any platform')
            return self._save_write_result('', '', '', 0)

        protocol, real_path, fs, adapter, locator, document_id = \
            self._resolve_document_target(target, source_document=document)
        if not hasattr(fs, 'write_doc_blocks'):
            raise TypeError(f'{type(fs).__name__} does not support structured document writes.')
        native_blocks = adapter.ir_to_blocks(document)
        fs.write_doc_blocks(document_id, native_blocks)
        return self._save_write_result(document_id, protocol, locator, len(native_blocks))

    def apply_patch_to_document(
        self,
        patch_set: Any,
        source_document: Any,
        target_document: Any = None,
    ) -> dict:
        '''Translate a PatchSet into native block operations and persist it.'''
        patch = self._unified_model(patch_set, PatchSet)
        source = self._unified_model(source_document, WriterDocument)
        if patch.target_doc_id != source.document_id:
            raise ValueError(
                f'patch target_doc_id {patch.target_doc_id!r} does not match '
                f'document_id {source.document_id!r}.'
            )

        target = self._unified_optional_model(target_document, TargetDocument) or TargetDocument()
        protocol, real_path, fs, adapter, locator, document_id = \
            self._resolve_document_target(target, source_document=source)

        applied_hunks: List[str] = []
        persisted_document = source
        for hunk in patch.hunks:
            operation = adapter.patch_to_operation(hunk, persisted_document)
            self._execute_native_operation(
                fs, document_id, operation, persisted_document.revision)
            applied_hunks.append(hunk.hunk_id or hunk.target_node_id)
            persisted_document = self._read_persisted_document(
                fs=fs,
                adapter=adapter,
                real_path=real_path,
                locator=locator,
                document_id=document_id,
                source_document=source,
            )

        if not patch.hunks:
            persisted_document = self._read_persisted_document(
                fs=fs,
                adapter=adapter,
                real_path=real_path,
                locator=locator,
                document_id=document_id,
                source_document=source,
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
                'operation_count': len(applied_hunks),
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
            revision=None,
        )
        document.metadata.update({
            'block_count': len(latest_blocks),
            'source': source_document.metadata.get('source', {}),
        })
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
        if operation.operation in {'update', 'delete'} and 'document_revision_id' not in params:
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
            extra={
                'adapter': adapter,
                'document_id': document_id,
            },
        ).model_dump()
