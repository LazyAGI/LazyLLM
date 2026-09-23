from __future__ import annotations
from contextvars import copy_context
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from lazyllm import LOG
from lazyllm.common import ThreadPoolExecutor
from pydantic import TypeAdapter, ValidationError

from .base import WriterToolBase
from ..data_models.resource import MaterialStyle, ResourceProfile
from ..data_models.multimodal import MediaAssetLibrary
from ..data_models.revision import PatchSet
from ..data_models.task import InputResource, TargetDocument, WritingTask
from ..data_models.writer_ir import WriterDocument, WriterStage
from ..prompts.profile_resources import RESOURCE_PROFILE_PROMPT
from ..provider import (
    GitHubWriterProvider,
    WriterProviderWriteOutcomeError,
    get_writer_provider,
    is_ambiguous_write_error,
    match_writer_provider,
)
from ..utils import make_markdown_tool_result

_WRITER_STAGE_ADAPTER = TypeAdapter(WriterStage)


class WriterResourceTools(WriterToolBase):
    __public_apis__ = [
        'profile_resources',
        'load_document',
        'document_to_docir',
        'create_document',
        'write_to_document',
        'append_to_document',
        'replace_document',
        'apply_patch_to_document',
    ]

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

        def profile_resource(index: int, res: InputResource) -> ResourceProfile:
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

            return ResourceProfile(
                resource_id=res.resource_id or f'res-{index}',
                resource_role=resource_role,
                template_usage=template_usage,
                summary=summary,
                key_facts=key_facts,
                style=style,
                confidence=confidence,
                extracted_constraints=extracted_constraints,
                extracted_outline=extracted_outline,
                raw_content=content[:3000] if content else None,
            )

        if self.llm is not None and len(inputs) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(inputs))) as executor:
                futures = [
                    executor.submit(copy_context().run, profile_resource, index, res)
                    for index, res in enumerate(inputs)
                ]
                profiles = [future.result() for future in futures]
        else:
            profiles = [profile_resource(index, res) for index, res in enumerate(inputs)]

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
        try:
            stage = _WRITER_STAGE_ADAPTER.validate_python(target.meta.get('stage', 'final'))
        except ValidationError as exc:
            raise ValueError('target_document.meta.stage must be a valid WriterStage') from exc
        provider = self._writer_provider(target)
        provider.require_capability('load')
        loaded = provider.load_document(target, stage=stage)
        document = loaded.get('source_document')
        if loaded.get('representation') != 'ir' or not isinstance(document, WriterDocument):
            raise TypeError(
                f'Provider {target.adapter!r} did not return a WriterDocument representation.')
        block_count = int(loaded.get('block_count') or 0)
        protocol = str(loaded.get('provider') or target.adapter or '')

        return self._save_artifacts(
            {'document': document},
            step_name='document_to_docir',
            primary_key='document',
            summary='Loaded target document into WriterDocument.',
            counts={'blocks': block_count},
            extra={
                'adapter': protocol,
                'document_id': document.document_id,
                'stage': document.stage,
            },
        ).model_dump()

    def load_document(self, target_document: Any, context: Any = None) -> dict:
        '''Load a provider document while preserving its existing Writer representation.'''
        target = self._unified_model(target_document, TargetDocument)
        try:
            stage = _WRITER_STAGE_ADAPTER.validate_python(target.meta.get('stage', 'final'))
        except ValidationError as exc:
            raise ValueError('target_document.meta.stage must be a valid WriterStage') from exc
        writer_provider = self._writer_provider(target)
        writer_provider.require_capability('load')
        load_options = {}
        if isinstance(writer_provider, GitHubWriterProvider) and self.artifact_store:
            load_options['resource_cache_dir'] = str(
                Path(self.artifact_store) / 'github-resources' / uuid.uuid4().hex,
            )
        loaded = writer_provider.load_document(target, stage=stage, **load_options)
        representation = str(loaded.get('representation') or '').strip().lower()
        source = loaded.get('source_document')
        resolved_target = self._unified_model(
            loaded.get('target_document', target), TargetDocument)
        provider = str(
            loaded.get('provider') or resolved_target.adapter or target.adapter or '',
        ).strip().lower()
        block_count = int(loaded.get('block_count') or 0)
        input_resources = self._unified_models(
            loaded.get('input_resources'), InputResource,
        )
        resource_warnings = [
            str(item) for item in (loaded.get('resource_warnings') or []) if str(item).strip()
        ]
        extra = {
            'adapter': provider,
            'document_id': str(resolved_target.doc_id or target.doc_id or ''),
            'representation': representation,
            'stage': stage,
        }
        if representation == 'ir':
            if not isinstance(source, WriterDocument):
                raise TypeError(f'Provider {provider!r} returned invalid WriterDocument content.')
            result = self._save_artifacts(
                {
                    'source_document': source,
                    'target_document': resolved_target,
                    **({'input_resources': input_resources} if input_resources else {}),
                },
                step_name='load_document',
                primary_key='source_document',
                context_key=None,
                summary='Loaded provider document into Writer IR.',
                counts={'blocks': block_count, 'input_resources': len(input_resources)},
                warnings=resource_warnings,
                extra=extra,
            ).model_dump()
        elif representation == 'markdown':
            if not isinstance(source, str):
                raise TypeError(f'Provider {provider!r} returned invalid Markdown content.')
            source_path = self._write_markdown_artifact('source_document.md', source)
            result = make_markdown_tool_result(
                path=source_path,
                step_name='load_document',
                artifact_key='source_document',
                summary='Loaded provider document as Markdown.',
                counts={'characters': len(source)},
                extra=extra,
            ).model_dump()
            target_path = self._write_single_artifact(
                resolved_target,
                'target_document.json',
                artifact_key='target_document',
                extra_meta={
                    'step_name': 'load_document',
                    'artifact_key': 'target_document',
                    'primary_key': 'source_document',
                    'status': 'success',
                },
            )
            result['metadata']['artifact_paths']['target_document'] = target_path
            result['metadata']['schema_names']['target_document'] = self._artifact_schema_name(
                resolved_target, 'target_document')
            if input_resources:
                resources_path = self._write_single_artifact(
                    input_resources,
                    'input_resources.json',
                    artifact_key='input_resources',
                    extra_meta={
                        'step_name': 'load_document',
                        'artifact_key': 'input_resources',
                        'primary_key': 'source_document',
                        'status': 'success',
                    },
                )
                result['metadata']['artifact_paths']['input_resources'] = resources_path
                result['metadata']['schema_names']['input_resources'] = self._artifact_schema_name(
                    input_resources, 'input_resources')
            result['metadata']['warnings'] = resource_warnings
            result['metadata']['counts']['input_resources'] = len(input_resources)
        else:
            raise ValueError(
                f'Provider {provider!r} returned unsupported representation {representation!r}.')
        result['representation'] = representation
        return result

    def create_document(
        self,
        title: str,
        parent_uri: str = '',
        adapter: str = '',
    ) -> dict:
        '''Create an empty provider document and return its normalized target artifact.'''
        title = (title or '').strip()
        if not title:
            raise ValueError('title is required')
        adapter = (adapter or '').strip().lower()
        if not adapter:
            raise ValueError('adapter is required')
        provider = get_writer_provider(adapter, adapters=self.adapters)
        provider.require_capability('create')
        target = provider.create_document(title, parent_uri)
        document_id = str(target.doc_id or '')
        return self._save_artifacts(
            {'target_document': target},
            step_name='create_document',
            primary_key='target_document',
            context_key=None,
            summary='Created an empty provider document.',
            counts={'documents': 1},
            extra={
                'adapter': adapter,
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
        provider_key = self._provider_key(target, source_document)
        locator = target.uri or (
            source_document.provider_binding.get('uri') if source_document else '')
        if not provider_key and not locator:
            LOG.warning(
                '%s_to_document: no target document URI or doc_id, '
                'content not written to any platform',
                mode,
            )
            return self._save_write_result('', '', '', 0)
        media_library = self._unified_optional_model(media_assets, MediaAssetLibrary)
        provider = self._writer_provider(target, source_document)
        provider_key = provider.provider
        provider.require_capability(mode)
        try:
            result = (
                provider.replace_document(source, target, media_assets=media_library)
                if mode == 'replace'
                else provider.append_document(source, target, media_assets=media_library)
            )
        except WriterProviderWriteOutcomeError:
            raise
        except Exception as exc:
            if is_ambiguous_write_error(exc):
                raise WriterProviderWriteOutcomeError(provider.provider, mode) from exc
            raise
        return self._save_write_result(
            str(result.get('doc_id') or ''),
            str(result.get('adapter') or provider_key),
            str(result.get('locator') or target.uri or ''),
            int(result.get('block_count') or 0),
            list(result.get('warnings') or []),
            provider_result=result,
            persisted_document=result.get('persisted_document'),
            representation=str(result.get('representation') or ''),
        )

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
        target = self._unified_optional_model(target_document, TargetDocument) or TargetDocument()
        provider = self._writer_provider(target, source)
        provider.require_capability('patch')
        try:
            result = provider.apply_patch_to_document(
                patch,
                source,
                target,
                media_assets=media_library,
            )
        except WriterProviderWriteOutcomeError:
            raise
        except Exception as exc:
            if is_ambiguous_write_error(exc):
                raise WriterProviderWriteOutcomeError(provider.provider, 'patch') from exc
            raise
        patch_result = result['patch_result']
        persisted_document = result['persisted_document']
        protocol = str(result.get('provider') or self._provider_key(target, source) or '')
        document_id = str(result.get('document_id') or '')
        return self._save_artifacts(
            {
                'patch_result': patch_result,
                'persisted_document': persisted_document,
            },
            step_name='apply_patch_to_document',
            primary_key='patch_result',
            summary='Applied patch to provider document.',
            counts={
                'applied': len(patch_result.applied_hunks),
                'failed': len(patch_result.failed_hunks),
            },
            extra={
                'adapter': protocol,
                'document_id': document_id,
            },
        ).model_dump()

    def _writer_provider(
        self,
        target: TargetDocument,
        source_document: Optional[WriterDocument] = None,
    ):
        provider_key = self._provider_key(target, source_document)
        if provider_key:
            return get_writer_provider(provider_key, adapters=self.adapters)
        locator = target.uri or ''
        if locator:
            return match_writer_provider(locator, adapters=self.adapters)
        raise ValueError(
            'target_document.adapter or source_document provider_binding.provider is required.')

    @staticmethod
    def _provider_key(
        target: TargetDocument,
        source_document: Optional[WriterDocument] = None,
    ) -> str:
        provider = target.adapter
        if not provider and source_document is not None:
            provider = source_document.provider_binding.get('provider')
        return str(provider or '').strip().lower()

    def _save_write_result(
        self,
        document_id: str,
        adapter: str,
        locator: str,
        block_count: int,
        warnings: Optional[List[str]] = None,
        provider_result: Optional[Dict[str, Any]] = None,
        persisted_document: Any = None,
        representation: str = '',
    ) -> dict:
        write_result = dict(provider_result or {})
        write_result.update({
            'doc_id': document_id,
            'adapter': adapter,
            'locator': locator,
            'block_count': block_count,
        })
        artifacts: Dict[str, Any] = {'write_result': write_result}
        if persisted_document is not None:
            artifacts['persisted_document'] = persisted_document
        return self._save_artifacts(
            artifacts,
            step_name='write_to_document',
            primary_key='write_result',
            summary='Wrote content to target document.' if document_id else 'No target document was provided.',
            counts={'blocks': block_count},
            warnings=warnings,
            extra={
                'adapter': adapter,
                'document_id': document_id,
                'representation': representation or None,
            },
        ).model_dump()
