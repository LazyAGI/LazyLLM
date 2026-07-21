from __future__ import annotations
from typing import Any, Dict, List, Optional

from lazyllm import LOG
from pydantic import TypeAdapter, ValidationError

from .base import WriterToolBase
from ..data_models.resource import MaterialStyle, ResourceProfile
from ..data_models.task import InputResource, TargetDocument, WritingTask
from ..data_models.writer_ir import WriterBlock, WriterDocument, WriterStage
from ..prompts.profile_resources import RESOURCE_PROFILE_PROMPT

_WRITER_STAGE_ADAPTER = TypeAdapter(WriterStage)

_BLOCK_TYPE_MAPS: Dict[str, Dict[Any, str]] = {
    'feishu': {
        1: 'document', 2: 'paragraph',
        3: 'heading', 4: 'heading', 5: 'heading', 6: 'heading',
        7: 'heading', 8: 'heading', 9: 'heading', 10: 'heading', 11: 'heading',
        12: 'list_item', 13: 'list_item',
        14: 'code', 15: 'quote',
        27: 'image', 31: 'table', 32: 'table_cell',
    },
}


class WriterResourceTools(WriterToolBase):
    __public_apis__ = [
        'profile_resources',
        'document_to_docir',
        'write_to_document',
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
        locator = target.uri or target.doc_id
        if not locator:
            raise ValueError('target_document must provide uri or doc_id.')

        import lazyllm.tools.fs.client as _fs_client
        protocol, space_id, real_path = _fs_client.FS._parse(locator)
        fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)

        if hasattr(fs, 'resolve_link'):
            resolved_ref = fs.resolve_link(real_path) or {}
        else:
            resolved_ref = {}
        plain_text = fs.read_bytes(real_path).decode('utf-8', errors='replace')
        document_id = fs.get_document_id(real_path) if hasattr(fs, 'get_document_id') else ''
        raw_blocks = fs.get_doc_blocks(real_path, with_descendants=True) if hasattr(fs, 'get_doc_blocks') else []
        raw_blocks = raw_blocks or []

        bt_map = _BLOCK_TYPE_MAPS.get(protocol, {})
        try:
            stage = _WRITER_STAGE_ADAPTER.validate_python(target.meta.get('stage', 'final'))
        except ValidationError as exc:
            raise ValueError('target_document.meta.stage must be a valid WriterStage') from exc

        blocks: List[WriterBlock] = []
        for index, raw in enumerate(raw_blocks, start=1):
            if not isinstance(raw, dict):
                continue
            raw_block_type = raw.get('block_type', 'block')
            block_type = (
                bt_map.get(raw_block_type, 'block')
                if isinstance(raw_block_type, int)
                else str(raw_block_type or 'block')
            )
            external_block_id = str(raw.get('block_id') or '')
            node_id = external_block_id or f'block-{index}'
            blocks.append(WriterBlock(
                node_id=node_id,
                type=block_type,
                content=raw.get('plain_text') or '',
                stage=stage,
                numbering=self._block_numbering(raw, raw_block_type, block_type),
                provider_binding={
                    'provider': protocol,
                    'block_id': external_block_id,
                    'parent_block_id': raw.get('parent_id') or raw.get('parent_block_id'),
                },
                provider_payload=raw,
            ))

        external_document_id = target.doc_id or document_id or (
            resolved_ref.get('object_id')
            or resolved_ref.get('obj_token')
            or resolved_ref.get('node_token')
            or ''
        )
        internal_document_id = str(external_document_id or locator)
        title = target.title or resolved_ref.get('title') or ''

        if not blocks and plain_text.strip():
            blocks.append(WriterBlock(
                node_id=f'{internal_document_id}-content',
                type='paragraph',
                content=plain_text,
                stage=stage,
                provider_binding={
                    'provider': protocol,
                    'document_id': external_document_id,
                },
            ))

        document = WriterDocument(
            document_id=internal_document_id,
            stage=stage,
            title=title,
            blocks=blocks,
            revision=resolved_ref.get('revision'),
            metadata={
                'block_count': len(blocks),
                'source': target.model_dump(),
            },
            provider_binding={
                'provider': protocol,
                'uri': locator,
                'document_id': external_document_id,
                'revision': resolved_ref.get('revision'),
            },
        )

        return self._save_artifacts(
            {'document': document},
            step_name='document_to_docir',
            primary_key='document',
            summary='Loaded target document into WriterDocument.',
            counts={'blocks': len(blocks)},
            extra={
                'adapter': protocol,
                'document_id': document.document_id,
                'stage': document.stage,
            },
        ).model_dump()

    def write_to_document(self, content: Any, target_document: Any) -> dict:
        '''Render a final WriterDocument as Markdown and write it to a target platform.'''
        document = self._unified_model(content, WriterDocument)
        if document.stage != 'final':
            raise ValueError(f'content must have stage="final", got {document.stage!r}')
        text = self._render_document_markdown(document)
        target = self._unified_optional_model(target_document, TargetDocument) or TargetDocument()
        locator = target.uri or target.doc_id
        adapter = target.adapter or ''
        doc_id = ''

        if not locator:
            LOG.warning('write_to_document: no target document URI or doc_id, content not written to any platform')

        if locator:
            try:
                import lazyllm.tools.fs.client as _fs_client
                protocol, space_id, real_path = _fs_client.FS._parse(locator)
                fs = _fs_client.FS._get_or_create_fs(protocol, space_id, real_path)
                adapter = adapter or protocol
                fs.write_file(real_path, text.encode('utf-8'))
                resolved_ref = fs.resolve_link(real_path) if hasattr(fs, 'resolve_link') else {}
                resolved_ref = resolved_ref or {}
                doc_id = resolved_ref.get('object_id') or resolved_ref.get('obj_token') or ''
            except Exception:
                LOG.warning('write_to_document: FS write failed, content not written to target platform')

        return self._save_artifacts(
            {'write_result': {
                'doc_id': doc_id,
                'adapter': adapter,
                'locator': locator or '',
            }},
            step_name='write_to_document',
            primary_key='write_result',
            summary='Wrote content to target document.',
            extra={
                'adapter': adapter,
                'document_id': doc_id,
            },
        ).model_dump()

    def _block_numbering(
        self,
        raw: Dict[str, Any],
        raw_block_type: Any,
        block_type: str,
    ) -> Dict[str, Any]:
        numbering: Dict[str, Any] = {}
        if block_type == 'heading':
            level = raw.get('level') or raw.get('heading_level')
            if level is None and isinstance(raw_block_type, int) and 3 <= raw_block_type <= 11:
                level = raw_block_type - 2
            if isinstance(level, int) and not isinstance(level, bool):
                numbering['level'] = max(1, min(level, 9))
        elif block_type == 'list_item':
            ordered = raw.get('ordered')
            if not isinstance(ordered, bool) and isinstance(raw_block_type, int):
                ordered = raw_block_type == 13
            if isinstance(ordered, bool):
                numbering['ordered'] = ordered
        return numbering

    def _render_document_markdown(self, document: WriterDocument) -> str:
        parts: List[str] = []
        if document.title.strip():
            parts.append(f'# {document.title.strip()}')

        def render(block: WriterBlock) -> None:
            content = block.content.strip()
            if block.type == 'heading' and content:
                level = block.numbering.get('level', 1)
                if not isinstance(level, int) or isinstance(level, bool):
                    level = 1
                parts.append(f'{"#" * max(1, min(level, 9))} {content}')
            elif block.type == 'list_item' and content:
                marker = '1.' if block.numbering.get('ordered') else '-'
                parts.append(f'{marker} {content}')
            elif block.type == 'code' and content:
                language = str(block.provider_payload.get('language') or '')
                parts.append(f'```{language}\n{content}\n```')
            elif block.type == 'quote' and content:
                parts.append('\n'.join(f'> {line}' for line in content.splitlines()))
            elif block.type == 'divider':
                parts.append('---')
            elif content:
                parts.append(content)

            for child in block.children:
                render(child)

        for block in document.blocks:
            render(block)
        return '\n\n'.join(parts)
