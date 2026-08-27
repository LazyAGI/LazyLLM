from __future__ import annotations
import re
from typing import Any, Dict, List, Literal, Optional

from .base import WriterToolBase
from .stream_tools import DraftPreviewStream, build_outline_stream
from ..data_models.context import WritingContext
from ..data_models.multimodal import VisualPlan
from ..data_models.resource import ResourceProfile
from ..data_models.task import WritingTask
from ..data_models.writer_ir import ContentRef, WriterBlock, WriterDocument
from ..data_models.planning import SectionInstruction, SectionInstructionList, ShortWritingPlan
from ..prompts import (
    GENERATE_OUTLINE_MARKDOWN_PROMPT,
    GENERATE_OUTLINE_PROMPT,
    GENERATE_REWRITE_OUTLINE_PROMPT,
    GENERATE_REWRITE_SECTION_INSTRUCTIONS_PROMPT,
    GENERATE_SECTION_INSTRUCTIONS_PROMPT,
    GENERATE_SHORT_VISUAL_PLAN_PROMPT,
    GENERATE_SHORT_WRITING_PLAN_PROMPT,
    GENERATE_VISUAL_PLAN_MARKDOWN_PROMPT,
    GENERATE_VISUAL_PLAN_PROMPT,
)
from ..utils import (
    get_markdown_outline_targets,
    make_markdown_tool_result,
    parse_markdown_sections,
    render_block_markdown,
    render_document_markdown,
    strip_heading_numbering,
    to_prompt_json,
)


_IMAGE_RESOURCE_FACT_PATTERN = re.compile(
    r'^\s*[^:\n]+\.(?:avif|gif|jpe?g|png|svg|webp)\s*:\s*',
    re.IGNORECASE,
)
_VISUAL_DIRECTIVE_PATTERN = re.compile(
    r'(?:必须|务必|只能|仅限|禁止|不得|不要|插入|放入|嵌入|复用|生成|改用|替换|替代|图片位置)'
    r'|\b(?:must|only|never|insert|embed|reuse|generate|replace|substitute)\b',
    re.IGNORECASE,
)


class WriterPlanningTools(WriterToolBase):
    __public_apis__ = [
        'generate_outline',
        'generate_rewrite_outline',
        'generate_rewrite_section_instructions',
        'generate_section_instructions',
        'generate_short_visual_plan',
        'generate_short_writing_plan',
    ]

    def generate_outline(
        self,
        task: Any,
        context: Any,
        resource_profiles: Any = None,
        execution_results: Any = None,
        representation: Literal['ir', 'markdown'] | None = None,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        writing_context = self._unified_model(context, WritingContext)
        profiles = self._unified_models(resource_profiles, ResourceProfile)
        execution_data = self._unified_raw_data(execution_results)
        resolved_representation = self._resolve_representation(writing_task, representation)

        if resolved_representation == 'markdown':
            prompt = GENERATE_OUTLINE_MARKDOWN_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                context_json=to_prompt_json(writing_context),
                resource_profiles_json=to_prompt_json(profiles),
                execution_results_json=to_prompt_json(execution_data),
            )
            outline = self._call_llm_text(prompt).strip() + '\n'
            outline = self._normalize_markdown_outline(outline)
            _, targets = get_markdown_outline_targets(outline)
            path = self._write_markdown_artifact('outline.md', outline)
            return make_markdown_tool_result(
                path=path,
                step_name='generate_outline',
                artifact_key='outline',
                summary='Generated writing outline as Markdown.',
                counts={
                    'top_level_sections': len(targets),
                    'outline_nodes': len(parse_markdown_sections(outline)),
                    'characters': len(outline),
                },
                extra={
                    'representation': 'markdown',
                    'task_id': writing_task.task_id,
                    'context_id': writing_context.context_id,
                    'resource_profile_count': len(profiles),
                    'has_execution_results': execution_data is not None,
                },
            ).model_dump()

        document_id = f'{writing_context.context_id}-outline'
        prompt = GENERATE_OUTLINE_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            document_id=document_id,
            context_json=to_prompt_json(writing_context),
            resource_profiles_json=to_prompt_json(profiles),
            execution_results_json=to_prompt_json(execution_data),
        )
        outline = self._call_llm_structured(prompt, WriterDocument)
        outline.document_id = document_id
        outline = self._normalize_outline(outline, writing_task, writing_context, profiles)

        result = self._save_artifacts(
            {'outline': outline},
            step_name='generate_outline',
            primary_key='outline',
            context_key=None,
            summary='Generated writing outline.',
            counts={
                'top_level_sections': len(outline.blocks),
                'outline_nodes': len(list(outline.iter_blocks())),
            },
            extra={'representation': 'ir'},
            artifact_meta={
                'task_id': writing_task.task_id,
                'context_id': writing_context.context_id,
                'resource_profile_count': len(profiles),
                'has_execution_results': execution_data is not None,
            },
        )
        return result.model_dump()

    def stream_outline(
        self,
        task: Any,
        context: Any,
        resource_profiles: Any = None,
        execution_results: Any = None,
        representation: Literal['ir', 'markdown'] | None = None,
        *,
        idle_timeout: float | None = None,
    ) -> DraftPreviewStream:
        return build_outline_stream(
            self,
            task,
            context,
            resource_profiles,
            execution_results,
            representation,
            idle_timeout=idle_timeout,
        )

    def generate_rewrite_outline(
        self,
        task: Any,
        source_document: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        source = self._unified_document(source_document)
        if not isinstance(source, WriterDocument):
            raise TypeError('generate_rewrite_outline requires a WriterDocument source.')
        writing_context = self._unified_model(context, WritingContext)
        document_id = f'{writing_context.context_id}-rewrite-outline'
        prompt = GENERATE_REWRITE_OUTLINE_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            document_id=document_id,
            context_json=to_prompt_json(writing_context),
            source_document_json=render_document_markdown(source),
        )
        outline = self._call_llm_structured(prompt, WriterDocument)
        outline.document_id = document_id
        outline = self._normalize_outline(outline, writing_task, writing_context, [])
        return self._save_artifacts(
            {'outline': outline},
            step_name='generate_rewrite_outline',
            primary_key='outline',
            context_key=None,
            summary='Generated an internal outline for a complete document rewrite.',
            counts={
                'top_level_sections': len(outline.blocks),
                'outline_nodes': len(list(outline.iter_blocks())),
            },
            extra={'representation': 'ir'},
            artifact_meta={
                'task_id': writing_task.task_id,
                'context_id': writing_context.context_id,
                'source_document_id': source.document_id,
            },
        ).model_dump()

    def generate_rewrite_section_instructions(
        self,
        task: Any,
        source_document: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        source = self._unified_document(source_document)
        writing_context = self._unified_model(context, WritingContext)
        representation, source_title, source_sections = self._rewrite_source_sections(source)
        prompt = GENERATE_REWRITE_SECTION_INSTRUCTIONS_PROMPT.format(
            representation=representation,
            task_json=to_prompt_json(writing_task),
            source_title=source_title,
            source_sections_json=to_prompt_json(source_sections),
            context_json=to_prompt_json(writing_context),
        )
        instructions = self._call_llm_structured(prompt, SectionInstructionList)
        instructions = self._normalize_rewrite_section_instructions(
            instructions,
            representation,
            source_title,
            source_sections,
            writing_context,
            writing_task,
        )
        return self._save_artifacts(
            {'section_instructions': instructions},
            step_name='generate_rewrite_section_instructions',
            primary_key='section_instructions',
            context_key=None,
            summary='Generated section instructions for a complete document rewrite.',
            counts={'section_instructions': len(instructions.instructions)},
            extra={
                'representation': representation,
                'document_title': instructions.meta['document_title'],
            },
            artifact_meta={
                'context_id': writing_context.context_id,
                'source_document_id': source.document_id if isinstance(source, WriterDocument) else None,
            },
        ).model_dump()

    def generate_short_writing_plan(
        self,
        task: Any,
        context: Any,
        execution_results: Any = None,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        writing_context = self._unified_model(context, WritingContext)
        execution_data = self._unified_raw_data(execution_results)
        prompt = GENERATE_SHORT_WRITING_PLAN_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            context_json=to_prompt_json(writing_context),
            execution_results_json=to_prompt_json(execution_data),
        )
        plan = self._normalize_short_writing_plan(
            self._call_llm_structured(prompt, ShortWritingPlan),
            writing_task,
            writing_context,
            execution_data,
        )
        return self._save_artifacts(
            {'short_writing_plan': plan},
            step_name='generate_short_writing_plan',
            primary_key='short_writing_plan',
            context_key=None,
            summary='Generated a whole-document writing plan for a short article.',
            counts={
                'required_points': len(plan.required_points),
                'expected_blocks': len(plan.expected_blocks),
            },
            extra={
                'representation': plan.meta['representation'],
                'document_title': plan.section_title,
            },
            artifact_meta={
                'task_id': writing_task.task_id,
                'context_id': writing_context.context_id,
                'has_execution_results': execution_data is not None,
            },
        ).model_dump()

    def generate_short_visual_plan(
        self,
        task: Any,
        short_writing_plan: Any,
        context: Any,
    ) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        writing_plan = self._unified_model(short_writing_plan, ShortWritingPlan)
        writing_context = self._unified_model(context, WritingContext)
        prompt = GENERATE_SHORT_VISUAL_PLAN_PROMPT.format(
            task_json=to_prompt_json(writing_task),
            short_writing_plan_json=to_prompt_json(writing_plan),
            context_json=to_prompt_json(writing_context),
        )
        visual_plan = self._normalize_short_visual_plan(
            self._call_llm_structured(
                prompt,
                VisualPlan,
                trace_label='short_visual_plan',
            ),
        )
        return self._save_artifacts(
            {'visual_plan': visual_plan},
            step_name='generate_short_visual_plan',
            primary_key='visual_plan',
            context_key=None,
            summary='Generated a visual plan for a flat short article.',
            counts={'visual_instructions': len(visual_plan.instructions)},
            extra={
                'representation': writing_plan.meta.get('representation', 'markdown'),
                'document_root': True,
            },
            artifact_meta={
                'task_id': writing_task.task_id,
                'context_id': writing_context.context_id,
                'short_writing_plan_id': writing_plan.instruction_id,
            },
        ).model_dump()

    def generate_visual_plan(self, task: Any, outline: Any, context: Any) -> dict:
        writing_task = self._unified_model(task, WritingTask)
        writing_outline = self._unified_document(outline)
        writing_context = self._unified_model(context, WritingContext)
        visual_plan = VisualPlan()
        if isinstance(writing_outline, WriterDocument):
            prompt = GENERATE_VISUAL_PLAN_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                context_json=to_prompt_json(writing_context),
                outline_json=to_prompt_json(writing_outline),
            )
            visual_plan = self._normalize_visual_plan(
                self._call_llm_structured(prompt, VisualPlan), writing_outline)
        else:
            _, targets = get_markdown_outline_targets(writing_outline)
            target_payload = [
                {
                    'content_ref': {'heading_path': heading_path, 'occurrence': occurrence},
                    'section_title': heading_path[-1],
                    'outline_heading_level': level,
                }
                for level, heading_path, occurrence, _ in targets
            ]
            prompt = GENERATE_VISUAL_PLAN_MARKDOWN_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                context_json=to_prompt_json(writing_context),
                outline_json=writing_outline,
                target_sections_json=to_prompt_json(target_payload),
            )
            visual_plan = self._normalize_markdown_visual_plan(
                self._call_llm_structured(prompt, VisualPlan), targets)
        return self._save_artifacts(
            {'visual_plan': visual_plan},
            step_name='generate_visual_plan',
            primary_key='visual_plan',
            context_key=None,
            summary='Generated visual plan.',
            counts={'visual_instructions': len(visual_plan.instructions)},
            extra={'representation': 'ir' if isinstance(writing_outline, WriterDocument) else 'markdown'},
        ).model_dump()

    @classmethod
    def _normalize_short_writing_plan(
        cls,
        plan: ShortWritingPlan,
        task: WritingTask,
        context: WritingContext,
        execution_results: Any,
    ) -> ShortWritingPlan:
        title = str(
            task.target_document.title
            if task.target_document and task.target_document.title
            else plan.section_title
        ).strip()
        if not title:
            raise ValueError('Short writing plan requires a document title.')
        if not plan.section_goal.strip():
            raise ValueError('Short writing plan requires a section_goal.')
        if not plan.core_viewpoint.strip():
            raise ValueError('Short writing plan requires a core_viewpoint.')

        valid_reference_ids = {
            value
            for fact in context.facts
            for value in [fact.fact_id, *fact.source]
            if value
        }
        plan.instruction_id = f'{context.context_id}-short-writing-plan'
        plan.content_ref = ContentRef(document_root=True)
        plan.section_title = strip_heading_numbering(title)
        plan.section_goal = plan.section_goal.strip()
        plan.core_viewpoint = plan.core_viewpoint.strip()
        plan.references = [
            dict(reference)
            for reference in plan.references
            if isinstance(reference, dict) and reference.get('id') in valid_reference_ids
        ]
        plan.visual_needs = []
        cls._normalize_fact_constraints(plan, bool(context.facts))

        representation = cls._resolve_representation(task, None)
        plan.meta.update({
            'source': 'llm',
            'representation': representation,
            'document_title': plan.section_title,
            'context_id': context.context_id,
            'has_execution_results': execution_results is not None,
        })
        for key in ('target_chars', 'max_chars'):
            value = task.constraints.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                plan.meta[key] = value
        return plan

    @staticmethod
    def _normalize_short_visual_plan(visual_plan: VisualPlan) -> VisualPlan:
        for index, need in enumerate(visual_plan.instructions, start=1):
            ref = need.content_ref
            if ref.node_id or ref.heading_path or ref.placeholder_id or not ref.document_root:
                raise ValueError('Short visual plan must target document_root only.')
            need.purpose = need.purpose.strip()
            if not need.purpose:
                raise ValueError(f'Short visual need {index} has an empty purpose.')
            placement_hint = str(need.meta.get('placement_hint') or '').strip()
            if placement_hint:
                need.meta['placement_hint'] = placement_hint
            else:
                need.meta.pop('placement_hint', None)
            need.need_id = f'visual-document-{index}'
            need.content_ref = ContentRef(document_root=True)
        return visual_plan

    def generate_section_instructions(
        self,
        outline: Any,
        context: Any,
        visual_plan: Any = None,
        execution_results: Any = None,
        task: Any = None,
    ) -> dict:
        writing_outline = self._unified_document(outline)
        writing_context = self._unified_model(context, WritingContext)
        writing_task = self._unified_model(task, WritingTask) if task is not None else None
        execution_data = self._unified_raw_data(execution_results)
        writing_visual_plan = self._unified_optional_model(visual_plan, VisualPlan) or VisualPlan()

        if isinstance(writing_outline, WriterDocument):
            target_blocks = [block for block in writing_outline.blocks if block.type == 'heading']
            target_payload = [
                {
                    'content_ref': {'node_id': block.node_id},
                    'section_title': block.content,
                    'outline_content': block.model_dump(exclude_defaults=True),
                }
                for block in target_blocks
            ]
            outline_payload = writing_outline
            representation = 'ir'
        else:
            _, targets = get_markdown_outline_targets(writing_outline)
            target_payload = [
                {
                    'content_ref': {
                        'heading_path': heading_path,
                        'occurrence': occurrence,
                    },
                    'section_title': heading_path[-1],
                    'outline_heading_level': level,
                    'outline_body': body,
                }
                for level, heading_path, occurrence, body in targets
            ]
            outline_payload = writing_outline
            representation = 'markdown'

        deterministic = self._use_deterministic_short_instructions(
            writing_task, len(target_payload),
        )
        if deterministic:
            instruction_list = self._build_deterministic_section_instructions(
                target_payload, writing_context, writing_task,
            )
        else:
            prompt = GENERATE_SECTION_INSTRUCTIONS_PROMPT.format(
                task_json=to_prompt_json(writing_task),
                outline_json=to_prompt_json(outline_payload),
                target_outline_blocks_json=to_prompt_json(target_payload),
                context_json=to_prompt_json(writing_context),
                execution_results_json=to_prompt_json(execution_data),
                visual_plan_json=to_prompt_json(writing_visual_plan),
            )
            instruction_list = self._call_llm_structured(prompt, SectionInstructionList)
        if isinstance(writing_outline, WriterDocument):
            instruction_list = self._normalize_ir_section_instructions(
                instruction_list,
                writing_outline,
                target_blocks,
                writing_context,
                execution_data,
                writing_task,
                writing_visual_plan,
            )
            outline_id = writing_outline.document_id
        else:
            instruction_list = self._normalize_markdown_section_instructions(
                instruction_list,
                targets,
                writing_context,
                execution_data,
                writing_task,
                writing_visual_plan,
            )
            outline_id = instruction_list.outline_id
        if deterministic:
            instruction_list.meta['source'] = 'deterministic_short'

        result = self._save_artifacts(
            {'section_instructions': instruction_list},
            step_name='generate_section_instructions',
            primary_key='section_instructions',
            context_key=None,
            summary='Generated section writing instructions.',
            counts={
                'section_instructions': len(instruction_list.instructions),
            },
            extra={'representation': representation},
            artifact_meta={
                'outline_id': outline_id,
                'context_id': writing_context.context_id,
                'has_execution_results': execution_data is not None,
                'visual_instructions': len(writing_visual_plan.instructions),
            },
        )
        return result.model_dump()

    @staticmethod
    def _use_deterministic_short_instructions(
        task: WritingTask | None,
        section_count: int,
    ) -> bool:
        if task is None or task.task_type != 'write' or section_count <= 0:
            return False
        target_chars = task.constraints.get('target_chars')
        max_chars = task.constraints.get('max_chars')
        limits = [
            value for value in (target_chars, max_chars)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        ]
        return bool(limits) and max(limits) <= 1200

    @classmethod
    def _build_deterministic_section_instructions(
        cls,
        targets: List[Dict[str, Any]],
        context: WritingContext,
        task: WritingTask,
    ) -> SectionInstructionList:
        fact_constraints = [
            f'{fact.key}: {fact.value}'
            for fact in context.facts
            if str(fact.key).strip() and str(fact.value).strip()
        ]
        style_constraints: List[str] = []
        if context.style_profile is not None:
            for label, value in (
                ('tone', context.style_profile.tone),
                ('formality', context.style_profile.formality),
                ('audience', context.style_profile.audience),
            ):
                if str(value or '').strip():
                    style_constraints.append(f'{label}: {value}')
            style_constraints.extend(
                str(note).strip() for note in context.style_profile.notes
                if str(note).strip()
            )

        instructions = []
        titles = [str(target.get('section_title') or '').strip() for target in targets]
        for index, target in enumerate(targets):
            title = titles[index] or f'Section {index + 1}'
            points = cls._deterministic_outline_points(target)
            relations = []
            if index > 0:
                relations.append(f'承接上一节“{titles[index - 1]}”，避免重复其主体内容。')
            if index + 1 < len(targets):
                relations.append(f'为下一节“{titles[index + 1]}”保留清晰衔接。')
            instructions.append(SectionInstruction(
                instruction_id=f'short-section-{index + 1}',
                content_ref=ContentRef.model_validate(target['content_ref']),
                section_title=title,
                section_goal=(
                    f'围绕“{title}”完成本节，覆盖大纲要点并遵守全文约束。'
                ),
                required_points=points,
                fact_constraints=list(fact_constraints),
                style_constraints=list(style_constraints),
                relation_constraints=relations,
                expected_blocks=points,
                meta={
                    'target_chars': max(1, cls._deterministic_target_weight(target, points)),
                    'cross_references': [],
                },
            ))
        return SectionInstructionList(
            instructions=instructions,
            meta={'source': 'deterministic_short', 'task_id': task.task_id},
        )

    @staticmethod
    def _deterministic_outline_points(target: Dict[str, Any]) -> List[str]:
        body = target.get('outline_body')
        if body is None:
            outline_content = target.get('outline_content')
            if isinstance(outline_content, dict):
                values = []
                pending = list(outline_content.get('children') or [])
                while pending:
                    block = pending.pop(0)
                    if not isinstance(block, dict):
                        continue
                    content = str(block.get('content') or '').strip()
                    if content:
                        values.append(content)
                    pending[0:0] = list(block.get('children') or [])
                return values
            return []
        points = []
        for line in str(body).splitlines():
            value = re.sub(r'^\s*(?:#{3,6}\s+|[-*+]\s+|\d+[.)]\s+)', '', line).strip()
            if value:
                points.append(value)
        return points

    @staticmethod
    def _deterministic_target_weight(
        target: Dict[str, Any],
        points: List[str],
    ) -> int:
        body = target.get('outline_body')
        if body is not None:
            return len(re.sub(r'\s+', '', str(body))) or len(points) or 1
        return sum(len(re.sub(r'\s+', '', point)) for point in points) or 1

    @staticmethod
    def _normalize_visual_plan(visual_plan: VisualPlan, outline: WriterDocument) -> VisualPlan:
        node_ids = {block.node_id for block in outline.blocks if block.type == 'heading'}
        canonical_ids = WriterPlanningTools._ir_outline_node_ids_from_outline(outline)
        counts: Dict[str, int] = {}
        for need in visual_plan.instructions:
            ref = need.content_ref
            node_id = ref.node_id
            if (
                not node_id or ref.heading_path or ref.placeholder_id or ref.document_root
                or node_id not in node_ids
            ):
                raise ValueError('Visual plan must target a top-level outline node_id.')
            if not need.purpose.strip():
                raise ValueError(f'Visual need for {node_id!r} has an empty purpose.')
            canonical_id = canonical_ids[node_id]
            counts[canonical_id] = counts.get(canonical_id, 0) + 1
            need.need_id = f'visual-{canonical_id}-{counts[canonical_id]}'
            need.content_ref.node_id = canonical_id
        return visual_plan

    @staticmethod
    def _normalize_markdown_visual_plan(
        visual_plan: VisualPlan,
        targets: List[tuple[int, List[str], int, str]],
    ) -> VisualPlan:
        target_by_ref = {
            (tuple(heading_path), occurrence): (level, heading_path)
            for level, heading_path, occurrence, _ in targets
        }
        for index, need in enumerate(visual_plan.instructions, start=1):
            ref = need.content_ref
            if ref.node_id or ref.document_root or not ref.heading_path:
                raise ValueError('Markdown visual plan must target an H2 section via heading_path.')
            requested_heading_path = list(ref.heading_path)
            heading_path = requested_heading_path[:2]
            key = (tuple(heading_path), ref.occurrence)
            if key not in target_by_ref:
                suffix_matches = [
                    candidate
                    for candidate in target_by_ref
                    if candidate[1] == ref.occurrence
                    and tuple(candidate[0][-len(requested_heading_path):])
                    == tuple(requested_heading_path)
                ]
                if len(suffix_matches) != 1:
                    raise ValueError(
                        f'Visual plan targets unknown Markdown H2 section {key!r}.'
                    )
                key = suffix_matches[0]
                _, heading_path = target_by_ref[key]
            if not need.purpose.strip():
                raise ValueError(f'Visual need for {key!r} has an empty purpose.')
            ref.heading_path = list(heading_path)
            ref.occurrence = key[1]
            ref.placeholder_id = f'IMAGE-{index}'
            need.need_id = ref.placeholder_id
        return visual_plan

    @staticmethod
    def _resolve_representation(
        task: WritingTask,
        representation: Literal['ir', 'markdown'] | None,
    ) -> Literal['ir', 'markdown']:
        resolved = representation or task.output.get('representation') or 'markdown'
        if resolved not in {'ir', 'markdown'}:
            raise ValueError("representation must be 'ir' or 'markdown'.")
        return resolved

    @staticmethod
    def _normalize_markdown_outline(outline: str) -> str:
        lines: List[str] = []
        for line, in_fence in WriterPlanningTools._markdown_lines_with_fence_state(outline):
            if in_fence:
                lines.append(line)
                continue
            had_content = bool(line.strip())
            line = WriterPlanningTools._remove_outline_image_markup(line)
            if had_content and not line.strip():
                continue
            heading = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
            if heading:
                title = strip_heading_numbering(heading.group(2))
                lines.append(f'{heading.group(1)} {title}')
                continue
            lines.append(line)
        return WriterPlanningTools._materialize_markdown_outline_anchors(
            '\n'.join(lines).rstrip() + '\n',
        )

    @staticmethod
    def _remove_outline_image_markup(line: str) -> str:
        '''Keep generated outlines structural; visual placement has a separate owner.'''
        line = re.sub(
            r'(?<!\\)!\[(?:\\.|[^\]\\])*\]\('
            r'(?:\\.|[^()\\]|\([^()\n]*\))*\)',
            '',
            line,
        )
        line = re.sub(
            r'(?<!\\)!\[(?:\\.|[^\]\\])*\]\s*\[(?:\\.|[^\]\\])*\]',
            '',
            line,
        )
        line = re.sub(
            r'(?<!\\)!\[(?:\\.|[^\]\\])+\](?!\s*[\[(])',
            '',
            line,
        )
        return re.sub(r'<img\b[^>]*>', '', line, flags=re.IGNORECASE)

    @staticmethod
    def _materialize_markdown_outline_anchors(outline: str) -> str:
        _, targets = get_markdown_outline_targets(outline)
        target_ids = iter(
            WriterPlanningTools._markdown_outline_node_ids(targets).values()
        )
        lines: List[str] = []
        for line, in_fence in WriterPlanningTools._markdown_lines_with_fence_state(outline):
            if not in_fence and re.match(r'^##\s+.+?\s*$', line):
                lines.append(f'<a id="block-{next(target_ids)}"></a>')
            lines.append(line)
        return '\n'.join(lines).rstrip() + '\n'

    @staticmethod
    def _markdown_lines_with_fence_state(markdown: str):
        fence: Optional[str] = None
        for line in markdown.splitlines():
            fence_match = re.match(r'^\s*(```+|~~~+)', line)
            if fence_match:
                marker = fence_match.group(1)[0]
                yield line, True
                fence = marker if fence is None else None if fence == marker else fence
                continue
            yield line, fence is not None

    def _normalize_outline(
        self,
        outline: WriterDocument,
        task: WritingTask,
        context: WritingContext,
        profiles: List[ResourceProfile],
    ) -> WriterDocument:
        outline.stage = 'outline'
        if task.target_document and task.target_document.title:
            outline.title = task.target_document.title
        outline.ui_editable = False

        valid_reference_ids = {profile.resource_id for profile in profiles if profile.resource_id}
        for fact in context.facts:
            if fact.fact_id:
                valid_reference_ids.add(fact.fact_id)
            valid_reference_ids.update(source for source in fact.source if source)

        pending = [(block, 1) for block in reversed(outline.blocks)]
        while pending:
            block, level = pending.pop()
            block.stage = 'outline'
            block.numbering['level'] = level
            if block.type == 'heading':
                block.content = strip_heading_numbering(block.content)
            block.references = [
                reference
                for reference in block.references
                if reference.get('id') in valid_reference_ids
            ]
            pending.extend(
                (child, level + 1)
                for child in reversed(block.children)
            )

        outline.metadata.setdefault('source', 'llm')
        return outline

    def _normalize_ir_section_instructions(
        self,
        instruction_list: SectionInstructionList,
        outline: WriterDocument,
        target_blocks: List[WriterBlock],
        context: WritingContext,
        execution_results: Any,
        task: WritingTask | None,
        visual_plan: VisualPlan | None = None,
    ) -> SectionInstructionList:
        target_by_id = {block.node_id: block for block in target_blocks}
        node_id_by_original = self._ir_outline_node_ids_from_outline(outline)
        instruction_by_node_id: Dict[str, SectionInstruction] = {}

        for instruction in instruction_list.instructions:
            ref = instruction.content_ref
            node_id = ref.node_id
            if not node_id or ref.heading_path or ref.placeholder_id:
                raise ValueError('IR section instructions must contain only content_ref.node_id.')
            if node_id in instruction_by_node_id:
                raise ValueError(f'Duplicate section instruction for outline node {node_id!r}.')
            if node_id not in target_by_id:
                raise ValueError(f'Section instruction references unknown outline node {node_id!r}.')
            instruction_by_node_id[node_id] = instruction

        missing_node_ids = [
            block.node_id for block in target_blocks
            if block.node_id not in instruction_by_node_id
        ]
        if missing_node_ids:
            raise ValueError(
                'Missing section instructions for outline nodes: '
                + ', '.join(missing_node_ids)
            )

        needs_by_section: Dict[str, List[Any]] = {}
        for need in (visual_plan or VisualPlan()).instructions:
            needs_by_section.setdefault(need.content_ref.node_id, []).append(need)
        visual_targets = {
            need.need_id
            for needs in needs_by_section.values()
            for need in needs
        }
        instruction_list.outline_id = outline.document_id
        instruction_list.instruction_set_id = f'{outline.document_id}-section-instructions'
        instruction_list.instructions = [
            self._normalize_ir_section_instruction(
                instruction_by_node_id[block.node_id],
                block,
                outline,
                bool(context.facts),
                node_id_by_original,
                node_id_by_original[block.node_id],
                visual_targets,
                needs_by_section.get(node_id_by_original[block.node_id], []),
            )
            for block in target_blocks
        ]
        self._set_cross_reference_targets(
            instruction_list.instructions,
            [*node_id_by_original.values(), *visual_targets],
        )
        instruction_list.meta.update({
            'source': 'llm',
            'representation': 'ir',
            'outline_id': outline.document_id,
            'outline_title': outline.title,
            'context_id': context.context_id,
            'has_execution_results': execution_results is not None,
            'cross_reference_count': sum(
                len(instruction.meta.get('cross_references') or [])
                for instruction in instruction_list.instructions
            ),
        })
        return self._normalize_section_length_budgets(instruction_list, task)

    def _normalize_markdown_section_instructions(
        self,
        instruction_list: SectionInstructionList,
        targets: List[tuple[int, List[str], int, str]],
        context: WritingContext,
        execution_results: Any,
        task: WritingTask | None,
        visual_plan: VisualPlan,
    ) -> SectionInstructionList:
        target_by_ref = {
            (tuple(heading_path), occurrence): (level, heading_path)
            for level, heading_path, occurrence, _ in targets
        }
        instruction_by_ref: Dict[tuple[tuple[str, ...], int], SectionInstruction] = {}

        for instruction in instruction_list.instructions:
            ref = instruction.content_ref
            if ref.node_id or ref.placeholder_id or not ref.heading_path:
                raise ValueError(
                    'Markdown section instructions must contain only '
                    'content_ref.heading_path and occurrence.'
                )
            key = (tuple(ref.heading_path), ref.occurrence)
            if key in instruction_by_ref:
                raise ValueError(f'Duplicate section instruction for Markdown heading {key!r}.')
            if key not in target_by_ref:
                raise ValueError(f'Section instruction references unknown Markdown heading {key!r}.')
            instruction_by_ref[key] = instruction

        missing_refs = [key for key in target_by_ref if key not in instruction_by_ref]
        if missing_refs:
            raise ValueError(f'Missing section instructions for Markdown headings: {missing_refs!r}.')

        outline_id = f'{context.context_id}-outline-markdown'
        node_id_by_ref = self._markdown_outline_node_ids(targets)
        needs_by_ref = self._markdown_visual_needs_by_ref(visual_plan)
        visual_targets = {
            need.need_id
            for needs in needs_by_ref.values()
            for need in needs
        }
        normalized = []
        for level, heading_path, occurrence, _ in targets:
            key = (tuple(heading_path), occurrence)
            instruction = instruction_by_ref[key]
            self._validate_instruction(instruction, '/'.join(heading_path))
            instruction.section_title = strip_heading_numbering(heading_path[-1])
            instruction.references = []
            self._normalize_fact_constraints(instruction, bool(context.facts))
            instruction.meta.update({
                'representation': 'markdown',
                'outline_heading_level': level,
                'outline_id': outline_id,
                'outline_title': heading_path[0],
                'outline_node_id': node_id_by_ref[key],
            })
            instruction.visual_needs = []
            self._normalize_cross_references(
                instruction, node_id_by_ref, visual_targets,
            )
            section_needs = needs_by_ref.get(key, [])
            self._bind_visual_references(instruction, section_needs)
            self._drop_out_of_scope_visual_constraints(
                instruction, section_needs, visual_targets,
            )
            normalized.append(instruction)

        self._set_cross_reference_targets(
            normalized,
            [*node_id_by_ref.values(), *visual_targets],
        )

        instruction_list.outline_id = outline_id
        instruction_list.instruction_set_id = f'{outline_id}-section-instructions'
        instruction_list.instructions = normalized
        instruction_list.meta.update({
            'source': 'llm',
            'representation': 'markdown',
            'outline_id': outline_id,
            'outline_title': targets[0][1][0],
            'context_id': context.context_id,
            'has_execution_results': execution_results is not None,
            'cross_reference_count': sum(
                len(instruction.meta.get('cross_references') or [])
                for instruction in instruction_list.instructions
            ),
        })
        return self._normalize_section_length_budgets(instruction_list, task)

    @staticmethod
    def _set_cross_reference_targets(
        instructions: List[SectionInstruction],
        targets: List[str],
    ) -> None:
        normalized = list(dict.fromkeys(targets))
        for instruction in instructions:
            instruction.meta['cross_reference_targets'] = normalized

    @staticmethod
    def _markdown_outline_node_ids(
        targets: List[tuple[int, List[str], int, str]],
    ) -> Dict[tuple[tuple[str, ...], int], str]:
        ids: Dict[tuple[tuple[str, ...], int], str] = {}
        counters: List[int] = []
        for level, heading_path, occurrence, _ in targets:
            depth = level - 1
            counters = counters[:depth]
            counters.extend([0] * (depth - len(counters)))
            counters[-1] += 1
            node_id = 'sec-' + '-'.join(f'{value:03d}' for value in counters)
            ids[(tuple(heading_path), occurrence)] = node_id
        return ids

    @staticmethod
    def _markdown_visual_needs_by_ref(
        visual_plan: VisualPlan,
    ) -> Dict[tuple[tuple[str, ...], int], List[Any]]:
        needs: Dict[tuple[tuple[str, ...], int], List[Any]] = {}
        for need in visual_plan.instructions:
            key = (tuple(need.content_ref.heading_path), need.content_ref.occurrence)
            needs.setdefault(key, []).append(need)
        return needs

    @staticmethod
    def _bind_visual_references(
        instruction: SectionInstruction,
        needs: List[Any],
    ) -> None:
        references = [
            dict(item) for item in instruction.meta.get('cross_references') or []
            if isinstance(item, dict) and not item.get('must_create')
        ]
        references_by_target = {
            str(item.get('target')): item for item in references
        }
        for need in needs:
            reference = references_by_target.get(need.need_id)
            if reference is None:
                reference = {
                    'target': need.need_id,
                    'kind': 'image',
                    'guidance': need.purpose,
                }
                references.append(reference)
            reference.update({
                'required': bool(reference.get('required')) or need.required,
                'must_create': True,
                'caption': need.purpose or instruction.section_title,
            })
        instruction.meta['cross_references'] = references

    @staticmethod
    def _drop_out_of_scope_visual_constraints(
        instruction: SectionInstruction,
        section_needs: List[Any],
        visual_targets: set[str],
    ) -> None:
        if not visual_targets or section_needs:
            return
        instruction.fact_constraints = [
            constraint for constraint in instruction.fact_constraints
            if not (
                _IMAGE_RESOURCE_FACT_PATTERN.search(constraint)
                and _VISUAL_DIRECTIVE_PATTERN.search(constraint)
            )
        ]

    @classmethod
    def _normalize_cross_references(
        cls,
        instruction: SectionInstruction,
        section_ids: Dict[tuple[tuple[str, ...], int], str] | Dict[str, str],
        visual_targets: set[str] | None = None,
    ) -> None:
        raw_references = instruction.meta.get('cross_references')
        if raw_references is None:
            instruction.meta['cross_references'] = []
            return
        if not isinstance(raw_references, list):
            raise ValueError(
                f'Section instruction {instruction.instruction_id!r} '
                'cross_references must be a list.'
            )

        normalized: List[Dict[str, Any]] = []
        for item in raw_references:
            if not isinstance(item, dict):
                raise ValueError('Each cross-reference must be an object.')
            must_create = bool(item.get('must_create'))
            if must_create:
                raise ValueError('Created image cross-references are owned by the visual plan.')
            target_ref = item.get('target_ref')
            target = cls._resolve_cross_reference_target(
                target_ref, section_ids, visual_targets or set(),
            )
            kind = (
                'image'
                if target in (visual_targets or set())
                else str(item.get('kind') or 'section')
            )
            if kind not in {'section', 'image'}:
                raise ValueError(f'Unsupported cross-reference kind {kind!r}.')

            normalized.append({
                'target': target,
                'kind': kind,
                'required': bool(item.get('required', True)),
                'must_create': False,
                'caption': str(item.get('caption') or '').strip(),
                'guidance': str(item.get('guidance') or '').strip(),
            })
        instruction.meta['cross_references'] = normalized

    @staticmethod
    def _resolve_cross_reference_target(
        target_ref: Any,
        section_ids: Dict[tuple[tuple[str, ...], int], str] | Dict[str, str],
        visual_targets: set[str],
    ) -> str:
        if isinstance(target_ref, dict) and isinstance(section_ids, dict):
            node_id = target_ref.get('node_id')
            if isinstance(node_id, str):
                mapped = section_ids.get(node_id)
                if mapped is not None:
                    return mapped
                if node_id in visual_targets:
                    return node_id
            heading_path = target_ref.get('heading_path')
            occurrence = int(target_ref.get('occurrence') or 1)
            if isinstance(heading_path, list):
                target = section_ids.get((tuple(heading_path), occurrence))
                if target is not None:
                    return target
        raise ValueError(f'Unknown cross-reference target: {target_ref!r}')

    @staticmethod
    def _rewrite_source_sections(
        source: WriterDocument | str,
    ) -> tuple[str, str, List[Dict[str, Any]]]:
        if isinstance(source, WriterDocument):
            if not source.blocks:
                raise ValueError('WriterDocument rewrite source must contain at least one block.')
            grouped_blocks: List[List[WriterBlock]] = []
            for block in source.blocks:
                if block.type == 'heading' or not grouped_blocks:
                    grouped_blocks.append([block])
                else:
                    grouped_blocks[-1].append(block)
            return 'ir', source.title or 'Rewrite', [
                {
                    'source_ref': {'node_id': blocks[0].node_id},
                    'section_title': (
                        blocks[0].content or source.title or f'Section {index}'
                    ),
                    'content': '\n\n'.join(
                        render_block_markdown(block, level=2).strip()
                        for block in blocks
                    ),
                    'format': {
                        'type': blocks[0].type,
                        'numbering': blocks[0].numbering,
                        'top_level_types': [block.type for block in blocks],
                        'child_types': [
                            child.type for block in blocks for child in block.children
                        ],
                    },
                }
                for index, blocks in enumerate(grouped_blocks, start=1)
            ]

        parsed = parse_markdown_sections(source)
        title = next(
            (heading_path[-1] for level, heading_path, _, _ in parsed if level == 1),
            'Rewrite',
        )
        sections = [section for section in parsed if section[0] == 2]
        if not sections:
            return 'markdown', title, [{
                'source_ref': {'document_root': True},
                'section_title': title,
                'content': source,
                'format': {'heading_level': 2},
            }]
        return 'markdown', title, [
            {
                'source_ref': {
                    'heading_path': heading_path,
                    'occurrence': occurrence,
                },
                'section_title': heading_path[-1],
                'content': body,
                'format': {'heading_level': level},
            }
            for level, heading_path, occurrence, body in sections
        ]

    @classmethod
    def _normalize_rewrite_section_instructions(
        cls,
        instruction_list: SectionInstructionList,
        representation: str,
        source_title: str,
        source_sections: List[Dict[str, Any]],
        context: WritingContext,
        task: WritingTask,
    ) -> SectionInstructionList:
        if not instruction_list.instructions:
            raise ValueError('Rewrite section instructions must contain at least one section.')
        document_title = str(instruction_list.meta.get('document_title') or source_title).strip()
        if not document_title:
            raise ValueError('Rewrite section instructions require a document title.')
        source_by_ref = {
            cls._rewrite_source_ref_key(section['source_ref']): section
            for section in source_sections
        }
        seen_titles: Dict[str, int] = {}
        for index, instruction in enumerate(instruction_list.instructions, start=1):
            cls._validate_instruction(instruction, f'rewrite-section-{index}')
            instruction.instruction_id = instruction.instruction_id.strip() or f'rewrite-instruction-{index}'
            instruction.section_title = instruction.section_title.strip()
            seen_titles[instruction.section_title] = seen_titles.get(instruction.section_title, 0) + 1
            selected_sections = []
            selected_refs = []
            for reference in instruction.references:
                source_section = source_by_ref.get(cls._rewrite_source_ref_key(reference))
                if source_section is not None and source_section not in selected_sections:
                    selected_sections.append(source_section)
                    selected_refs.append(dict(source_section['source_ref']))
            if not selected_sections:
                source_section = source_sections[min(index - 1, len(source_sections) - 1)]
                selected_sections = [source_section]
                selected_refs = [dict(source_section['source_ref'])]
            if representation == 'ir':
                instruction.content_ref = ContentRef(node_id=f'rewrite-section-{index}')
            else:
                instruction.content_ref = ContentRef(
                    heading_path=[document_title, instruction.section_title],
                    occurrence=seen_titles[instruction.section_title],
                )
            instruction.references = []
            cls._normalize_fact_constraints(instruction, bool(context.facts))
            instruction.meta.update({
                'representation': representation,
                'rewrite': True,
                'document_title': document_title,
                'outline_title': document_title,
                'source_content_refs': selected_refs,
                'source_content': '\n\n'.join(
                    str(section.get('content') or '').strip()
                    for section in selected_sections
                    if str(section.get('content') or '').strip()
                ),
                'source_format': [section.get('format') or {} for section in selected_sections],
            })
            if representation == 'markdown':
                instruction.meta['outline_heading_level'] = 2
            else:
                instruction.meta['outline_node_level'] = 1
        instruction_list.instruction_set_id = f'{context.context_id}-rewrite-section-instructions'
        instruction_list.outline_id = None
        instruction_list.meta.update({
            'source': 'llm',
            'representation': representation,
            'rewrite': True,
            'document_title': document_title,
            'context_id': context.context_id,
        })
        return cls._normalize_section_length_budgets(instruction_list, task)

    @classmethod
    def _normalize_section_length_budgets(
        cls,
        instruction_list: SectionInstructionList,
        task: WritingTask | None,
    ) -> SectionInstructionList:
        if task is None:
            return instruction_list
        target_chars = task.constraints.get('target_chars')
        max_chars = task.constraints.get('max_chars')
        if not isinstance(target_chars, int) or not isinstance(max_chars, int):
            return instruction_list

        weights = []
        for instruction in instruction_list.instructions:
            value = instruction.meta.get('target_chars')
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    'Every section instruction must provide a positive meta.target_chars '
                    'when the writing task has a document length budget.'
                )
            weights.append(float(value))

        targets = cls._allocate_by_weight(target_chars, weights)
        headroom = cls._allocate_by_weight(max_chars - target_chars, weights)
        for instruction, target, extra in zip(
            instruction_list.instructions, targets, headroom,
        ):
            instruction.meta['target_chars'] = target
            instruction.meta['max_chars'] = target + extra
        instruction_list.meta.update({
            'target_chars': target_chars,
            'max_chars': max_chars,
        })
        return instruction_list

    @staticmethod
    def _allocate_by_weight(total: int, weights: List[float]) -> List[int]:
        if total <= 0:
            return [0] * len(weights)
        weight_sum = sum(weights)
        raw = [total * weight / weight_sum for weight in weights]
        allocated = [int(value) for value in raw]
        remainder = total - sum(allocated)
        order = sorted(
            range(len(weights)),
            key=lambda index: raw[index] - allocated[index],
            reverse=True,
        )
        for index in order[:remainder]:
            allocated[index] += 1
        return allocated

    @staticmethod
    def _rewrite_source_ref_key(reference: Dict[str, Any]) -> tuple[Any, ...]:
        if reference.get('node_id'):
            return 'node_id', str(reference['node_id'])
        if reference.get('heading_path'):
            return (
                'heading_path',
                tuple(reference['heading_path']),
                int(reference.get('occurrence') or 1),
            )
        if reference.get('document_root'):
            return 'document_root',
        return 'invalid',

    def _normalize_ir_section_instruction(
        self,
        instruction: SectionInstruction,
        block: WriterBlock,
        outline: WriterDocument,
        has_available_facts: bool,
        node_id_by_original: Dict[str, str],
        outline_node_id: str,
        visual_targets: set[str],
        section_needs: List[Any],
    ) -> SectionInstruction:
        self._validate_instruction(instruction, block.node_id)
        instruction.section_title = block.content
        instruction.content_ref.node_id = outline_node_id
        instruction.references = [dict(reference) for reference in block.references]
        instruction.visual_needs = []
        self._normalize_fact_constraints(instruction, has_available_facts)
        instruction.meta.update({
            'representation': 'ir',
            'outline_node_level': block.numbering.get('level'),
            'outline_id': outline.document_id,
            'outline_title': outline.title,
            'outline_node_id': outline_node_id,
        })
        self._normalize_cross_references(
            instruction, node_id_by_original, visual_targets,
        )
        self._bind_visual_references(instruction, section_needs)
        self._drop_out_of_scope_visual_constraints(
            instruction, section_needs, visual_targets,
        )
        return instruction

    @staticmethod
    def _ir_outline_node_ids_from_outline(outline: WriterDocument) -> Dict[str, str]:
        ids: Dict[str, str] = {}

        def walk(blocks: List[WriterBlock], counters: List[int]) -> None:
            for block in blocks:
                if block.type == 'heading':
                    level = int(block.numbering.get('level') or 1)
                    del counters[level:]
                    if len(counters) < level:
                        counters.extend([0] * (level - len(counters)))
                    counters[level - 1] += 1
                    node_id = 'sec-' + '-'.join(
                        f'{value:03d}' for value in counters
                    )
                    ids[block.node_id] = node_id
                    walk(block.children, counters)
                else:
                    walk(block.children, counters)

        walk(outline.blocks, [])
        return ids

    @staticmethod
    def _validate_instruction(instruction: SectionInstruction, target: str) -> None:
        if not instruction.instruction_id.strip():
            raise ValueError(f'Section instruction for {target!r} has an empty instruction_id.')
        if not instruction.section_goal.strip():
            raise ValueError(f'Section instruction for {target!r} has an empty section_goal.')

    @staticmethod
    def _normalize_fact_constraints(
        instruction: SectionInstruction | ShortWritingPlan,
        has_available_facts: bool,
    ) -> None:
        if not has_available_facts:
            instruction.fact_constraints = []
