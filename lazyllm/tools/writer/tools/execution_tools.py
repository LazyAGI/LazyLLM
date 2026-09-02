from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import WriterToolBase
from ..data_models.context import WritingContext
from ..data_models.writer_ir import WritingSubTask, WriterDocument
from ..utils import (
    apply_markdown_outline_instructions,
    make_markdown_tool_result,
    parse_document_markdown,
    to_prompt_json,
)


class _WritingSubTaskResolution(BaseModel):
    result_summary: str
    result_references: List[Dict[str, Any]] = Field(default_factory=list)


class WriterExecutionTools(WriterToolBase):
    '''Execute document-level writing subtasks without creating a workflow step.'''

    __public_apis__ = ['execute_writing_subtasks']

    def execute_writing_subtasks(
        self,
        outline: Any,
        context: Any,
        *,
        max_retries: int = 1,
        retry_failed: bool = False,
        on_progress: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        retrieve: Optional[Callable[[str], Any]] = None,
    ) -> dict:
        if max_retries < 0:
            raise ValueError('max_retries must be non-negative.')
        source = self._unified_document(outline)
        writing_context = self._unified_model(context, WritingContext)
        is_markdown = isinstance(source, str)
        document = parse_document_markdown(
            source,
            document_id=f'{writing_context.context_id}-outline-markdown',
            stage='outline',
        ) if is_markdown else source
        if not isinstance(document, WriterDocument):
            raise TypeError('execute_writing_subtasks requires Markdown or WriterDocument.')

        counts = {'completed': 0, 'failed': 0, 'skipped': 0}
        self._publish_progress(document, on_progress)
        for block in document.iter_blocks():
            for subtask in block.subtasks:
                if subtask.status == 'completed' or (
                    subtask.status == 'failed' and not retry_failed
                ):
                    counts['skipped'] += 1
                    continue
                self._execute_subtask(
                    subtask, block.model_dump(exclude={'subtasks'}), writing_context,
                    max_retries, lambda: self._publish_progress(document, on_progress), retrieve,
                )
                counts['completed' if subtask.status == 'completed' else 'failed'] += 1

        result_meta = {
            'step_name': 'execute_writing_subtasks',
            'artifact_key': 'outline',
            'summary': 'Executed writing subtasks.',
            'counts': counts,
            'extra': {'max_retries': max_retries, 'retry_failed': retry_failed},
        }
        if is_markdown:
            completed = apply_markdown_outline_instructions(source, document)
            path = self._write_markdown_artifact('outline.md', completed)
            return make_markdown_tool_result(path=path, **result_meta).model_dump()
        result = self._save_artifacts(
            {'outline': document},
            primary_key='outline',
            context_key=None,
            step_name=result_meta['step_name'],
            summary=result_meta['summary'],
            counts=result_meta['counts'],
            extra=result_meta['extra'],
        )
        return result.model_dump()

    @staticmethod
    def _publish_progress(
        document: WriterDocument,
        callback: Optional[Callable[[List[Dict[str, Any]]], None]],
    ) -> None:
        if callback is None:
            return
        callback([
            {
                **subtask.model_dump(),
                'node_title': block.content,
            }
            for block in document.iter_blocks()
            for subtask in block.subtasks
        ])

    def _execute_subtask(
        self,
        subtask: WritingSubTask,
        block: Dict[str, Any],
        context: WritingContext,
        max_retries: int,
        publish: Callable[[], None],
        retrieve: Optional[Callable[[str], Any]],
    ) -> None:
        retries_this_run = 0
        while True:
            subtask.status = 'running'
            publish()
            try:
                retrieval_result = None
                if subtask.subtask_type == 'retrieve':
                    if retrieve is None:
                        raise RuntimeError('No knowledge retrieval handler is configured.')
                    self._record_tool(subtask, 'kb_search')
                    publish()
                    retrieval_result = retrieve(subtask.question)
                self._record_tool(subtask, 'llm')
                publish()
                resolution = self._call_llm_structured(
                    self._subtask_prompt(subtask, block, context, retrieval_result),
                    _WritingSubTaskResolution,
                )
                subtask.result_summary = resolution.result_summary.strip()
                subtask.result_references = resolution.result_references
                subtask.status = 'completed'
                publish()
                return
            except Exception as exc:
                if retries_this_run >= max_retries:
                    subtask.status = 'failed'
                    subtask.result_summary = f'{type(exc).__name__}: {exc}'
                    publish()
                    return
                retries_this_run += 1
                subtask.retry_count += 1
                subtask.status = 'retrying'
                publish()

    @staticmethod
    def _record_tool(subtask: WritingSubTask, tool_name: str) -> None:
        if tool_name not in subtask.tools_used:
            subtask.tools_used.append(tool_name)

    @staticmethod
    def _subtask_prompt(
        subtask: WritingSubTask,
        block: Dict[str, Any],
        context: WritingContext,
        retrieval_result: Any = None,
    ) -> str:
        return '''Resolve one writing subtask using the supplied writing context and outline node.

Return an object with result_summary and result_references. result_summary must be concise,
specific, and usable by the later drafting step. result_references contains only references
actually available in the supplied material. Do not invent sources or facts.

Subtask:
{subtask_json}

Outline node:
{block_json}

Writing context:
{context_json}

Knowledge retrieval result (null unless subtask_type is retrieve):
{retrieval_result_json}
'''.format(
            subtask_json=to_prompt_json(subtask),
            block_json=to_prompt_json(block),
            context_json=to_prompt_json(context),
            retrieval_result_json=to_prompt_json(retrieval_result),
        )
