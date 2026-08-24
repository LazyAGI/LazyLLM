from __future__ import annotations
import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar
from pydantic import BaseModel
from lazyllm.components.formatter import JsonFormatter
from lazyllm.module import ModuleBase
from lazyllm.thirdparty import json_repair
from lazyllm.tracing import finish_span, set_span_attributes, set_span_error, set_span_output, start_span
from ..data_models.planning import SectionInstructionList
from ..data_models.writer_ir import WriterBlock, WriterDocument
from ..prompts.structured_output import STRUCTURED_OUTPUT_SYSTEM_PROMPT
from ..utils.artifact import ToolResult, load_artifact_json, save_artifact_json

T = TypeVar('T', bound=BaseModel)


def _writer_structured_llm_attempt_trace():
    """Name the diagnostic span emitted around selected structured Writer calls."""


def _strip_leading_think_blocks(text: str) -> str:
    cleaned = text
    pattern = re.compile(r'^\s*<think\b[^>]*>.*?</think>\s*', re.IGNORECASE | re.DOTALL)
    while True:
        stripped = pattern.sub('', cleaned, count=1)
        if stripped == cleaned:
            return cleaned
        cleaned = stripped


class _WriterJsonFormatter(JsonFormatter):
    def _load(self, msg: str):
        cleaned = _strip_leading_think_blocks(msg)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # JsonFormatter rejects unbalanced braces before reaching json_repair.
            # Writer structured output is schema-validated immediately afterwards,
            # so repair syntax here without weakening semantic validation.
            repaired = json_repair.loads(cleaned)
            if isinstance(repaired, (dict, list)):
                return repaired
            return super()._load(cleaned)


class WriterToolBase(ModuleBase):
    def __init__(
        self,
        llm=None,
        artifact_store: Optional[str] = None,
        adapters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.artifact_store = artifact_store or ''
        self.adapters = adapters or {}

    def _load_artifact(
        self,
        path: str,
        model_class: Optional[Type[T]] = None,
        *,
        expected_schema_name: Optional[str] = None,
        validate_schema: bool = True,
    ) -> Any:
        if model_class is None and expected_schema_name is None:
            return load_artifact_json(path, validate_schema=False)
        return load_artifact_json(
            path,
            model_class,
            expected_schema_name=expected_schema_name,
            validate_schema=validate_schema,
        )

    def _unified_model(self, value: Any, model_class: Type[T]) -> T:
        if isinstance(value, model_class):
            return value
        if isinstance(value, str):
            return self._load_artifact(value, model_class)
        if isinstance(value, dict):
            return model_class.model_validate(value)
        raise TypeError(f'Expected {model_class.__name__}, dict, or artifact path, got {type(value).__name__}.')

    def _unified_optional_model(self, value: Any, model_class: Type[T]) -> Optional[T]:
        if value is None:
            return None
        return self._unified_model(value, model_class)

    def _unified_section(self, value: Any) -> WriterBlock | str:
        if isinstance(value, WriterBlock):
            return value
        if isinstance(value, dict):
            return WriterBlock.model_validate(value)
        if isinstance(value, str):
            if os.path.isfile(value):
                if value.lower().endswith(('.md', '.markdown')):
                    with open(value, 'r', encoding='utf-8') as stream:
                        return stream.read()
                return self._unified_model(value, WriterBlock)
            return value
        raise TypeError('value must be WriterBlock, Markdown text, or an artifact path.')

    def _unified_document(self, value: Any) -> WriterDocument | str:
        if isinstance(value, WriterDocument):
            return value
        if isinstance(value, dict):
            return WriterDocument.model_validate(value)
        if isinstance(value, str):
            if os.path.isfile(value):
                if value.lower().endswith(('.md', '.markdown')):
                    with open(value, 'r', encoding='utf-8') as stream:
                        return stream.read()
                return self._unified_model(value, WriterDocument)
            return value
        raise TypeError('value must be WriterDocument, Markdown text, or an artifact path.')

    def _unified_models(self, value: Any, model_class: Type[T]) -> List[T]:
        if value is None:
            return []
        if isinstance(value, str):
            value = self._load_artifact(value, validate_schema=False)
        if isinstance(value, Iterable) and not isinstance(value, (dict, bytes, str)):
            return [self._unified_model(item, model_class) for item in value]
        raise TypeError(f'Expected a list of {model_class.__name__}, or an artifact path.')

    def _unified_raw_data(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return self._load_artifact(value, validate_schema=False)
        if isinstance(value, BaseModel):
            return value.model_dump(exclude_defaults=True)
        return value

    def _write_single_artifact(
        self,
        artifact: Any,
        filename: str,
        *,
        artifact_key: Optional[str] = None,
        schema_name: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.artifact_store:
            raise ValueError('artifact_store is not set')
        path = os.path.join(self.artifact_store, filename)
        return save_artifact_json(
            artifact,
            path,
            schema_name=schema_name or self._artifact_schema_name(artifact, artifact_key),
            created_by=type(self).__name__,
            extra_meta=extra_meta,
        )

    def _write_markdown_artifact(self, filename: str, content: str) -> str:
        if not self.artifact_store:
            raise ValueError('artifact_store is not set')
        path = os.path.join(self.artifact_store, filename)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write(content)
        return os.path.abspath(path)

    def _save_artifacts(
        self,
        artifacts: Dict[str, Any],
        *,
        primary_key: Optional[str] = None,
        context_key: Optional[str] = 'writing_context',
        summary: str = '',
        step_name: Optional[str] = None,
        status: str = 'success',
        warnings: Optional[List[str]] = None,
        counts: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        artifact_meta: Optional[Dict[str, Any]] = None,
        artifact_filenames: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        if not artifacts:
            raise ValueError('artifacts must contain at least one artifact.')

        resolved_primary_key = primary_key or next(iter(artifacts))
        if resolved_primary_key not in artifacts:
            raise ValueError(f'primary_key {resolved_primary_key!r} is not present in artifacts.')

        artifact_paths: Dict[str, str] = {}
        schema_names: Dict[str, str] = {}

        for artifact_key, artifact in artifacts.items():
            schema_name = self._artifact_schema_name(artifact, artifact_key)
            filename = (
                artifact_filenames.get(artifact_key)
                if artifact_filenames and artifact_key in artifact_filenames
                else self._default_artifact_filename(artifact_key, artifact)
            )
            artifact_extra_meta = {
                'step_name': step_name or type(self).__name__,
                'artifact_key': artifact_key,
                'primary_key': resolved_primary_key,
                'status': status,
            }
            if artifact_meta:
                artifact_extra_meta.update(artifact_meta)
            artifact_paths[artifact_key] = self._write_single_artifact(
                artifact,
                filename,
                artifact_key=artifact_key,
                schema_name=schema_name,
                extra_meta=artifact_extra_meta,
            )
            schema_names[artifact_key] = schema_name

        metadata = {
            'step_name': step_name or type(self).__name__,
            'artifact_key': resolved_primary_key,
            'artifact_paths': artifact_paths,
            'schema_names': schema_names,
            'counts': counts or {},
            'status': status,
            'warnings': warnings or [],
            'extra': extra or {},
        }

        return ToolResult(
            artifact_path=artifact_paths[resolved_primary_key],
            context_path=artifact_paths.get(context_key or ''),
            summary=summary,
            metadata=metadata,
        )

    def _default_artifact_filename(self, artifact_key: str, artifact: Any) -> str:
        if isinstance(artifact, (WriterDocument, WriterBlock)):
            return f'{artifact_key}_ir.lmd'
        return f'{artifact_key}.json'

    def _artifact_schema_name(self, artifact: Any, artifact_key: Optional[str] = None) -> str:
        if isinstance(artifact, BaseModel):
            cls = type(artifact)
            module = cls.__module__ or ''
            return f'{module}.{cls.__qualname__}'
        if artifact_key:
            return f'lazyllm.tools.writer.artifacts.{artifact_key}'
        cls = type(artifact)
        module = cls.__module__ or ''
        return f'{module}.{cls.__qualname__}'

    def _call_llm_structured(
        self,
        prompt: str,
        schema: Type[T],
        stream_output: Any = False,
        *,
        trace_label: Optional[str] = None,
    ) -> T:
        if self.llm is None:
            raise ValueError('llm is not set')

        system_prompt = self._structured_output_prompt(schema)
        model = self._build_structured_llm(
            system_prompt,
            stream_output=stream_output,
            apply_formatter=not bool(trace_label),
        )
        attempts = 1 if stream_output else 2
        for attempt in range(attempts):
            span = None
            if trace_label:
                span = start_span(
                    span_kind='callable',
                    target=_writer_structured_llm_attempt_trace,
                    args=(),
                    kwargs={
                        'trace_label': trace_label,
                        'schema': schema.__name__,
                        'attempt': attempt + 1,
                    },
                )
            failure_stage = 'model_call'
            try:
                response = model(prompt)
                if span:
                    set_span_output(span, response)
                    set_span_attributes(span, {
                        'writer.structured.trace_label': trace_label,
                        'writer.structured.schema': schema.__name__,
                        'writer.structured.attempt': attempt + 1,
                        'writer.structured.response_type': type(response).__name__,
                        'writer.structured.response_empty': (
                            not response.strip() if isinstance(response, str)
                            else len(response) == 0 if isinstance(response, (dict, list))
                            else response is None
                        ),
                    })
                failure_stage = 'validation'
                result = self._validate_structured_response(response, schema)
                if span:
                    instructions = getattr(result, 'instructions', None)
                    set_span_attributes(span, {
                        'writer.structured.validation': 'success',
                        'writer.structured.instruction_count': (
                            len(instructions) if isinstance(instructions, list) else -1
                        ),
                    })
                return result
            except Exception as exc:
                if span:
                    set_span_attributes(span, {
                        'writer.structured.validation': 'error',
                        'writer.structured.failure_stage': failure_stage,
                    })
                    set_span_error(span, exc)
                if attempt + 1 >= attempts:
                    raise
                # Keep transient calls and malformed structured output local to the
                # Writer tool instead of repeating completed Workspace steps.
            finally:
                if span:
                    finish_span(span)
        raise RuntimeError('Structured Writer call exhausted without a result.')

    def _call_llm_text(self, prompt: str, stream_output: Any = False) -> str:
        if self.llm is None:
            raise ValueError('llm is not set')
        model = self.llm
        if hasattr(model, 'share'):
            try:
                model = model.share(stream=stream_output)
            except TypeError as exc:
                if stream_output:
                    raise TypeError('llm.share() must accept stream for text streaming.') from exc
                model = model.share()
        elif stream_output:
            raise TypeError('llm must support share(stream=...) for text streaming.')
        response = model(prompt)
        text = response if isinstance(response, str) else str(response)
        return self._strip_leading_think_blocks(text)

    @staticmethod
    def _strip_leading_think_blocks(text: str) -> str:
        '''Remove provider reasoning blocks without touching document content.'''
        return _strip_leading_think_blocks(text)

    def _structured_output_prompt(self, schema: Type[BaseModel]) -> str:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        return STRUCTURED_OUTPUT_SYSTEM_PROMPT.format(schema_name=schema.__name__, schema_json=schema_json)

    def _build_structured_llm(
        self,
        system_prompt: str,
        stream_output: Any = False,
        *,
        apply_formatter: bool = True,
    ) -> Any:
        model = self.llm
        if hasattr(model, 'share'):
            try:
                model = model.share(stream=stream_output)
            except TypeError as exc:
                if stream_output:
                    raise TypeError('llm.share() must accept stream for structured streaming.') from exc
                model = model.share()
        elif stream_output:
            raise TypeError('llm must support share(stream=...) for structured streaming.')
        if hasattr(model, 'prompt'):
            model = model.prompt(system_prompt)
        if apply_formatter and hasattr(model, 'formatter'):
            model = model.formatter(_WriterJsonFormatter())
        return model

    @classmethod
    def _inherit_document_stage(cls, blocks: Any, document_stage: str) -> None:
        if not isinstance(blocks, list):
            return
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block.setdefault('stage', document_stage)
            cls._inherit_document_stage(block.get('children'), document_stage)

    @classmethod
    def _prepare_structured_candidate(cls, candidate: Any, schema: Type[T]) -> Any:
        if schema is SectionInstructionList and isinstance(candidate, dict):
            normalized = deepcopy(candidate)
            instructions = normalized.get('instructions')
            if isinstance(instructions, dict):
                instructions = list(instructions.values())
            if isinstance(instructions, list):
                normalized['instructions'] = [
                    cls._normalize_section_instruction_candidate(item)
                    for item in instructions
                    if isinstance(item, dict)
                ]
            if not isinstance(normalized.get('meta'), dict):
                normalized['meta'] = {}
            return normalized
        if schema is not WriterDocument or not isinstance(candidate, dict):
            return candidate
        document_stage = candidate.get('stage')
        if not document_stage:
            return candidate
        normalized = deepcopy(candidate)
        cls._inherit_document_stage(normalized.get('blocks'), document_stage)
        return normalized

    @staticmethod
    def _normalize_section_instruction_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        normalized = deepcopy(candidate)
        list_fields = (
            'required_points', 'references', 'fact_constraints', 'style_constraints',
            'relation_constraints', 'visual_needs', 'expected_blocks',
            'pending_subtasks', 'revision_notes',
        )
        for field in list_fields:
            value = normalized.get(field)
            if value is None:
                normalized[field] = []
            elif isinstance(value, dict):
                if field == 'relation_constraints':
                    normalized[field] = [
                        f'{key}: {item}' for key, item in value.items()
                        if str(item).strip()
                    ]
                else:
                    normalized[field] = list(value.values())
            elif not isinstance(value, list):
                normalized[field] = [value]
        if not isinstance(normalized.get('meta'), dict):
            normalized['meta'] = {}
        content_ref = normalized.get('content_ref')
        if isinstance(content_ref, dict):
            occurrence = content_ref.get('occurrence')
            if isinstance(occurrence, str) and occurrence.isdigit():
                content_ref['occurrence'] = int(occurrence)
        return normalized

    @staticmethod
    def _parse_structured_response(response: Any, schema: Type[T]) -> Any:
        if not isinstance(response, str):
            return response
        try:
            return _WriterJsonFormatter()(response)
        except Exception as exc:
            raise ValueError(
                f'Failed to parse LLM output as JSON for {schema.__name__}. '
                f'Response: {response!r}'
            ) from exc

    @classmethod
    def _select_structured_candidate(
        cls,
        parsed: List[Any],
        schema: Type[T],
        response: Any,
    ) -> Optional[T]:
        candidates = []
        for item in parsed:
            try:
                candidates.append(schema.model_validate(cls._prepare_structured_candidate(item, schema)))
            except Exception:
                continue
        unique_candidates = {
            candidate.model_dump_json(exclude_defaults=True): candidate
            for candidate in candidates
        }
        if len(unique_candidates) == 1:
            return next(iter(unique_candidates.values()))
        if len(unique_candidates) > 1:
            raise ValueError(
                f'Failed to select one unambiguous {schema.__name__} from LLM output. '
                f'Response: {response!r}'
            )
        return None

    def _validate_structured_response(self, response: Any, schema: Type[T]) -> T:
        parsed = self._parse_structured_response(response, schema)
        if isinstance(parsed, schema):
            return parsed
        direct_error = None
        if isinstance(parsed, (dict, list)):
            try:
                return schema.model_validate(self._prepare_structured_candidate(parsed, schema))
            except Exception as exc:
                direct_error = exc

        if isinstance(parsed, list):
            candidate = self._select_structured_candidate(parsed, schema, response)
            if candidate is not None:
                return candidate

        if direct_error is not None:
            raise ValueError(
                f'Failed to validate LLM output as {schema.__name__}. '
                f'Response: {parsed!r}'
            ) from direct_error
        raise ValueError(
            f'Failed to parse LLM output as {schema.__name__}. '
            f'Response: {response!r}'
        )
