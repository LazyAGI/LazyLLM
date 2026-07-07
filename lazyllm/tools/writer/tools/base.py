from __future__ import annotations
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar
from pydantic import BaseModel
from lazyllm.components.formatter import JsonFormatter
from lazyllm.module import ModuleBase
from ..prompts.structured_output import STRUCTURED_OUTPUT_SYSTEM_PROMPT
from ..utils.artifact import ToolResult, load_artifact_json, save_artifact_json

T = TypeVar('T', bound=BaseModel)


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
            return value.model_dump()
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
                else f'{artifact_key}.json'
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

    def _call_llm_structured(self, prompt: str, schema: Type[T]) -> T:
        if self.llm is None:
            raise ValueError('llm is not set')

        system_prompt = self._structured_output_prompt(schema)
        model = self._build_structured_llm(system_prompt)
        response = model(prompt)
        return self._validate_structured_response(response, schema)

    def _structured_output_prompt(self, schema: Type[BaseModel]) -> str:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        return STRUCTURED_OUTPUT_SYSTEM_PROMPT.format(schema_name=schema.__name__, schema_json=schema_json)

    def _build_structured_llm(self, system_prompt: str) -> Any:
        model = self.llm
        if hasattr(model, 'share'):
            try:
                model = model.share(stream=False)
            except TypeError:
                model = model.share()
        if hasattr(model, 'prompt'):
            model = model.prompt(system_prompt)
        if hasattr(model, 'formatter'):
            model = model.formatter(JsonFormatter())
        return model

    def _validate_structured_response(self, response: Any, schema: Type[T]) -> T:
        parsed = response
        if isinstance(parsed, schema):
            return parsed
        if isinstance(parsed, str):
            try:
                parsed = JsonFormatter()(parsed)
            except Exception as exc:
                raise ValueError(
                    f'Failed to parse LLM output as JSON for {schema.__name__}. '
                    f'Response: {response!r}'
                ) from exc
        if isinstance(parsed, list) and len(parsed) == 1:
            parsed = parsed[0]
        if isinstance(parsed, dict):
            try:
                return schema.model_validate(parsed)
            except Exception as exc:
                raise ValueError(
                    f'Failed to validate LLM output as {schema.__name__}. '
                    f'Response: {parsed!r}'
                ) from exc
        raise ValueError(
            f'Failed to parse LLM output as {schema.__name__}. '
            f'Response: {response!r}'
        )
