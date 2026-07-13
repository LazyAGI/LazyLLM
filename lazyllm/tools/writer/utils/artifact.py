from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)

SCHEMA_VERSION = '0.1'


class Artifact(BaseModel):
    schema_name: str = Field(serialization_alias='schema')
    schema_version: str = SCHEMA_VERSION
    data: Any
    meta: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    artifact_path: Optional[str] = None
    context_path: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _schema_name(obj: BaseModel) -> str:
    cls = type(obj)
    return _schema_name_for_class(cls)


def _schema_name_for_class(cls: Type[BaseModel]) -> str:
    module = cls.__module__ or ''
    return f'{module}.{cls.__qualname__}'


def _infer_schema_name(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return _schema_name(obj)
    cls = type(obj)
    module = cls.__module__ or ''
    return f'{module}.{cls.__qualname__}'


def _to_json_data(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


def save_artifact_json(
    obj: Any,
    path: str,
    *,
    schema_name: Optional[str] = None,
    created_by: str = '',
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    meta: Dict[str, Any] = {
        'created_by': created_by,
        'created_at': datetime.now().astimezone().isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)

    artifact = Artifact(
        schema_name=schema_name or _infer_schema_name(obj),
        data=_to_json_data(obj),
        meta=meta,
    )

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(artifact.model_dump_json(indent=2, by_alias=True))

    return os.path.abspath(path)


def load_artifact_json(
    path: str,
    model_class: Optional[Type[T]] = None,
    *,
    expected_schema_name: Optional[str] = None,
    validate_schema: bool = True,
) -> Any:
    with open(path, 'r', encoding='utf-8') as fh:
        raw = json.load(fh)

    if 'data' not in raw:
        raise ValueError(f'Artifact file {path!r} is missing the \'data\' field.')

    if validate_schema:
        expected = expected_schema_name
        if expected is None and model_class is not None:
            expected = _schema_name_for_class(model_class)
        actual = raw.get('schema')
        if expected is not None and actual != expected:
            raise ValueError(
                f'Artifact schema mismatch for {path!r}: expected {expected!r}, got {actual!r}.'
            )

    if model_class is None:
        return raw['data']

    return model_class.model_validate(raw['data'])


class ArtifactModel(BaseModel):
    def save(self, path: str, *, created_by: str = '', extra_meta: Optional[Dict[str, Any]] = None) -> str:
        return save_artifact_json(self, path, created_by=created_by, extra_meta=extra_meta)

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        return load_artifact_json(path, cls)
