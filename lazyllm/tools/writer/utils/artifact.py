from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T", bound="ArtifactModel")

SCHEMA_VERSION = "0.1"


class Artifact(BaseModel):
    schema_name: str = Field(serialization_alias="schema")
    schema_version: str = SCHEMA_VERSION
    data: Dict[str, Any]
    meta: Dict[str, Any] = {}


class ToolResult(BaseModel):
    artifact_path: Optional[str] = None
    context_path: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = {}


def _schema_name(obj: BaseModel) -> str:
    cls = type(obj)
    module = cls.__module__ or ""
    return f"{module}.{cls.__qualname__}"


def save_artifact_json(
    obj: BaseModel,
    path: str,
    *,
    created_by: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    meta: Dict[str, Any] = {
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)

    artifact = Artifact(
        schema_name=_schema_name(obj),
        data=obj.model_dump(),
        meta=meta,
    )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(artifact.model_dump_json(indent=2, by_alias=True))

    return os.path.abspath(path)


def load_artifact_json(path: str, model_class: Type[T]) -> T:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if "data" not in raw:
        raise ValueError(f"Artifact file {path!r} is missing the 'data' field.")

    return model_class.model_validate(raw["data"])


class ArtifactModel(BaseModel):
    def save(self, path: str, *, created_by: str = "", extra_meta: Optional[Dict[str, Any]] = None) -> str:
        return save_artifact_json(self, path, created_by=created_by, extra_meta=extra_meta)

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        return load_artifact_json(path, cls)
