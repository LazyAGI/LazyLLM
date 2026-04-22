from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lazyllm.tracing.semantics import is_valid_span_id, is_valid_trace_id


@dataclass
class RawTraceRecord:
    trace_id: str
    name: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]
    input: Optional[Any]
    output: Optional[Any]
    start_time: Optional[float]
    end_time: Optional[float]
    status: Optional[str]
    raw: Dict[str, Any]

    def __post_init__(self) -> None:
        if not is_valid_trace_id(self.trace_id):
            raise ValueError(f'invalid trace_id: {self.trace_id!r}')


@dataclass
class RawSpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float]
    status: str
    attributes: Dict[str, Any]
    input: Optional[Any]
    output: Optional[Any]
    metadata: Dict[str, Any]
    error_message: Optional[str]
    raw: Dict[str, Any]

    def __post_init__(self) -> None:
        if not is_valid_trace_id(self.trace_id):
            raise ValueError(f'invalid trace_id: {self.trace_id!r}')
        if not is_valid_span_id(self.span_id):
            raise ValueError(f'invalid span_id: {self.span_id!r}')
        if self.parent_span_id is not None and not is_valid_span_id(self.parent_span_id):
            raise ValueError(f'invalid parent_span_id: {self.parent_span_id!r}')


@dataclass
class RawTracePayload:
    trace: RawTraceRecord
    spans: List[RawSpanRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        trace_id = self.trace.trace_id
        for span in self.spans:
            if span.trace_id != trace_id:
                raise ValueError(
                    f'RawTracePayload span trace_id mismatch: {span.trace_id!r} != {trace_id!r}'
                )


__all__ = ['RawSpanRecord', 'RawTracePayload', 'RawTraceRecord']
