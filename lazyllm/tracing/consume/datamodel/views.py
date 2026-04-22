from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RawDataView:
    input: Optional[Any]
    output: Optional[Any]
    metadata: Dict[str, Any]
    truncated: bool


@dataclass
class ExecutionStepView:
    step_id: str
    name: str
    node_type: str
    semantic_type: Optional[str]
    status: str
    latency_ms: Optional[float]
    start_time: float
    error_message: Optional[str]
    semantic_data: Optional[Dict[str, Any]]
    raw_data: RawDataView
    children: List[ExecutionStepView] = field(default_factory=list)


@dataclass
class TraceDetailMetadata:
    trace_id: str
    name: Optional[str]
    tags: List[str]
    session_id: Optional[str]
    user_id: Optional[str]
    status: str
    latency_ms: Optional[float]
    start_time: float
    error_message: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class TraceDetailView:
    metadata: TraceDetailMetadata
    execution_tree: ExecutionStepView


__all__ = [
    'ExecutionStepView',
    'RawDataView',
    'TraceDetailMetadata',
    'TraceDetailView',
]
