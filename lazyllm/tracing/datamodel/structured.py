from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RawData:
    input: Optional[Any]
    output: Optional[Any]


@dataclass
class TraceMetadata:
    name: Optional[str]
    start_time: float
    end_time: Optional[float]
    latency_ms: Optional[float]
    status: str
    error_message: Optional[str]
    tags: List[str]
    session_id: Optional[str]
    user_id: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ExecutionStep:
    step_id: str
    name: str
    node_type: str
    semantic_type: Optional[str]
    status: str
    start_time: float
    end_time: Optional[float]
    latency_ms: Optional[float]
    raw_data: RawData
    semantic_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    children: List['ExecutionStep'] = field(default_factory=list)


@dataclass
class StructuredTrace:
    trace_id: str
    metadata: TraceMetadata
    execution_tree: ExecutionStep


__all__ = ['ExecutionStep', 'RawData', 'StructuredTrace', 'TraceMetadata']
