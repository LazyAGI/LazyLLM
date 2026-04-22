from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionStep:
    step_id: str
    name: str
    node_type: str
    semantic_type: Optional[str]
    status: str
    latency_ms: Optional[float]
    start_time: float
    error_message: Optional[str]
    semantic_data: Optional[Dict[str, Any]]
    raw_data: Dict[str, Any]
    children: List[ExecutionStep] = field(default_factory=list)


@dataclass
class StructuredTrace:
    trace_id: str
    name: Optional[str]
    tags: List[str]
    session_id: Optional[str]
    user_id: Optional[str]
    metadata: Dict[str, Any]
    start_time: float
    latency_ms: Optional[float]
    status: str
    error_message: Optional[str]
    root_step_id: Optional[str]
    execution_tree: ExecutionStep
    has_orphans: bool


__all__ = ['ExecutionStep', 'StructuredTrace']
