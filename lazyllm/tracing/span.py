from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LazySpan:
    # --- identity ---
    name: str = ''
    span_kind: str = ''
    semantic_type: Optional[str] = None
    component_class: str = ''
    component_id: Optional[str] = None

    # --- chain (populated from OTel span after creation) ---
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    is_root_span: bool = False

    # --- data ---
    input: Optional[Any] = None
    output: Optional[Any] = None
    capture_payload: bool = False

    # --- state ---
    status: str = 'ok'
    error: Optional[Exception] = None

    # --- extensions ---
    config: Dict[str, Any] = field(default_factory=dict)
    output_attrs: Dict[str, Any] = field(default_factory=dict)
    usage: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)

    # --- trace-level context (populated only on root spans) ---
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_tags: List[str] = field(default_factory=list)

    # --- OTel handles (internal, not part of the semantic model) ---
    _otel_span: Any = field(default=None, repr=False)
    _otel_span_cm: Any = field(default=None, repr=False)


@dataclass
class LazyTrace:
    trace_id: str = ''
    root_span_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_tags: List[str] = field(default_factory=list)
    status: str = 'ok'
    metadata: Dict[str, Any] = field(default_factory=dict)
