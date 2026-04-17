import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_TRACE_ID_RE = re.compile(r'^[0-9a-f]{32}$')
_SPAN_ID_RE = re.compile(r'^[0-9a-f]{16}$')


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

    # --- trace-level context (populated only on root spans for backend compatibility) ---
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_tags: List[str] = field(default_factory=list)

    # --- OTel handles (internal, not part of the semantic model) ---
    _otel_span: Any = field(default=None, repr=False)
    _otel_span_cm: Any = field(default=None, repr=False)

    # --- LazyTrace linkage (internal, set by runtime when this span owns the trace) ---
    _owns_lazy_trace: bool = field(default=False, repr=False)


@dataclass
class LazyTrace:
    """Per-request trace-level semantic object.

    Lives in a ContextVar (not in ``globals['trace']``) because it is heavy and not
    meant to be serialized across processes. Cross-thread/process propagation is done
    via the lightweight ``LazyTraceContext`` instead; each worker rebuilds its own
    ``LazyTrace`` anchored on the same ``trace_id``.

    Responsibilities:
    - single source of truth for trace-level fields (session/user/tags/metadata)
    - validate identifiers and field types at construction/update time
    - aggregate per-span outcomes into a trace-level status/latency
    """

    trace_id: str
    root_span_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_tags: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = 'ok'
    metadata: Dict[str, Any] = field(default_factory=dict)

    # True when this LazyTrace only represents a local slice of a remote trace
    # (e.g. a Parallel worker reconstructing the parent chain). Finalization in
    # a reconstructed trace is best-effort and does not imply the request is over.
    is_reconstructed: bool = False

    _span_count: int = field(default=0, repr=False)
    _error_count: int = field(default=0, repr=False)
    _lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.trace_id, str) or not _TRACE_ID_RE.match(self.trace_id):
            raise ValueError(f'LazyTrace.trace_id must be 32-char lowercase hex, got {self.trace_id!r}')
        if self.root_span_id is not None and not _SPAN_ID_RE.match(self.root_span_id):
            raise ValueError(f'LazyTrace.root_span_id must be 16-char lowercase hex, got {self.root_span_id!r}')
        if self.status not in ('ok', 'error'):
            raise ValueError(f'LazyTrace.status must be ok|error, got {self.status!r}')
        if not isinstance(self.request_tags, list):
            raise TypeError(f'LazyTrace.request_tags must be list, got {type(self.request_tags).__name__}')
        if not isinstance(self.metadata, dict):
            raise TypeError(f'LazyTrace.metadata must be dict, got {type(self.metadata).__name__}')

    # ---- state ----
    @property
    def is_active(self) -> bool:
        return self.end_time is None

    @property
    def latency_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    @property
    def span_count(self) -> int:
        with self._lock:
            return self._span_count

    # ---- mutation (trace-level user API) ----
    def add_tag(self, tag: str) -> None:
        if not isinstance(tag, str):
            raise TypeError(f'tag must be str, got {type(tag).__name__}')
        with self._lock:
            if tag not in self.request_tags:
                self.request_tags.append(tag)

    def set_metadata(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f'metadata key must be str, got {type(key).__name__}')
        with self._lock:
            self.metadata[key] = value

    def update_metadata(self, items: Dict[str, Any]) -> None:
        if not isinstance(items, dict):
            raise TypeError(f'metadata update expects dict, got {type(items).__name__}')
        with self._lock:
            self.metadata.update(items)

    # ---- lifecycle (runtime API) ----
    def _record_span_start(self, span: 'LazySpan') -> None:
        with self._lock:
            self._span_count += 1
            if self.root_span_id is None and not self.is_reconstructed and span.is_root_span:
                if span.span_id and _SPAN_ID_RE.match(span.span_id):
                    self.root_span_id = span.span_id

    def _record_span_end(self, span: 'LazySpan') -> None:
        with self._lock:
            self._span_count = max(0, self._span_count - 1)
            if span.error is not None or span.status == 'error':
                self._error_count += 1
                self.status = 'error'

    def finish(self) -> None:
        with self._lock:
            if self.end_time is None:
                self.end_time = time.time()
