from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Optional

from lazyllm.common import LOG


def _normalize_tags(tags: Any) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, (str, bytes, bytearray)):
        return [str(tags)]
    if isinstance(tags, Iterable):
        return [str(tag) for tag in tags if tag is not None]
    return [str(tags)]


@dataclass
class LazyTraceContext:
    enabled: Optional[bool] = None
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_tags: List[str] = field(default_factory=list)
    module_trace: Optional[Dict[str, Any]] = None
    sampled: Optional[bool] = None
    debug_capture_payload: Optional[bool] = None

    def to_dict(self) -> dict:
        result: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, list):
                result[f.name] = list(value)
            elif isinstance(value, dict):
                result[f.name] = dict(value)
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'LazyTraceContext':
        if not isinstance(data, dict):
            LOG.warning(f'LazyTraceContext.from_dict expected a dict, got {type(data).__name__}; '
                        f'returning empty context.')
            return cls()
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        filtered['request_tags'] = _normalize_tags(filtered.get('request_tags'))
        return cls(**filtered)
