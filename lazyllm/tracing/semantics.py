import re
from typing import Final

_TRACE_ID_RE = re.compile(r'^[0-9a-f]{32}$')
_SPAN_ID_RE = re.compile(r'^[0-9a-f]{16}$')
_VALID_SPAN_KINDS = frozenset({'flow', 'module', 'callable'})
_VALID_SPAN_STATUS = frozenset({'ok', 'error'})


def is_valid_trace_id(value: str) -> bool:
    return isinstance(value, str) and bool(_TRACE_ID_RE.match(value))


def is_valid_span_id(value: str) -> bool:
    return isinstance(value, str) and bool(_SPAN_ID_RE.match(value))


class SemanticType:
    LLM: Final[str] = 'llm'
    AGENT: Final[str] = 'agent'
    RETRIEVER: Final[str] = 'retriever'
    EMBEDDING: Final[str] = 'embedding'
    TOOL: Final[str] = 'tool'
    RERANK: Final[str] = 'rerank'
    WORKFLOW_CONTROL: Final[str] = 'workflow_control'


__all__ = [
    'SemanticType',
    '_TRACE_ID_RE',
    '_SPAN_ID_RE',
    '_VALID_SPAN_KINDS',
    '_VALID_SPAN_STATUS',
    'is_valid_trace_id',
    'is_valid_span_id',
]
