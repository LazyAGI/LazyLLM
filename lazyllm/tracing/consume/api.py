from typing import Optional

from lazyllm.tracing.semantics import is_valid_trace_id

from .configs import read_consume_backend_name, read_consume_timeout_seconds
from .datamodel.structured import StructuredTrace
from .reconstruction import rebuild


def get_single_trace(
    trace_id: str,
    *,
    backend: Optional[str] = None,
) -> StructuredTrace:
    if not is_valid_trace_id(trace_id):
        raise ValueError(f'invalid trace_id: {trace_id!r}')
    from lazyllm.tracing.backends import get_consume_backend

    consume_backend = get_consume_backend(backend or read_consume_backend_name())
    raw_payload = consume_backend.fetch_trace_payload(
        trace_id,
        timeout_seconds=read_consume_timeout_seconds(),
    )
    return rebuild(raw_payload.trace, raw_payload.spans)
