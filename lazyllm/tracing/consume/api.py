from typing import Optional

from lazyllm.tracing.semantics import is_valid_trace_id

from .datamodel.views import TraceDetailView


def get_single_trace(
    trace_id: str,
    *,
    backend: Optional[str] = None,
) -> TraceDetailView:
    if not is_valid_trace_id(trace_id):
        raise ValueError(f'invalid trace_id: {trace_id!r}')
    raise NotImplementedError
