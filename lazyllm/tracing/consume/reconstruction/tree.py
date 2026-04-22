from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from ..datamodel.raw import RawSpanRecord, RawTraceRecord
from ..datamodel.structured import ExecutionStep, StructuredTrace
from .extractors import extract_semantic


_VALID_NODE_TYPES = frozenset({'flow', 'module', 'callable'})
_TRUNCATED_SUFFIX = '...<truncated>'
_VIRTUAL_ROOT_STEP_ID = '__root__'


def _latency_ms(start_time: Optional[float], end_time: Optional[float]) -> Optional[float]:
    if start_time is None or end_time is None:
        return None
    return max(0.0, (end_time - start_time) * 1000.0)


def _contains_truncated(value: Any) -> bool:
    if isinstance(value, str):
        return value.endswith(_TRUNCATED_SUFFIX)
    if isinstance(value, dict):
        return any(_contains_truncated(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_truncated(v) for v in value)
    return False


def _span_truncated(span: RawSpanRecord) -> bool:
    return (
        _contains_truncated(span.input)
        or _contains_truncated(span.output)
        or _contains_truncated(span.attributes.get('lazyllm.io.input'))
        or _contains_truncated(span.attributes.get('lazyllm.io.output'))
    )


def _trace_truncated(trace: RawTraceRecord) -> bool:
    return (
        _contains_truncated(trace.input)
        or _contains_truncated(trace.output)
        or _contains_truncated(trace.metadata)
    )


def _node_type(span: RawSpanRecord) -> str:
    value = span.attributes.get('lazyllm.span.kind')
    return value if value in _VALID_NODE_TYPES else 'callable'


def _error_message(span: RawSpanRecord) -> Optional[str]:
    return span.error_message or span.attributes.get('lazyllm.error.message')


def _build_step(span: RawSpanRecord) -> ExecutionStep:
    return ExecutionStep(
        step_id=span.span_id,
        name=span.name,
        node_type=_node_type(span),
        semantic_type=span.attributes.get('lazyllm.semantic_type'),
        status=span.status,
        latency_ms=_latency_ms(span.start_time, span.end_time),
        start_time=span.start_time,
        error_message=_error_message(span),
        semantic_data=extract_semantic(span),
        raw_data={
            'input': span.input,
            'output': span.output,
            'metadata': span.metadata,
            'truncated': _span_truncated(span),
        },
    )


def _parent_chain_has_cycle(span: RawSpanRecord, spans_by_id: Dict[str, RawSpanRecord]) -> bool:
    seen = {span.span_id}
    parent_id = span.parent_span_id
    while parent_id is not None and parent_id in spans_by_id:
        if parent_id in seen:
            return True
        seen.add(parent_id)
        parent_id = spans_by_id[parent_id].parent_span_id
    return False


def _sort_steps(steps: List[ExecutionStep]) -> None:
    steps.sort(key=lambda step: (step.start_time, step.name, step.step_id))
    for step in steps:
        _sort_steps(step.children)


def _iter_steps_dfs(roots: Iterable[ExecutionStep]) -> Iterable[ExecutionStep]:
    for step in roots:
        yield step
        yield from _iter_steps_dfs(step.children)


def _first_error_message(roots: Iterable[ExecutionStep]) -> Optional[str]:
    for step in _iter_steps_dfs(roots):
        if step.status == 'error' and step.error_message:
            return step.error_message
    return None


def _aggregate_status(trace: RawTraceRecord, spans: List[RawSpanRecord]) -> str:
    if any(span.status == 'error' for span in spans):
        return 'error'
    return trace.status if trace.status in ('ok', 'error') else 'ok'


def _aggregate_start_time(trace: RawTraceRecord, spans: List[RawSpanRecord]) -> float:
    if trace.start_time is not None:
        return trace.start_time
    if spans:
        return min(span.start_time for span in spans)
    return 0.0


def _trace_raw_latency_ms(trace: RawTraceRecord) -> Optional[float]:
    raw_latency = trace.raw.get('latency')
    if isinstance(raw_latency, (int, float)):
        return max(0.0, float(raw_latency) * 1000.0)
    return None


def _aggregate_latency_ms(trace: RawTraceRecord, spans: List[RawSpanRecord]) -> Optional[float]:
    starts = [span.start_time for span in spans]
    ends = [span.end_time for span in spans if span.end_time is not None]
    if starts and ends:
        return _latency_ms(min(starts), max(ends))
    return _trace_raw_latency_ms(trace)


def _virtual_root(
    trace: RawTraceRecord,
    children: List[ExecutionStep],
    *,
    status: str,
    start_time: float,
    latency_ms: Optional[float],
    error_message: Optional[str],
) -> ExecutionStep:
    return ExecutionStep(
        step_id=_VIRTUAL_ROOT_STEP_ID,
        name=trace.name or trace.trace_id,
        node_type='flow',
        semantic_type=None,
        status=status,
        latency_ms=latency_ms,
        start_time=start_time,
        error_message=error_message,
        semantic_data=None,
        raw_data={
            'input': trace.input,
            'output': trace.output,
            'metadata': trace.metadata,
            'truncated': _trace_truncated(trace),
        },
        children=children,
    )


def rebuild(trace: RawTraceRecord, spans: List[RawSpanRecord]) -> StructuredTrace:
    status = _aggregate_status(trace, spans)
    start_time = _aggregate_start_time(trace, spans)
    aggregate_latency_ms = _aggregate_latency_ms(trace, spans)

    spans_by_id = {span.span_id: span for span in spans}
    steps_by_id = {span.span_id: _build_step(span) for span in spans}

    root_ids: List[str] = []
    root_id_set: Set[str] = set()
    has_orphans = not spans

    for span in spans:
        parent_id = span.parent_span_id
        cycle = _parent_chain_has_cycle(span, spans_by_id)
        parent_missing = parent_id is not None and parent_id not in spans_by_id

        if parent_id is None or parent_missing or cycle:
            if parent_missing or cycle:
                has_orphans = True
            if span.span_id not in root_id_set:
                root_ids.append(span.span_id)
                root_id_set.add(span.span_id)
            continue

        steps_by_id[parent_id].children.append(steps_by_id[span.span_id])

    root_steps = [steps_by_id[span_id] for span_id in root_ids]
    _sort_steps(root_steps)

    error_message = _first_error_message(root_steps)
    use_virtual_root = len(root_steps) != 1 or has_orphans

    if use_virtual_root:
        execution_tree = _virtual_root(
            trace,
            root_steps,
            status=status,
            start_time=start_time,
            latency_ms=aggregate_latency_ms,
            error_message=error_message,
        )
        root_step_id = None
        has_orphans = True
    else:
        execution_tree = root_steps[0]
        root_step_id = execution_tree.step_id
        if execution_tree.status == 'error' and execution_tree.error_message:
            error_message = execution_tree.error_message
        latency = execution_tree.latency_ms
        if latency is None:
            latency = aggregate_latency_ms
        aggregate_latency_ms = latency

    return StructuredTrace(
        trace_id=trace.trace_id,
        name=trace.name,
        tags=list(trace.tags),
        session_id=trace.session_id,
        user_id=trace.user_id,
        metadata=dict(trace.metadata),
        start_time=start_time,
        latency_ms=aggregate_latency_ms,
        status=status,
        error_message=error_message,
        root_step_id=root_step_id,
        execution_tree=execution_tree,
        has_orphans=has_orphans,
    )
