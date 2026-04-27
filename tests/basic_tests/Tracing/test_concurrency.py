from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from lazyllm import barrier, loop, parallel


CONCURRENT_REQUESTS = 32
REQUESTS = {f'req-{index:02d}': index for index in range(CONCURRENT_REQUESTS)}


def _run_concurrent_requests(flow):
    results = {}
    with ThreadPoolExecutor(max_workers=len(REQUESTS)) as executor:
        futures = {executor.submit(flow, request): tag for tag, request in REQUESTS.items()}
        for future in as_completed(futures):
            tag = futures[future]
            results[tag] = future.result()
    assert set(results) == set(REQUESTS)
    return results


def _group_spans_by_trace_id(spans):
    grouped_spans = defaultdict(list)
    for span in spans:
        grouped_spans[span.context.trace_id].append(span)
    return grouped_spans


def _assert_overlapping(spans):
    assert any(
        left.start_time < right.end_time and right.start_time < left.end_time
        for index, left in enumerate(spans)
        for right in spans[index + 1:]
    ), f'Expected overlapping spans, but spans are sequential: {[s.name for s in spans]}'


def _assert_parallel_trace_group(spans):
    assert len(spans) == 4
    child_spans, parallel_span = spans[:3], spans[-1]
    assert parallel_span.name == 'Parallel'
    assert {span.name for span in child_spans} == {'branch_a', 'branch_b', 'branch_c'}
    assert all(span.context.trace_id == parallel_span.context.trace_id for span in spans)
    assert all(span.parent.span_id == parallel_span.context.span_id for span in child_spans)
    _assert_overlapping(child_spans)


def _assert_loop_parallel_trace_group(spans):
    assert len(spans) == 10
    loop_span = spans[-1]
    assert loop_span.name == 'Loop'
    assert all(span.context.trace_id == loop_span.context.trace_id for span in spans)

    for iteration in range(3):
        start = iteration * 3
        children = spans[start:start + 2]
        parallel_span = spans[start + 2]
        assert parallel_span.name == 'Parallel'
        assert parallel_span.parent.span_id == loop_span.context.span_id
        assert {span.name for span in children} == {'increment', 'keep_zero'}
        assert all(span.parent.span_id == parallel_span.context.span_id for span in children)
        _assert_overlapping(children)


def test_parallel_tracing_propagates_context_across_threads(exporter):
    def branch_a(value):
        barrier(value)
        time.sleep(0.03)
        return value + 1

    def branch_b(value):
        barrier(value)
        time.sleep(0.03)
        return value * 2

    def branch_c(value):
        barrier(value)
        return value - 1

    with parallel(_concurrent=3) as flow:
        flow.branch_a = branch_a
        flow.branch_b = branch_b
        flow.branch_c = branch_c
    flow.aslist

    results = _run_concurrent_requests(flow)

    spans = exporter.get_finished_spans()
    grouped_spans = _group_spans_by_trace_id(spans)
    assert len(spans) == CONCURRENT_REQUESTS * 4
    assert len(grouped_spans) == CONCURRENT_REQUESTS
    for tag, index in REQUESTS.items():
        assert results[tag] == [index + 1, index * 2, index - 1]
    for trace_spans in grouped_spans.values():
        _assert_parallel_trace_group(trace_spans)


def test_loop_with_parallel_keeps_nested_parent_context(exporter):
    def increment(value):
        barrier(value)
        return value + 1

    def keep_zero(value):
        barrier(value)
        return 0

    with parallel(_concurrent=2) as branches:
        branches.increment = increment
        branches.keep_zero = keep_zero
    branches.sum

    with loop(count=3) as flow:
        flow.branches = branches

    results = _run_concurrent_requests(flow)

    spans = exporter.get_finished_spans()
    grouped_spans = _group_spans_by_trace_id(spans)
    assert len(spans) == CONCURRENT_REQUESTS * 10
    assert len(grouped_spans) == CONCURRENT_REQUESTS
    for tag, index in REQUESTS.items():
        assert results[tag] == index + 3
    for trace_spans in grouped_spans.values():
        _assert_loop_parallel_trace_group(trace_spans)
