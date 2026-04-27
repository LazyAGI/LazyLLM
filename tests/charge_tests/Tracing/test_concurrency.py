import time

from lazyllm import barrier, loop, parallel


def _assert_overlapping(spans):
    assert any(
        left.start_time < right.end_time and right.start_time < left.end_time
        for index, left in enumerate(spans)
        for right in spans[index + 1:]
    ), f"Expected overlapping spans, but spans are sequential: {[s.name for s in spans]}"


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
    result = flow(3)

    spans = exporter.get_finished_spans()
    child_spans, parallel_span = spans[:3], spans[3]
    assert list(result) == [4, 6, 2]
    assert parallel_span.name == "Parallel"
    assert {s.name for s in child_spans} == {"branch_a", "branch_b", "branch_c"}
    assert all(s.context.trace_id == parallel_span.context.trace_id for s in spans)
    assert all(s.parent.span_id == parallel_span.context.span_id for s in child_spans)
    _assert_overlapping(child_spans)


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
    result = flow(0)

    spans = exporter.get_finished_spans()
    loop_span = spans[-1]
    parallel_spans = spans[2:9:3]
    branch_spans = spans[0:2] + spans[3:5] + spans[6:8]
    assert result == 3
    assert loop_span.name == "Loop"
    assert all(s.name == "Parallel" for s in parallel_spans)

    assert all(s.context.trace_id == loop_span.context.trace_id for s in spans)
    assert all(s.parent.span_id == loop_span.context.span_id for s in parallel_spans)
    for parallel_span in parallel_spans:
        children = [s for s in branch_spans if s.parent.span_id == parallel_span.context.span_id]
        assert {s.name for s in children} == {"increment", "keep_zero"}
        _assert_overlapping(children)
