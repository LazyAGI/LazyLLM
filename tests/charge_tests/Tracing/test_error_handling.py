import pytest

from lazyllm import parallel, pipeline


def first(value):
    return value + 1

def raises_error(value):
    raise ValueError(f"boom:{value}")

def unreachable(value):
    return value * 2


def test_pipeline_step_error_tracing(exporter):
    with pipeline() as flow:
        flow.first = first
        flow.raises_error = raises_error
        flow.unreachable = unreachable

    with pytest.raises(Exception, match="boom:2"):
        flow(1)

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["first", "raises_error", "Pipeline"]

    first_span, error_span, pipeline_span = spans
    assert first_span.attributes.get("lazyllm.status") == "ok"
    assert error_span.attributes.get("lazyllm.status") == "error"
    assert pipeline_span.attributes.get("lazyllm.status") == "error"
    assert error_span.attributes.get("lazyllm.error.message") == "boom:2"


def test_parallel_branch_error_tracing(exporter):
    def third(value):
        return value * 2

    with parallel(_concurrent=3) as flow:
        flow.first = first
        flow.raises_error = raises_error
        flow.third = third

    with pytest.raises(Exception, match="boom:1"):
        flow(1)

    spans = exporter.get_finished_spans()
    assert len(spans) == 4

    branch_spans, parallel_span = spans[:3], spans[3]
    assert parallel_span.name == "Parallel"
    assert {s.name for s in branch_spans} == {"first", "raises_error", "third"}
    assert parallel_span.attributes.get("lazyllm.status") == "error"

    error_span = [s for s in branch_spans if s.name == "raises_error"][0]
    ok_spans = [s for s in branch_spans if s is not error_span]
    assert {s.attributes.get("lazyllm.status") for s in ok_spans} == {"ok"}
    assert error_span.attributes.get("lazyllm.status") == "error"
    assert {s.context.trace_id for s in spans} == {parallel_span.context.trace_id}


def test_nested_parallel_error_propagates_to_pipeline(exporter):
    with parallel(_concurrent=2) as branches:
        branches.first = first
        branches.raises_error = raises_error

    with pipeline() as flow:
        flow.branches = branches
        flow.unreachable = unreachable

    with pytest.raises(Exception, match="boom:1"):
        flow(1)

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    assert {s.name for s in spans} == {"first", "raises_error", "Parallel", "Pipeline"}
    assert "unreachable" not in {s.name for s in spans}

    branch_spans, parallel_span, pipeline_span = spans[:2], spans[2], spans[3]
    error_span = [s for s in branch_spans if s.name == "raises_error"][0]
    assert error_span.attributes.get("lazyllm.status") == "error"
    assert parallel_span.attributes.get("lazyllm.status") == "error"
    assert pipeline_span.attributes.get("lazyllm.status") == "error"
