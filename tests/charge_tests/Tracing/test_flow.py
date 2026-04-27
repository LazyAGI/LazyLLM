import json

from lazyllm import ifs, loop, parallel, pipeline, switch


def add_one(value):
    return value + 1

def double(value):
    return value * 2


def test_pipeline_tracing(exporter):
    with pipeline() as flow:
        flow.add_one = add_one
        flow.double = double
        flow.format_result = lambda v: f"result:{v}"
    result = flow(3)

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    add_one_span, double_span, format_result_span, pipeline_span = spans
    assert [s.name for s in (add_one_span, double_span, format_result_span)] == ["add_one", "double", "<lambda>"]
    assert pipeline_span.name == "Pipeline"
    assert pipeline_span.attributes.get("lazyllm.span.kind") == "flow"
    assert pipeline_span.attributes.get("lazyllm.semantic_type") == "workflow_control"
    assert json.loads(pipeline_span.attributes.get("lazyllm.io.input")) == {"args": [3], "kwargs": {}}
    assert pipeline_span.attributes.get("lazyllm.io.output") == "result:8"
    child_spans = [add_one_span, double_span, format_result_span]
    assert all(s.parent.span_id == pipeline_span.context.span_id for s in child_spans)
    assert result == "result:8"


def test_parallel_tracing(exporter):
    with parallel() as flow:
        flow.add_one = add_one
        flow.double = double
        flow.minus_one = lambda v: v - 1
    result = flow(3)

    spans = exporter.get_finished_spans()
    parallel_span, child_spans = spans[-1], spans[:3]
    assert len(spans) == 4 and len(child_spans) == 3
    assert {s.name for s in child_spans} == {"add_one", "double", "<lambda>"}
    assert parallel_span.name == "Parallel"
    assert list(result) == [4, 6, 2]


def test_switch_tracing(exporter):
    def is_positive(value):
        return value > 0

    def positive_branch(value):
        return f"positive:{value}"

    def default_branch(value):
        return f"default:{value}"

    flow = switch(is_positive, positive_branch, "default", default_branch)
    result = flow(3)

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    branch_span, switch_span = spans
    assert switch_span.name == "Switch"
    assert branch_span.name == "positive_branch"
    assert switch_span.attributes.get("lazyllm.matched.index") == 0
    assert switch_span.attributes.get("lazyllm.matched.branch") == "positive_branch"
    assert result == "positive:3"


def test_loop_tracing(exporter):
    def increment(value):
        return value + 1

    def stop_at_three(value):
        return value >= 3

    with loop(count=5, stop_condition=stop_at_three) as flow:
        flow.increment = increment
    result = flow(0)

    spans = exporter.get_finished_spans()
    loop_span, iteration_spans = spans[-1], spans[:3]
    assert len(spans) == 4 and len(iteration_spans) == 3
    assert loop_span.name == "Loop"
    assert all(s.name == "increment" for s in iteration_spans)
    assert loop_span.attributes.get("lazyllm.loop.actual_iterations") == 3
    assert result == 3


def test_ifs_tracing(exporter):
    def is_even(value=4):
        return value % 2 == 0

    def true_path(value):
        return f"even:{value}"

    def false_path(value):
        return f"odd:{value}"

    flow = ifs(is_even, true_path, false_path)
    result = flow(4)

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    condition_span, branch_span, ifs_span = spans
    assert [condition_span.name, branch_span.name, ifs_span.name] == ["is_even", "true_path", "IFS"]
    assert ifs_span.attributes.get("lazyllm.matched.branch") == "true_path"
    assert ifs_span.attributes.get("lazyllm.matched.chosen_node") == "true_path"
    assert ifs_span.attributes.get("lazyllm.matched.condition_result") is True
    assert result == "even:4"
