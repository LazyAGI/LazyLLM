import json

from lazyllm import barrier, diverter, graph, ifs, loop, parallel, pipeline, switch, warp


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
    def chat_branch(route):
        return f"chat:{route}"

    def search_branch(route):
        return f"search:{route}"

    def default_branch(route):
        return f"default:{route}"

    flow = switch("chat", chat_branch, "search", search_branch, "default", default_branch)
    result = flow("search")

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    branch_span, switch_span = spans
    assert switch_span.name == "Switch"
    assert branch_span.name == "search_branch"
    assert branch_span.parent.span_id == switch_span.context.span_id
    assert switch_span.attributes.get("lazyllm.matched.index") == 1
    assert switch_span.attributes.get("lazyllm.matched.branch") == "search_branch"
    assert result == "search:search"


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


def test_diverter_tracing(exporter):
    with diverter() as flow:
        flow.add_one = add_one
        flow.double = double
        flow.minus_one = lambda v: v - 1
    result = flow(3, 5, 7)

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    diverter_span = spans[-1]
    child_spans = spans[:-1]
    assert diverter_span.name == "Diverter"
    assert diverter_span.attributes.get("lazyllm.span.kind") == "flow"
    assert {s.name for s in child_spans} == {"add_one", "double", "<lambda>"}
    assert all(s.parent.span_id == diverter_span.context.span_id for s in child_spans)
    assert all(s.context.trace_id == diverter_span.context.trace_id for s in child_spans)
    assert json.loads(diverter_span.attributes.get("lazyllm.io.input")) == {"args": [3, 5, 7], "kwargs": {}}
    assert tuple(result) == (4, 10, 6)


def test_warp_tracing(exporter):
    flow = warp(add_one)
    result = flow(1, 2, 3)

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    warp_span = spans[-1]
    child_spans = spans[:-1]
    assert warp_span.name == "Warp"
    assert warp_span.attributes.get("lazyllm.span.kind") == "flow"
    assert {s.name for s in child_spans} == {"add_one"}
    assert all(s.parent.span_id == warp_span.context.span_id for s in child_spans)
    assert all(s.context.trace_id == warp_span.context.trace_id for s in child_spans)
    assert json.loads(warp_span.attributes.get("lazyllm.io.input")) == {"args": [1, 2, 3], "kwargs": {}}
    assert tuple(result) == (2, 3, 4)


def test_graph_tracing(exporter):
    def first(x):
        return x + 1

    def second(x):
        return x * 2

    def combine(a, b):
        return a + b

    with graph() as g:
        g.first = first
        g.second = second
        g.combine = combine

    g.add_edge(g.start_node_name, ["first", "second"])
    g.add_edge(["first", "second"], "combine")
    g.add_edge("combine", g.end_node_name)

    result = g(3)

    spans = exporter.get_finished_spans()
    user_node_spans, graph_span = spans[:3], spans[-1]
    assert [s.name for s in user_node_spans] == ["first", "second", "combine"]
    assert graph_span.name == "Graph"
    assert graph_span.attributes.get("lazyllm.span.kind") == "flow"
    assert all(s.context.trace_id == graph_span.context.trace_id for s in user_node_spans)
    assert all(s.parent.span_id == graph_span.context.span_id for s in user_node_spans)
    assert json.loads(graph_span.attributes.get("lazyllm.io.input")) == {"args": [3], "kwargs": {}}
    assert result == (3 + 1) + (3 * 2)


def test_barrier_tracing(exporter):
    order = []

    def record(tag):
        def _step(value):
            order.append(tag)
            return value + 1
        _step.__name__ = tag
        return _step

    with parallel() as flow:
        with pipeline() as flow.left:
            flow.left.l1 = record("left_pre")
            flow.left.bar = barrier
            flow.left.l2 = record("left_post")
        with pipeline() as flow.right:
            flow.right.r1 = record("right_pre")
            flow.right.bar = barrier
            flow.right.r2 = record("right_post")
    result = flow(0)

    spans = exporter.get_finished_spans()
    pre_spans, rest_spans = spans[:2], spans[2:-1]
    rest_names = [s.name for s in rest_spans]
    assert spans[-1].name == "Parallel"
    assert {s.name for s in pre_spans} == {"left_pre", "right_pre"}
    assert {"left_post", "right_post"}.issubset(rest_names)
    pre_indices = [order.index("left_pre"), order.index("right_pre")]
    post_indices = [order.index("left_post"), order.index("right_post")]
    assert max(pre_indices) < min(post_indices)
    assert tuple(result) == (2, 2)
