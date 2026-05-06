import json

import pytest

from lazyllm import barrier, diverter, graph, ifs, loop, parallel, pipeline, switch, warp
from lazyllm.tracing.semantics import SemanticType


def add_one(value): return value + 1
def double(value): return value * 2

def chat_branch(route): return f'chat:{route}'
def search_branch(route): return f'search:{route}'
def boom(value): raise ValueError(f'boom:{value}')


def test_pipeline_tracing(exporter):
    with pipeline() as flow:
        flow.add_one = add_one
        flow.double = double
        flow.format_result = lambda v: f'result:{v}'
    flow(3)

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    a1_span, dbl_span, fmt_span, pipe_span = spans
    assert [s.name for s in (a1_span, dbl_span, fmt_span)] == ['add_one', 'double', '<lambda>']
    assert pipe_span.name == 'Pipeline'
    assert pipe_span.attributes.get('lazyllm.span.kind') == 'flow'
    assert pipe_span.attributes.get('lazyllm.semantic_type') == SemanticType.WORKFLOW_CONTROL
    assert pipe_span.attributes.get('lazyllm.status') == 'ok'
    assert json.loads(pipe_span.attributes.get('lazyllm.io.input')) == {'args': [3], 'kwargs': {}}
    assert pipe_span.attributes.get('lazyllm.io.output') == 'result:8'
    child_spans = [a1_span, dbl_span, fmt_span]
    assert all(s.parent.span_id == pipe_span.context.span_id for s in child_spans)
    assert all(s.context.trace_id == pipe_span.context.trace_id for s in child_spans)


def test_parallel_tracing(exporter):
    with parallel() as flow:
        flow.add_one = add_one
        flow.double = double
        flow.minus_one = lambda v: v - 1
    result = flow(3)

    spans = exporter.get_finished_spans()
    parallel_span, child_spans = spans[-1], spans[:3]
    assert len(spans) == 4 and len(child_spans) == 3
    assert {s.name for s in child_spans} == {'add_one', 'double', '<lambda>'}
    assert parallel_span.name == 'Parallel'
    assert list(result) == [4, 6, 2]


def test_switch_tracing(exporter):
    flow = switch('chat', chat_branch, 'search', search_branch, 'boom', boom)

    flow('search')
    with pytest.raises(Exception, match='boom:boom'):
        flow('boom')
    assert flow('missing') is None

    spans = exporter.get_finished_spans()
    matched_branch, matched_switch, error_branch, error_switch, unmatched_switch = spans

    assert matched_branch.name == 'search_branch' and matched_switch.name == 'Switch'
    assert matched_branch.parent.span_id == matched_switch.context.span_id
    assert matched_switch.attributes.get('lazyllm.matched.index') == 1
    assert matched_switch.attributes.get('lazyllm.matched.condition') == 'search'
    assert matched_switch.attributes.get('lazyllm.matched.branch') == 'search_branch'
    assert matched_switch.attributes.get('lazyllm.status') == 'ok'

    assert error_branch.name == 'boom' and error_switch.name == 'Switch'
    assert error_branch.parent.span_id == error_switch.context.span_id
    assert error_switch.attributes.get('lazyllm.matched.branch') is None

    assert unmatched_switch.name == 'Switch'
    assert unmatched_switch.attributes.get('lazyllm.matched.branch') is None
    assert unmatched_switch.attributes.get('lazyllm.status') == 'ok'


def test_ifs_tracing(exporter):
    def is_even(value): return value % 2 == 0

    flow = ifs(is_even, chat_branch, boom)

    flow(4)
    with pytest.raises(Exception, match='boom:5'):
        flow(5)
    assert flow(6) == 'chat:6'
    spans = exporter.get_finished_spans()

    true_fc, true_cond, true_branch, true_ifs = spans[0:4]
    assert true_cond.name == 'is_even' and true_branch.name == 'chat_branch'
    assert true_branch.parent.span_id == true_ifs.context.span_id
    assert true_ifs.attributes.get('lazyllm.matched.branch') == 'true_path'
    assert true_ifs.attributes.get('lazyllm.matched.chosen_node') == 'chat_branch'
    assert true_ifs.attributes.get('lazyllm.matched.condition_result') is True

    error_fc, error_cond, error_branch, error_ifs = spans[4:8]
    assert error_cond.name == 'is_even' and error_branch.name == 'boom'
    assert error_ifs.attributes.get('lazyllm.matched.branch') is None
    assert error_ifs.attributes.get('lazyllm.status') == 'error'
    assert error_ifs.attributes.get('lazyllm.error.message') == 'boom:5'

    recovery_fc, recovery_cond, recovery_branch, recovery_ifs = spans[8:12]
    assert recovery_cond.name == 'is_even' and recovery_branch.name == 'chat_branch'
    assert recovery_ifs.attributes.get('lazyllm.error.message') is None


def test_loop_tracing(exporter):
    def increment(value): return value + 1
    def stop_at_three(value): return value >= 3

    with loop(count=5, stop_condition=stop_at_three) as flow:
        flow.increment = increment
    result = flow(0)

    spans = exporter.get_finished_spans()
    loop_span, iteration_spans = spans[-1], spans[:3]
    assert len(spans) == 4 and len(iteration_spans) == 3
    assert loop_span.name == 'Loop'
    assert all(s.name == 'increment' for s in iteration_spans)
    assert loop_span.attributes.get('lazyllm.loop.actual_iterations') == 3


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
    assert diverter_span.name == 'Diverter'
    assert {s.name for s in child_spans} == {'add_one', 'double', '<lambda>'}
    assert json.loads(diverter_span.attributes.get('lazyllm.io.input')) == {'args': [3, 5, 7], 'kwargs': {}}


def test_warp_tracing(exporter):
    flow = warp(add_one)
    result = flow(1, 2, 3)

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    warp_span = spans[-1]
    child_spans = spans[:-1]
    assert warp_span.name == 'Warp'
    assert {s.name for s in child_spans} == {'add_one'}


def test_graph_tracing(exporter):
    def first(x): return x + 1
    def second(x): return x * 2
    def combine(a, b): return a + b

    with graph() as g:
        g.first = first
        g.second = second
        g.combine = combine

    g.add_edge(g.start_node_name, ['first', 'second'])
    g.add_edge(['first', 'second'], 'combine')
    g.add_edge('combine', g.end_node_name)

    result = g(3)

    spans = exporter.get_finished_spans()
    user_node_spans, graph_span = spans[:3], spans[-1]
    first_level_spans, combine_span = user_node_spans[:2], user_node_spans[2]
    assert {s.name for s in first_level_spans} == {'first', 'second'}
    assert combine_span.name == 'combine'
    assert graph_span.name == 'Graph'
    assert all(s.context.trace_id == graph_span.context.trace_id for s in user_node_spans)
    assert all(s.parent.span_id == graph_span.context.span_id for s in user_node_spans)


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
            flow.left.l1 = record('left_pre')
            flow.left.bar = barrier
            flow.left.l2 = record('left_post')
        with pipeline() as flow.right:
            flow.right.r1 = record('right_pre')
            flow.right.bar = barrier
            flow.right.r2 = record('right_post')
    result = flow(0)

    spans = exporter.get_finished_spans()
    pre_spans, rest_spans = spans[:2], spans[2:-1]
    rest_names = [s.name for s in rest_spans]
    assert spans[-1].name == 'Parallel'
    assert {s.name for s in pre_spans} == {'left_pre', 'right_pre'}
    assert {'left_post', 'right_post'}.issubset(rest_names)
    pre_indices = [order.index('left_pre'), order.index('right_pre')]
    post_indices = [order.index('left_post'), order.index('right_post')]
    assert max(pre_indices) < min(post_indices)
