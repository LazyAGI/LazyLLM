from unittest.mock import patch

import pytest

import lazyllm
from lazyllm import LazyTraceContext, OnlineChatModule, OnlineEmbeddingModule, pipeline, set_trace_context


@pytest.mark.parametrize("context, exp_cnt", [
    (LazyTraceContext(), 2),
    (LazyTraceContext(enabled=False, sampled=True), 0),
    (LazyTraceContext(enabled=True, sampled=False), 0),
    (LazyTraceContext(enabled=True, sampled=True), 2),
], ids=["default", "disabled", "sampled_false", "enabled"])
def test_trace_context_controls_span_recording(exporter, context, exp_cnt):
    with pipeline() as flow:
        flow.add_one = lambda value: value + 1

    set_trace_context(context)
    result = flow(1)

    assert result == 2
    assert len(exporter.get_finished_spans()) == exp_cnt


@pytest.mark.parametrize("trace_enabled, exp_cnt", [(False, 0), (True, 2)], ids=["disabled", "enabled"])
def test_trace_global_config_controls_span_recording(exporter, trace_enabled, exp_cnt):
    with lazyllm.config.temp("trace_enabled", trace_enabled):
        with pipeline() as flow:
            flow.add_one = lambda value: value + 1
        result = flow(1)

    assert result == 2
    assert len(exporter.get_finished_spans()) == exp_cnt


@pytest.mark.parametrize("enabled, exp_cnt", [(False, 2), (True, 3)], ids=["disabled", "enabled"])
def test_trace_module_trace_by_class(exporter, enabled, exp_cnt):
    set_trace_context(LazyTraceContext(enabled=True, module_trace={
        "by_class": {"OnlineChatModule": enabled},
    }))

    with patch.object(OnlineChatModule, "forward", return_value=1):
        llm = OnlineChatModule(source="dynamic", type="llm", model="mock-chat")
        with pipeline() as flow:
            flow.llm = llm
            flow.add_one = lambda value: value + 1
        result = flow("hello")

    assert result == 2
    assert len(exporter.get_finished_spans()) == exp_cnt


def test_trace_module_trace_name_overrides_class(exporter):
    set_trace_context(LazyTraceContext(enabled=True, module_trace={
        "by_class": {"OnlineEmbeddingModule": False, "OnlineChatModule": False},
        "by_name": {"test_embed": True},
    }))

    with patch.object(OnlineChatModule, "forward", return_value=1), \
            patch.object(OnlineEmbeddingModule, "forward", return_value=1):
        llm = OnlineChatModule(source="dynamic", type="llm", model="mock-chat")
        embedding = OnlineEmbeddingModule(
            source="dynamic", type="embed", model="mock-embedding", name="test_embed")
        with pipeline() as flow:
            flow.llm = llm
            flow.embedding = embedding
            flow.add_one = lambda value: value + 1
        result = flow("hello")

    assert result == 2
    assert [s.name for s in exporter.get_finished_spans()] == ["test_embed", "<lambda>", "Pipeline"]
