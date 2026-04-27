import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace

import lazyllm
from lazyllm import set_trace_context, LazyTraceContext
from lazyllm.tracing.collect import runtime as tracing_runtime


class MemoryTracingBackend:
    name = 'memory'

    def __init__(self, exporter):
        self._exporter = exporter

    def build_exporter(self):
        return self._exporter

    def map_attributes(self, otel_attrs):
        return {}


@pytest.fixture
def exporter(monkeypatch):
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = trace.Once()

    set_trace_context(LazyTraceContext(trace_id='test-trace', enabled=True))
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(tracing_runtime, '_runtime', tracing_runtime.TracingRuntime())
    monkeypatch.setattr(tracing_runtime, 'get_tracing_backend', lambda name: MemoryTracingBackend(exporter))
    monkeypatch.setattr(tracing_runtime.opentelemetry.sdk.trace.export, 'BatchSpanProcessor', SimpleSpanProcessor)

    with lazyllm.config.temp('trace_backend', 'memory'):
        with lazyllm.config.temp('trace_enabled', True):
            with lazyllm.config.temp('trace_content_enabled', True):
                assert tracing_runtime.tracing_available()
                yield exporter

    tracing_runtime._runtime.shutdown()
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = trace.Once()
