import base64

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace

import lazyllm
from lazyllm import set_trace_context, LazyTraceContext
from lazyllm.tracing.collect import runtime as tracing_runtime
from lazyllm.tracing.backends.langfuse.backend import LangfuseConsumeBackend

TRACE_ID = '0' * 32
OTHER_TRACE_ID = '1' * 32
ROOT_SPAN_ID = '1' * 16
CHILD_SPAN_ID = '2' * 16

LANGFUSE_HOST = 'https://langfuse.example'
LANGFUSE_PUBLIC_KEY = 'public'
LANGFUSE_SECRET_KEY = 'secret'
LANGFUSE_AUTH_HEADER = (
    'Basic ' + base64.b64encode(f'{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}'.encode('utf-8')).decode()
)
LANGFUSE_ENV = {
    'LANGFUSE_HOST': LANGFUSE_HOST + '/',
    'LANGFUSE_PUBLIC_KEY': LANGFUSE_PUBLIC_KEY,
    'LANGFUSE_SECRET_KEY': LANGFUSE_SECRET_KEY,
}
_MISSING = object()
_RAW_BACKEND = LangfuseConsumeBackend()


class MemoryTracingBackend:
    name = 'memory'

    def __init__(self, exporter):
        self._exporter = exporter

    def build_exporter(self):
        return self._exporter

    def map_attributes(self, otel_attrs):
        return {}


class _FakeResponse:
    def __init__(
        self, status_code=200, body=None, *,
        headers=None, content=True, json_exc=None,
    ):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.headers = headers or {}
        self._body = body
        self.content = b'{}' if content else b''
        self._json_exc = json_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._body


def langfuse_trace_url(trace_id=TRACE_ID):
    return f'{LANGFUSE_HOST}/api/public/traces/{trace_id}'


def make_response(body=None, *, status_code=200, headers=None, content=True, json_exc=None):
    return _FakeResponse(
        status_code=status_code, body=body,
        headers=headers, content=content, json_exc=json_exc,
    )


def set_langfuse_env(monkeypatch):
    for name, value in LANGFUSE_ENV.items():
        monkeypatch.setenv(name, value)


def make_langfuse_trace_payload(observations=_MISSING, *, trace_id=TRACE_ID, **overrides):
    payload = {
        'id': trace_id,
        'timestamp': '2026-04-29T08:00:00Z',
        'name': 'trace-name',
        'sessionId': 'session-1',
        'userId': 'user-1',
        'tags': ['tag-a'],
        'metadata': {'source': 'langfuse'},
        'input': {'query': 'hello'},
        'output': {'answer': 'world'},
    }
    if observations is not _MISSING:
        payload['observations'] = observations
    payload.update(overrides)
    return payload


def make_langfuse_observation(
    observation_id, *, trace_id=TRACE_ID,
    parent_id=None, obs_type='CHAIN', name='root-step',
    start='2026-04-29T08:00:01Z', end='2026-04-29T08:00:02Z',
    obs_input=_MISSING, obs_output=_MISSING,
    attrs=None, metadata_extra=None,
    **overrides,
):
    metadata = {
        'attributes': {'lazyllm.span.kind': 'flow', **(attrs or {})},
        'resourceAttributes': {'service.name': 'lazyllm'},
        'scope': {'name': 'lazyllm.tracing'},
    }
    metadata.update(metadata_extra or {})
    observation = {
        'id': observation_id,
        'traceId': trace_id,
        'parentObservationId': parent_id,
        'type': obs_type,
        'name': name,
        'startTime': start,
        'endTime': end,
        'input': {'args': ['hello'], 'kwargs': {}} if obs_input is _MISSING else obs_input,
        'output': {'root': 'ok'} if obs_output is _MISSING else obs_output,
        'metadata': metadata,
        'model': None,
        'level': 'DEFAULT',
        'statusMessage': None,
    }
    observation.update(overrides)
    return observation


def make_raw_trace(trace_id=TRACE_ID, **overrides):
    return _RAW_BACKEND._raw_trace_from_body(
        trace_id,
        make_langfuse_trace_payload(trace_id=trace_id, **overrides),
    )


def make_raw_span(
    span_id=ROOT_SPAN_ID, *,
    trace_id=TRACE_ID, parent_span_id=None, **overrides,
):
    return _RAW_BACKEND._raw_span_from_obs(
        trace_id,
        make_langfuse_observation(
            span_id,
            trace_id=trace_id,
            parent_id=parent_span_id,
            **overrides,
        ),
    )


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
