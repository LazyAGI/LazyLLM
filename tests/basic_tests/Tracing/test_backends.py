import base64
import json
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import lazyllm
import lazyllm.tracing.backends.langfuse.backend as langfuse_backend_module
from lazyllm.tracing.backends.langfuse.backend import LangfuseBackend, LangfuseConsumeBackend
from lazyllm.tracing.consume.api import get_single_trace
from lazyllm.tracing.semantics import SemanticType


TRACE_ID = '0' * 32
ROOT_SPAN_ID = '1' * 16
RETRIEVER_SPAN_ID = '2' * 16
LLM_SPAN_ID = '3' * 16

LANGFUSE_HOST = 'https://langfuse.example'
LANGFUSE_PUBLIC_KEY = 'public'
LANGFUSE_SECRET_KEY = 'secret'
LANGFUSE_AUTH_HEADER = (
    'Basic ' + base64.b64encode(f'{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}'.encode('utf-8')).decode()
)

LANGFUSE_TRACE_BODY = {
    'id': TRACE_ID,
    'name': 'paginated-rag-trace',
    'sessionId': 'session-1',
    'userId': 'user-1',
    'tags': ['tag-a', 'rag'],
    'metadata': {'source': 'langfuse', 'tenant': 'tenant-a'},
}
LANGFUSE_OBSERVATIONS_PAGE_1 = {
    'data': [
        {
            'id': ROOT_SPAN_ID,
            'name': 'rag-root',
            'startTime': '2026-04-29T08:00:01Z',
            'input': {'args': ['hello'], 'kwargs': {}},
            'metadata': {
                'attributes': {
                    'lazyllm.span.kind': 'flow',
                    'lazyllm.semantic_type': SemanticType.WORKFLOW_CONTROL,
                },
            },
        },
        {
            'id': RETRIEVER_SPAN_ID,
            'parentObservationId': ROOT_SPAN_ID,
            'name': 'retrieve-docs',
            'startTime': '2026-04-29T08:00:02Z',
            'metadata': {
                'attributes': {
                    'lazyllm.span.kind': 'module',
                    'lazyllm.semantic_type': SemanticType.RETRIEVER,
                },
            },
        },
    ],
    'meta': {'totalPages': 2},
}
LANGFUSE_OBSERVATIONS_PAGE_2 = {
    'data': [
        {
            'id': LLM_SPAN_ID,
            'parentObservationId': ROOT_SPAN_ID,
            'name': 'call-llm',
            'startTime': '2026-04-29T08:00:03Z',
            'metadata': {
                'attributes': {
                    'lazyllm.span.kind': 'module',
                    'lazyllm.semantic_type': SemanticType.LLM,
                },
            },
            'model': 'gpt-4o-mini',
            'level': 'ERROR',
            'statusMessage': 'provider timeout',
        },
    ],
    'meta': {'totalPages': 2},
}
MINIMAL_LANGFUSE_TRACE_BODY = {
    'id': TRACE_ID,
    'name': 'retried-trace',
    'metadata': {},
    'tags': [],
    'observations': [],
}


class _FakeResponse:
    def __init__(self, body=None, *, status_code=200, headers=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.headers = headers or {}
        self.content = b'{}'
        self._body = body

    def json(self):
        return self._body


def _set_langfuse_env(monkeypatch):
    monkeypatch.setenv('LANGFUSE_HOST', LANGFUSE_HOST + '/')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', LANGFUSE_PUBLIC_KEY)
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', LANGFUSE_SECRET_KEY)


def test_get_single_trace_rebuilds_langfuse_payload(monkeypatch):
    _set_langfuse_env(monkeypatch)
    request = Mock(side_effect=[
        _FakeResponse(LANGFUSE_TRACE_BODY),
        _FakeResponse(LANGFUSE_OBSERVATIONS_PAGE_1),
        _FakeResponse(LANGFUSE_OBSERVATIONS_PAGE_2),
    ])
    monkeypatch.setattr(langfuse_backend_module.requests, 'request', request)

    with lazyllm.config.temp('trace_consume_backend', 'langfuse'):
        with lazyllm.config.temp('trace_consume_timeout', 3.5):
            trace = get_single_trace(TRACE_ID)

    metadata = trace.metadata
    assert trace.trace_id == TRACE_ID
    assert (metadata.name, metadata.session_id, metadata.user_id) == (
        'paginated-rag-trace', 'session-1', 'user-1',
    )
    assert metadata.tags == ['tag-a', 'rag']
    assert metadata.metadata == {'source': 'langfuse', 'tenant': 'tenant-a'}
    assert (metadata.status, metadata.error_message) == ('error', 'provider timeout')

    root = trace.execution_tree
    retriever, llm = root.children
    assert (root.step_id, root.name, root.node_type, root.semantic_type) == (
        ROOT_SPAN_ID, 'rag-root', 'flow', SemanticType.WORKFLOW_CONTROL,
    )
    assert root.raw_data.input == {'args': ['hello'], 'kwargs': {}}
    assert [child.step_id for child in root.children] == [RETRIEVER_SPAN_ID, LLM_SPAN_ID]
    assert (retriever.name, retriever.semantic_type) == ('retrieve-docs', SemanticType.RETRIEVER)
    assert (llm.semantic_data['model_name'], llm.status, llm.error_message) == (
        'gpt-4o-mini', 'error', 'provider timeout',
    )

    calls = request.call_args_list
    assert len(calls) == 3
    assert [call.args[0] for call in calls] == ['GET', 'GET', 'GET']
    assert calls[0].args[1] == f'{LANGFUSE_HOST}/api/public/traces/{TRACE_ID}'
    assert calls[0].kwargs['headers'] == {'Authorization': LANGFUSE_AUTH_HEADER, 'Accept': 'application/json'}
    assert calls[0].kwargs['timeout'] == (LangfuseConsumeBackend._CONNECT_TIMEOUT_S, 3.5)
    assert urlparse(calls[1].args[1]).path == '/api/public/observations'
    assert parse_qs(urlparse(calls[1].args[1]).query) == {
        'traceId': [TRACE_ID], 'limit': ['1000'], 'page': ['1'],
    }
    assert parse_qs(urlparse(calls[2].args[1]).query) == {
        'traceId': [TRACE_ID], 'limit': ['1000'], 'page': ['2'],
    }


def test_get_single_trace_retries_rate_limit_then_succeeds(monkeypatch):
    _set_langfuse_env(monkeypatch)

    sleeps = []
    monkeypatch.setattr(langfuse_backend_module.time, 'sleep', sleeps.append)
    request = Mock(side_effect=[
        _FakeResponse(status_code=429, headers={'Retry-After': '1.0'}),
        _FakeResponse(MINIMAL_LANGFUSE_TRACE_BODY),
    ])
    monkeypatch.setattr(langfuse_backend_module.requests, 'request', request)

    trace = get_single_trace(TRACE_ID, backend='langfuse')

    assert trace.metadata.name == 'retried-trace'
    assert request.call_count == 2
    assert sleeps == [1.0]


def test_export_backend_maps_trace_and_observation_contract():
    backend = LangfuseBackend()

    root_attrs = backend.map_attributes({
        'lazyllm.span.is_root': True,
        'lazyllm.semantic_type': SemanticType.LLM,
        'lazyllm.trace.name': 'rag-run',
        'lazyllm.trace.tags': ['prod', 'rag'],
        'lazyllm.trace.metadata.tenant': 'tenant-a',
        'session.id': 'session-1',
        'lazyllm.error.message': 'model failed',
        'gen_ai.request.model': 'gpt-4o-mini',
        'gen_ai.usage.total_tokens': 20,
    })
    child_attrs = backend.map_attributes({
        'lazyllm.semantic_type': SemanticType.RETRIEVER,
        'lazyllm.trace.name': 'must-not-promote',
    })

    assert root_attrs['langfuse.observation.type'] == 'generation'
    assert root_attrs['gen_ai.request.model'] == 'gpt-4o-mini'
    assert root_attrs['gen_ai.usage.total_tokens'] == 20
    assert root_attrs['langfuse.observation.status_message'] == 'model failed'
    assert root_attrs['langfuse.trace.name'] == 'rag-run'
    assert json.loads(root_attrs['langfuse.trace.tags']) == ['prod', 'rag']
    assert root_attrs['langfuse.trace.metadata.tenant'] == 'tenant-a'
    assert root_attrs['session.id'] == 'session-1'
    assert 'langfuse.trace.name' not in child_attrs
    assert child_attrs['langfuse.observation.type'] == 'retriever'
