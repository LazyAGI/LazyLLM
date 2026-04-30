import json
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest

import lazyllm.tracing.backends as backends_module
import lazyllm.tracing.backends.langfuse.backend as langfuse_backend_module
from lazyllm.tracing.backends.langfuse.backend import LangfuseBackend, LangfuseConsumeBackend
from lazyllm.tracing.consume.errors import ConsumeBackendError, TraceNotFound
from lazyllm.tracing.semantics import SemanticType

from .conftest import (
    CHILD_SPAN_ID,
    LANGFUSE_AUTH_HEADER,
    ROOT_SPAN_ID,
    TRACE_ID,
    langfuse_trace_url,
    make_langfuse_observation,
    make_langfuse_trace_payload,
    make_response,
    set_langfuse_env,
)


def test_export_backend_maps_trace_and_observation_contract():
    backend = LangfuseBackend()

    root_attrs = backend.map_attributes({
        'lazyllm.span.is_root': True,
        'lazyllm.semantic_type': SemanticType.LLM,
        'lazyllm.trace.name': 'rag-run',
        'lazyllm.trace.tags': ['prod', 'rag'],
        'lazyllm.trace.metadata.tenant': 'tenant-a',
        'session.id': 'session-1',
        'user.id': 'user-1',
        'lazyllm.io.input': {'query': 'hello'},
        'lazyllm.io.output': {'answer': 'world'},
        'lazyllm.error.message': 'model failed',
        'gen_ai.request.model': 'gpt-4o-mini',
        'gen_ai.usage.input_tokens': 12,
        'gen_ai.usage.output_tokens': 8,
        'gen_ai.usage.total_tokens': 20,
    })
    child_attrs = backend.map_attributes({
        'lazyllm.semantic_type': SemanticType.RETRIEVER,
        'lazyllm.trace.name': 'must-not-promote',
        'lazyllm.io.input': 'retriever input',
        'lazyllm.io.output': [{'doc': 'doc-1'}],
    })

    assert root_attrs['langfuse.observation.type'] == 'generation'
    assert root_attrs['gen_ai.request.model'] == 'gpt-4o-mini'
    assert root_attrs['gen_ai.usage.total_tokens'] == 20
    assert root_attrs['langfuse.observation.status_message'] == 'model failed'
    assert root_attrs['langfuse.trace.name'] == 'rag-run'
    assert json.loads(root_attrs['langfuse.trace.tags']) == ['prod', 'rag']
    assert root_attrs['langfuse.trace.metadata.tenant'] == 'tenant-a'
    assert root_attrs['session.id'] == 'session-1'
    assert root_attrs['user.id'] == 'user-1'
    assert root_attrs['langfuse.trace.input'] == {'query': 'hello'}
    assert root_attrs['langfuse.trace.output'] == {'answer': 'world'}

    assert child_attrs == {
        'langfuse.observation.type': 'retriever',
        'langfuse.observation.input': 'retriever input',
        'langfuse.observation.output': [{'doc': 'doc-1'}],
    }


def test_consume_fetch_trace_payload_with_inline_observations(monkeypatch):
    set_langfuse_env(monkeypatch)
    response = make_response(make_langfuse_trace_payload(
        name='inline-trace', sessionId='session-1', userId='user-1',
        tags=['tag-a'], metadata={'source': 'langfuse'},
        input={'query': 'hello'}, output={'answer': 'world'},
        public=True, observations=[
            make_langfuse_observation(
                ROOT_SPAN_ID,
                name='root-step',
                attrs={
                    'langfuse.observation.type': 'chain',
                    'gen_ai.request.model': 'metadata-model',
                },
                model='top-level-model',
            ),
            make_langfuse_observation(
                CHILD_SPAN_ID,
                parent_id=ROOT_SPAN_ID,
                obs_type='GENERATION',
                model='gpt-4o-mini',
                level='ERROR',
                statusMessage='provider timeout',
            ),
        ],
    ))
    request = Mock(return_value=response)
    monkeypatch.setattr(langfuse_backend_module.requests, 'request', request)

    payload = LangfuseConsumeBackend().fetch_trace_payload(TRACE_ID, timeout_seconds=3.5)

    assert payload.trace.trace_id == TRACE_ID
    assert payload.trace.name == 'inline-trace'
    assert payload.trace.session_id == 'session-1'
    assert payload.trace.user_id == 'user-1'
    assert payload.trace.tags == ['tag-a']
    assert payload.trace.metadata == {'source': 'langfuse'}
    assert payload.trace.input == {'query': 'hello'}
    assert payload.trace.output == {'answer': 'world'}
    assert payload.trace.raw == {'public': True}

    root, child = payload.spans
    assert root.name == 'root-step'
    assert root.attributes['langfuse.observation.type'] == 'chain'
    assert root.attributes['gen_ai.request.model'] == 'metadata-model'
    assert root.metadata['resourceAttributes'] == {'service.name': 'lazyllm'}
    assert child.parent_span_id == ROOT_SPAN_ID
    assert child.attributes['langfuse.observation.type'] == 'GENERATION'
    assert child.attributes['gen_ai.request.model'] == 'gpt-4o-mini'
    assert child.status == 'error'
    assert child.error_message == 'provider timeout'
    request.assert_called_once()
    args, kwargs = request.call_args
    assert args == ('GET', langfuse_trace_url())
    assert kwargs['timeout'] == (5.0, 3.5)
    assert kwargs['headers']['Authorization'] == LANGFUSE_AUTH_HEADER
    assert kwargs['headers']['Accept'] == 'application/json'


def test_consume_fetch_spans_paginates_observations(monkeypatch):
    set_langfuse_env(monkeypatch)
    responses = [
        make_response(make_langfuse_trace_payload(name='paginated-trace')),
        make_response({
            'data': [make_langfuse_observation(ROOT_SPAN_ID, name='page-one-root')],
            'meta': {'page': 1, 'limit': 1000, 'totalItems': 2, 'totalPages': 2},
        }),
        make_response({
            'data': [make_langfuse_observation(
                CHILD_SPAN_ID,
                parent_id=ROOT_SPAN_ID,
                name='page-two-child',
            )],
            'meta': {'page': 2, 'limit': 1000, 'totalItems': 2, 'totalPages': 2},
        }),
    ]
    request = Mock(side_effect=responses)
    monkeypatch.setattr(langfuse_backend_module.requests, 'request', request)

    payload = LangfuseConsumeBackend().fetch_trace_payload(TRACE_ID)

    assert payload.trace.trace_id == TRACE_ID
    assert payload.trace.name == 'paginated-trace'
    assert [span.name for span in payload.spans] == ['page-one-root', 'page-two-child']
    assert payload.spans[0].span_id == ROOT_SPAN_ID
    assert payload.spans[1].span_id == CHILD_SPAN_ID
    assert payload.spans[1].parent_span_id == ROOT_SPAN_ID

    calls = request.call_args_list
    assert [call.args[0] for call in calls] == ['GET', 'GET', 'GET']
    assert calls[0].args[1] == langfuse_trace_url()

    obs_url_1 = calls[1].args[1]
    obs_url_2 = calls[2].args[1]
    assert urlparse(obs_url_1).path == '/api/public/observations'
    assert parse_qs(urlparse(obs_url_1).query) == {
        'traceId': [TRACE_ID], 'limit': ['1000'], 'page': ['1'],
    }
    assert parse_qs(urlparse(obs_url_2).query) == {
        'traceId': [TRACE_ID], 'limit': ['1000'], 'page': ['2'],
    }


@pytest.mark.parametrize(
    ('status_code', 'expected_error', 'expected_message'),
    [
        (404, TraceNotFound, TRACE_ID),
        (401, ConsumeBackendError, 'authentication failed'),
    ],
)
def test_consume_http_errors(monkeypatch, status_code, expected_error, expected_message):
    set_langfuse_env(monkeypatch)
    monkeypatch.setattr(
        langfuse_backend_module.requests, 'request',
        Mock(return_value=make_response(status_code=status_code, body={'message': 'error'})),
    )

    with pytest.raises(expected_error, match=expected_message):
        LangfuseConsumeBackend().fetch_trace_payload(TRACE_ID)


def test_consume_retries_on_rate_limit_then_succeeds(monkeypatch):
    set_langfuse_env(monkeypatch)
    sleeps = []
    monkeypatch.setattr(langfuse_backend_module.time, 'sleep', sleeps.append)

    responses = [
        make_response(status_code=429, headers={'Retry-After': '0'}),
        make_response(make_langfuse_trace_payload(name='retried-trace')),
    ]
    request = Mock(side_effect=responses)
    monkeypatch.setattr(langfuse_backend_module.requests, 'request', request)

    trace = LangfuseConsumeBackend().fetch_trace(TRACE_ID)

    assert trace.name == 'retried-trace'
    assert request.call_count == 2
