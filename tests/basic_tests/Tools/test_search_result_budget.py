from types import SimpleNamespace

import pytest

from lazyllm.tools.tools.search import SciverseSearch, TavilySearch
from lazyllm.tools.tools.search.base import _make_result
from lazyllm.tools.tools.search import sciverse_search


def response(payload):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


@pytest.mark.parametrize('size', [0, 699, 700, 701, 10000])
def test_search_snippet_budget_preserves_identity(size):
    result = _make_result('Title', 'https://example.test', '中' * size, 'test', doc_id='doc')
    assert result['snippet'] == '中' * min(size, 700)
    assert result['extra'].get('truncated', False) == (size > 700)
    assert result['extra']['doc_id'] == 'doc'
    assert result['url'] == 'https://example.test'


def test_tavily_caps_requested_content_without_hidden_originals(monkeypatch):
    provider = TavilySearch(api_key='test')
    payload = {'results': [{'title': 'Title', 'url': 'https://example.test',
                            'content': 's' * 2000, 'raw_content': 'r' * 3000}], 'answer': 'a' * 2000}
    monkeypatch.setattr(provider, '_request', lambda *a, **k: response(payload))
    results = provider.search('query', include_answer=True, include_raw_content=True)
    assert results[0]['snippet'] == 's' * 700
    assert results[0]['extra']['truncated'] is True
    assert results[0]['extra']['raw_content'] == 'r' * 700
    assert type(results[0]) is dict
    assert results[1]['snippet'] == 'a' * 700
    assert type(results[1]) is dict


@pytest.mark.parametrize('meta', [False, True])
def test_sciverse_default_search_has_one_preview(monkeypatch, meta):
    provider = SciverseSearch(api_key='test')
    payload = {'hits': [{'title': 'Paper', 'doc_id': 'doc', 'abstract': 'a' * 3000,
                         'chunk': 'c' * 1500, 'offset': 0, 'doi': '10.1/test'}]}
    monkeypatch.setattr(sciverse_search, 'httpx', SimpleNamespace(post=lambda *a, **k: response(payload)))

    def search(**kwargs):
        return provider.meta_search('query', **kwargs)['items'] if meta else provider.search('query', **kwargs)
    item = search()[0]
    assert item['snippet'] == 'c' * 700
    assert item['extra']['truncated'] is True
    assert item['extra']['content'] == 'c' * 700
    assert item['extra']['offset'] == 0
    assert item['extra']['doc_id'] == 'doc'
    assert 'content' not in search(include_content=False)[0]['extra']
    payload['hits'][0].pop('chunk')
    assert search()[0]['snippet'] == 'a' * 700


@pytest.mark.parametrize('kwargs,offset,limit', [
    ({}, 0, 16384), ({'offset': 700, 'limit': 200}, 700, 200),
])
def test_sciverse_always_requests_a_page_and_bounds_response(monkeypatch, kwargs, offset, limit):
    provider = SciverseSearch(api_key='test')
    calls = []

    def get(*args, **kw):
        calls.append(kw['params'])
        return response({'text': 'x' * 20000})
    monkeypatch.setattr(sciverse_search, 'httpx', SimpleNamespace(get=get))
    item = {'title': 'Paper', 'snippet': 's' * 2000, 'extra': {'doc_id': 'doc', 'content': 'old', 'offset': 42}}
    result = provider.get_content(item, **kwargs)
    assert calls == [{'doc_id': 'doc', 'offset': offset, 'limit': limit}]
    assert result['content'] == 'x' * limit
    assert len(result['snippet']) == 700
    assert 'content' not in result['extra']
    assert result['extra']['offset'] == 42
    assert result['extra']['content_read'] == {
        'offset': offset, 'limit': limit, 'truncated': True, 'fallback': False,
        'content_type': 'document', 'pagination_error': 'response_exceeds_limit',
    }
    assert item['extra']['content'] == 'old'


@pytest.mark.parametrize('payload', [{'text': ''}, {'text': 'x' * 700}])
def test_sciverse_empty_or_exact_page_is_not_fallback_or_known_truncation(monkeypatch, payload):
    monkeypatch.setattr(sciverse_search, 'httpx', SimpleNamespace(get=lambda *a, **k: response(payload)))
    result = SciverseSearch(api_key='test').get_content(
        {'extra': {'doc_id': 'doc'}, 'snippet': 'old'}, offset=700, limit=700)
    assert result['content'] == payload['text']
    assert result['extra']['content_read'] == {
        'offset': 700, 'limit': 700, 'truncated': False, 'fallback': False,
        'content_type': 'document', 'pagination_error': 'invalid_continuation',
    }


@pytest.mark.parametrize('source', ['content', 'snippet', 'url'])
def test_sciverse_failure_fallback_is_bounded_and_not_a_document_page(monkeypatch, source):
    def fail(*args, **kwargs):
        raise RuntimeError('unavailable')
    monkeypatch.setattr(sciverse_search, 'httpx', SimpleNamespace(get=fail))
    provider = SciverseSearch(api_key='test')
    monkeypatch.setattr(provider, '_fetch_content_text', lambda item: 'u' * 2000)
    item = {'extra': {'doc_id': 'doc'}}
    if source == 'content':
        item['extra']['content'] = 'c' * 2000
    elif source == 'snippet':
        item['snippet'] = 's' * 2000
    result = provider.get_content(item, offset=1000, limit=100)
    assert len(result['content']) == 100
    assert result['extra']['content_read'] == {
        'offset': None, 'limit': 100, 'truncated': True, 'fallback': True,
        'content_type': 'search_preview',
    }


@pytest.mark.parametrize('text,more,next_offset', [('short', True, 1400), ('tail', False, 1200), ('', False, 700)])
def test_sciverse_preserves_server_cursor_independent_of_text_length(monkeypatch, text, more, next_offset):
    payload = {'text': text, 'more': more, 'next_offset': next_offset}
    monkeypatch.setattr(sciverse_search, 'httpx', SimpleNamespace(get=lambda *a, **k: response(payload)))
    result = SciverseSearch(api_key='test').get_content({'extra': {'doc_id': 'doc'}}, offset=700, limit=700)
    read = result['extra']['content_read']
    assert read['more'] is more
    assert read['next_offset'] == next_offset


@pytest.mark.parametrize('payload', [
    {'text': 'x' * 701, 'more': False, 'next_offset': 1401},
    {'text': 'x', 'more': True, 'next_offset': 700},
    {'text': 'x', 'more': True, 'next_offset': 699},
    {'text': 'x', 'more': False, 'next_offset': 699},
    {'text': 'x', 'more': 'false', 'next_offset': 1400},
    {'text': 'x', 'more': True, 'next_offset': '1400'},
    {'text': 'x', 'more': False, 'next_offset': True},
    {'text': 'x', 'more': True},
    {'more': False, 'next_offset': 1400},
])
def test_sciverse_omits_unsafe_pagination(monkeypatch, payload):
    monkeypatch.setattr(sciverse_search, 'httpx', SimpleNamespace(get=lambda *a, **k: response(payload)))
    item = {'snippet': 'fallback', 'extra': {'doc_id': 'doc', 'content_read': {'more': False, 'next_offset': 9999}}}
    result = SciverseSearch(api_key='test').get_content(item, offset=700, limit=700)
    read = result['extra']['content_read']
    assert 'more' not in read
    assert 'next_offset' not in read


@pytest.mark.parametrize('field', ['raw_content', 'content', 'answer'])
def test_only_extra_truncation_is_marked(field):
    result = _make_result('Title', 'https://example.test', 'short', **{field: '🙂' * 701})
    assert result['extra'][field] == '🙂' * 700
    assert result['extra']['truncated'] is True


@pytest.mark.parametrize('kwargs', [{'offset': -1}, {'limit': 0}, {'limit': True}, {'offset': None}])
def test_sciverse_rejects_invalid_window_before_request(monkeypatch, kwargs):
    monkeypatch.setattr(sciverse_search, 'httpx', SimpleNamespace(get=lambda *a, **k: pytest.fail('network')))
    with pytest.raises(ValueError):
        SciverseSearch(api_key='test').get_content({'extra': {'doc_id': 'doc'}}, **kwargs)
