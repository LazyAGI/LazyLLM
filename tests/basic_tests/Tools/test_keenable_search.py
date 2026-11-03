from types import SimpleNamespace

from lazyllm.tools.tools.search import KeenableSearch, SearchBase


def _capture(monkeypatch, provider, payload):
    calls = []

    def fake_execute(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return SimpleNamespace(json=lambda: payload)

    monkeypatch.setattr(provider, '_http_execute', fake_execute)
    return calls


def test_keenable_uses_public_endpoint_without_key(monkeypatch):
    provider = KeenableSearch()
    calls = _capture(monkeypatch, provider, {'results': [
        {'title': 'A', 'url': 'https://a.example', 'description': 'short', 'snippet': 'long'},
        {'title': 'No URL', 'description': 'dropped'},
        {'title': 'B', 'url': 'https://b.example', 'snippet': 'x' * 2000},
    ]})

    results = provider.search('lazyllm', max_results=3)

    method, url, kwargs = calls[0]
    assert method == 'POST'
    assert url == 'https://api.keenable.ai/v1/search/public'
    assert kwargs['json'] == {'query': 'lazyllm', 'max_results': 3}
    assert kwargs['headers']['X-Keenable-Title'] == 'LazyLLM'
    assert 'X-API-Key' not in kwargs['headers']
    assert [r['url'] for r in results] == ['https://a.example', 'https://b.example']
    assert results[0]['snippet'] == 'short'
    assert results[0]['source'] == 'keenable'
    assert results[1]['extra']['truncated'] is True


def test_keenable_uses_keyed_endpoint_with_key(monkeypatch):
    provider = KeenableSearch(api_key='test-key')
    calls = _capture(monkeypatch, provider, {'results': []})

    provider.search('lazyllm')

    _, url, kwargs = calls[0]
    assert url == 'https://api.keenable.ai/v1/search'
    assert kwargs['headers']['X-API-Key'] == 'test-key'


def test_keenable_uses_shared_content_reader():
    assert KeenableSearch.get_content is SearchBase.get_content
