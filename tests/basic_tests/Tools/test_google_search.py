import importlib
from types import SimpleNamespace

import httpx
import pytest

from lazyllm.tools.tools.search import GoogleSearch

http_request = importlib.import_module('lazyllm.tools.http_request.http_request')


@pytest.mark.parametrize('kwargs, expected_url', [
    ({}, 'https://customsearch.googleapis.com/customsearch/v1'),
    ({'base_url': 'https://search.example.com/customsearch/v1'}, 'https://search.example.com/customsearch/v1'),
])
def test_google_search_sends_request_to_base_url(monkeypatch, kwargs, expected_url):
    calls = []

    def fake_request(method, url, params=None, **_):
        calls.append((method, url, params))
        payload = {'items': [{'title': 'Title', 'link': 'https://example.test', 'snippet': 'Snippet'}]}
        return httpx.Response(200, json=payload, request=httpx.Request(method, url, params=params))

    monkeypatch.setattr(http_request, 'httpx', SimpleNamespace(request=fake_request))
    results = GoogleSearch(custom_search_api_key='test', search_engine_id='cx', **kwargs).search('query')

    assert len(calls) == 1
    method, url, params = calls[0]
    assert (method, url) == ('GET', expected_url)
    assert (params['key'], params['cx'], params['q']) == ('test', 'cx', 'query')
    assert results[0]['url'] == 'https://example.test'
    assert results[0]['source'] == 'google'
