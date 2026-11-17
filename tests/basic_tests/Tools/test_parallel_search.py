import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import lazyllm
from lazyllm.tools.tools import ParallelSearch, TavilySearch, WikipediaSearch
from lazyllm.tools.tools.search import SearchBase


def response(data=None, *, error=False, text=False):
    data = data if data is not None else {'results': [
        {'title': 'Reference', 'url': 'https://example.test/doc', 'excerpts': ['x' * 800]}]}
    return SimpleNamespace(isError=error, structuredContent=None if text else data,
                           content=[SimpleNamespace(type='text', text=json.dumps(data))])


@pytest.mark.parametrize('text', [False, True])
def test_exports_result_contract_and_query_arguments(monkeypatch, text):
    assert lazyllm.ParallelSearch is ParallelSearch
    assert lazyllm.tools.ParallelSearch is ParallelSearch
    provider = ParallelSearch()
    calls = []

    async def call(name, arguments):
        calls.append((name, arguments))
        return response(text=text)

    monkeypatch.setattr(provider._client, 'call_tool', call)
    result = provider.search('Find a reference', max_results=1, search_queries=['reference documentation'])
    assert calls == [('web_search', {
        'objective': 'Find a reference', 'search_queries': ['reference documentation']})]
    assert result == [{'title': 'Reference', 'url': 'https://example.test/doc', 'snippet': 'x' * 700,
                       'source': 'parallel', 'extra': {'truncated': True}}]
    assert provider.__key_source__() is True
    assert asyncio.run(_invoke(provider)) == result


async def _invoke(provider):
    return provider('Find a reference')


def test_disabled_never_calls_search_or_content_transport(monkeypatch):
    provider = ParallelSearch(enabled=False)

    async def fail(*args, **kwargs):
        pytest.fail('disabled search made a request')

    def fail_fetch(*args, **kwargs):
        pytest.fail('disabled fetch made a request')

    monkeypatch.setattr(provider._client, 'call_tool', fail)
    monkeypatch.setattr(httpx, 'get', fail_fetch)
    assert provider('query') == []
    assert provider.get_content({'url': 'https://example.test'})['content'] == ''
    assert provider.get_contents([{'url': 'https://example.test'}])[0]['content'] == ''
    assert provider.__key_source__() is False


def test_errors_are_not_retried_or_replaced(monkeypatch):
    provider = ParallelSearch()
    calls = []

    async def call(*args, **kwargs):
        calls.append(args)
        return response(error=True)

    monkeypatch.setattr(provider._client, 'call_tool', call)
    with pytest.raises(RuntimeError, match='MCP tool error'):
        provider.search('reference')
    assert provider('reference') == []
    with pytest.raises(RuntimeError, match='MCP tool error'):
        provider('reference', raise_on_error=True)
    assert len(calls) == 3

    async def denied(*args, **kwargs):
        raise PermissionError('denied')

    monkeypatch.setattr(provider._client, 'call_tool', denied)
    with pytest.raises(PermissionError, match='denied'):
        provider.search('reference')


@pytest.mark.parametrize('kwargs', [{'max_results': 0}, {'max_results': True},
                                    {'search_queries': []}, {'search_queries': ['']}])
def test_invalid_options_fail_before_network(kwargs):
    with pytest.raises(ValueError):
        ParallelSearch().search('reference', **kwargs)


def test_anonymous_contract_does_not_change_other_credentials():
    with pytest.raises(TypeError):
        ParallelSearch(api_key='explicit')
    with pytest.raises(ValueError, match='dynamic credential is not configured'):
        TavilySearch().search('reference')
    assert TavilySearch(api_key='explicit').get_current_token() == 'explicit'
    assert WikipediaSearch().__key_source__() is True


def test_content_headers_and_budgets(monkeypatch):
    seen = []

    def fetch(url, **kwargs):
        seen.append(kwargs)
        return httpx.Response(200, text='<p>' + 'a' * 1000 + '</p>', request=httpx.Request('GET', url))

    monkeypatch.setattr(httpx, 'get', fetch)
    provider = ParallelSearch()
    item = {'title': 'Reference', 'url': 'https://example.test/doc'}
    assert provider.get_content(item, offset=10, limit=50)['content'] == 'a' * 50
    assert provider.get_contents([item, item], limit=100)[0]['content'] == 'a' * 50
    assert all(call['headers'] == {'User-Agent': provider._UA} for call in seen)
    assert provider._client._headers == {'User-Agent': provider._UA}
    SearchBase(skip_auth=True).get_content(item)
    assert seen[-1]['headers'] is None


def test_tool_manager_preserves_permissions_and_disabled_choice(monkeypatch):
    from lazyllm.tools import AuthorizationDecision, ToolManager

    provider = ParallelSearch()

    async def fail(*args, **kwargs):
        pytest.fail('denied tool made a request')

    monkeypatch.setattr(provider._client, 'call_tool', fail)
    manager = ToolManager([provider])
    manager.activate_group('ParallelSearch')
    names = {item['function']['name'] for item in manager.tools_description}
    assert names == {'ParallelSearch_search', 'ParallelSearch_get_content', 'ParallelSearch_get_contents'}
    batch = manager.prepare_tool_calls(
        {'function': {'name': 'ParallelSearch_search', 'arguments': {'query': 'reference'}}},
        authorization_policy=lambda _: AuthorizationDecision.DENY)
    result = manager.execute_prepared(batch)
    assert result.records[0].disposition == 'skipped'
    assert result.records[0].reason == 'authorization_rejected'
    disabled = ToolManager([ParallelSearch(enabled=False)])
    disabled.activate_group('ParallelSearch')
    assert disabled.tools_description == []
