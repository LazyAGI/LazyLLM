import pytest

import lazyllm.tools.tools.search.base as search_base
from lazyllm.common import KeyAuthError
from lazyllm.tools.tools import FirecrawlSearch


class _Response:

    def __init__(self, payload=None, text='', status_code=200):
        self._payload = payload
        self._text = text
        self.status_code = status_code

    @property
    def text(self):
        return self._text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f'HTTP {self.status_code}')


class _FakeHttpx:

    def __init__(self, handler, html='<html><body><p>Hello <b>world</b></p></body></html>'):
        self._handler = handler
        self._html = html
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._handler(method, url, kwargs)

    def get(self, url, **kwargs):
        self.calls.append(('GET', url, kwargs))
        return _Response({}, self._html)


@pytest.fixture
def fake_http(monkeypatch):

    def install(handler, html=None):
        fake = _FakeHttpx(handler) if html is None else _FakeHttpx(handler, html)
        monkeypatch.setattr(search_base, 'httpx', fake)
        return fake

    return install


def _search_payload(**buckets):
    return {'success': True, 'data': buckets}


def test_search_sends_bearer_auth_and_maps_web_results(fake_http):
    fake = fake_http(lambda method, url, kwargs: _Response(_search_payload(web=[{
        'title': 'Result', 'description': 'Snippet', 'url': 'https://example.test/1',
    }])))

    results = FirecrawlSearch(api_key='fc-test')('agent tooling', limit=2)

    assert results == [{
        'title': 'Result',
        'url': 'https://example.test/1',
        'snippet': 'Snippet',
        'source': 'firecrawl',
    }]
    method, url, kwargs = fake.calls[0]
    assert (method, url) == ('POST', 'https://api.firecrawl.dev/v2/search')
    assert kwargs['headers']['Authorization'] == 'Bearer fc-test'
    assert kwargs['json'] == {'query': 'agent tooling', 'limit': 2, 'sources': [{'type': 'web'}]}


def test_search_keeps_requested_source_order_and_marks_news(fake_http):
    fake_http(lambda method, url, kwargs: _Response(_search_payload(
        web=[{'title': 'Page', 'description': 'Web hit', 'url': 'https://example.test/page'}],
        news=[{'title': 'Story', 'snippet': 'News hit', 'url': 'https://example.test/story',
               'date': '2026-11-01'}],
    )))

    results = FirecrawlSearch(api_key='fc-test')('release', sources=['news', 'web'])

    assert [item['title'] for item in results] == ['Story', 'Page']
    assert results[0]['snippet'] == 'News hit'
    assert results[0]['extra'] == {'result_type': 'news', 'published_at': '2026-11-01'}
    assert 'extra' not in results[1]


def test_scrape_markdown_requests_scrape_options_and_truncates_preview(fake_http):
    fake = fake_http(lambda method, url, kwargs: _Response(_search_payload(web=[{
        'title': 'Page', 'description': 'Snippet', 'url': 'https://example.test/page',
        'markdown': 'm' * 900,
    }])))

    results = FirecrawlSearch(api_key='fc-test')('agent tooling', scrape_markdown=True)

    assert fake.calls[0][2]['json']['scrapeOptions'] == {
        'formats': [{'type': 'markdown'}], 'onlyMainContent': True,
    }
    assert results[0]['extra']['raw_content'] == 'm' * 700
    assert results[0]['extra']['truncated'] is True


def test_invalid_arguments_fail_before_any_request(fake_http):
    fake = fake_http(lambda method, url, kwargs: _Response(_search_payload(web=[])))
    provider = FirecrawlSearch(api_key='fc-test')

    with pytest.raises(ValueError, match='mutually exclusive'):
        provider('q', include_domains=['a.test'], exclude_domains=['b.test'], raise_on_error=True)
    with pytest.raises(ValueError, match='unsupported source'):
        provider('q', sources=['images'], raise_on_error=True)
    for limit in (0, 101, '5'):
        with pytest.raises(ValueError, match='limit must be an integer'):
            provider('q', limit=limit, raise_on_error=True)

    assert fake.calls == []


def test_api_error_is_reported_only_when_asked(fake_http):
    fake_http(lambda method, url, kwargs: _Response({
        'success': False, 'error': 'quota exceeded',
    }))
    provider = FirecrawlSearch(api_key='fc-test')

    assert provider('q') == []
    with pytest.raises(RuntimeError, match='quota exceeded'):
        provider('q', raise_on_error=True)


def test_legacy_flat_payload_still_maps(fake_http):
    fake_http(lambda method, url, kwargs: _Response({'success': True, 'data': [{
        'title': 'Legacy', 'description': 'Old shape', 'url': 'https://example.test/legacy',
    }]}))

    assert FirecrawlSearch(api_key='fc-test')('q')[0]['title'] == 'Legacy'


def test_get_content_reads_markdown_through_scrape(fake_http):
    fake = fake_http(lambda method, url, kwargs: _Response(
        {'success': True, 'data': {'markdown': '# Title\n\nBody text'}}
        if url.endswith('/scrape') else _search_payload(web=[])
    ))
    provider = FirecrawlSearch(api_key='fc-test')
    item = {'title': 'Page', 'url': 'https://example.test/page', 'snippet': 'Snippet', 'source': 'firecrawl'}

    result = provider.get_content(item, offset=9, limit=9)

    assert result['content'] == 'Body text'
    assert result['extra']['content_read']['offset'] == 9
    assert result['extra']['content_read']['more'] is False
    method, url, kwargs = fake.calls[0]
    assert (method, url) == ('POST', 'https://api.firecrawl.dev/v2/scrape')
    assert kwargs['json'] == {
        'url': 'https://example.test/page',
        'formats': [{'type': 'markdown'}],
        'onlyMainContent': True,
    }


def test_get_content_falls_back_to_shared_html_reader(fake_http):
    def handler(method, url, kwargs):
        raise RuntimeError('scrape rejected')

    fake_http(handler)

    result = FirecrawlSearch(api_key='fc-test').get_content(
        {'title': 'Page', 'url': 'https://example.test/page', 'snippet': '', 'source': 'firecrawl'})

    assert result['content'] == 'Hello world'
    assert result['extra']['content_read']['fallback'] is False


def test_get_content_keeps_auth_error_visible(fake_http):
    fake_http(lambda method, url, kwargs: _Response({'success': False}, status_code=401))

    with pytest.raises(KeyAuthError):
        FirecrawlSearch(api_key='fc-bad').get_content(
            {'title': 'Page', 'url': 'https://example.test/page', 'snippet': '', 'source': 'firecrawl'})


def test_provider_is_exported_from_public_namespaces():
    from lazyllm.tools.tools.search import FirecrawlSearch as package_level

    assert package_level is FirecrawlSearch
    assert issubclass(FirecrawlSearch, search_base.SearchBase)


def test_response_without_data_returns_no_results(fake_http):
    fake_http(lambda method, url, kwargs: _Response({'success': True}))

    assert FirecrawlSearch(api_key='fc-test')('q') == []
