from lazyllm.tools.tools.search import (
    ArxivSearch,
    SciverseSearch,
    SearchBase,
    SemanticScholarSearch,
    StackOverflowSearch,
    WikipediaSearch,
)


class FakeSearch(SearchBase):

    def __init__(self):
        super().__init__(source_name='fake', skip_auth=True)

    def search(self, query: str):
        return []

    def _fetch_content_text(self, item):
        return f"Fetched {item['title']}"


def test_search_content_preserves_identity_without_framework_citation_fields():
    provider = FakeSearch()
    item = {
        'title': 'Result',
        'url': 'https://example.test/result',
        'snippet': 'Snippet',
        'source': 'fake',
        'extra': {'doc_id': 'doc-1'},
        'citation_index': '9.1',
        'ref': '[[9.1]]',
    }

    result = provider.get_content(item)
    batch = provider.get_contents([item])

    assert result == {
        'title': 'Result',
        'url': 'https://example.test/result',
        'snippet': 'Snippet',
        'source': 'fake',
        'extra': {'doc_id': 'doc-1'},
        'content': 'Fetched Result',
    }
    assert batch == [result]
    assert item['ref'] == '[[9.1]]'
    assert result['extra'] is not item['extra']


def test_search_provider_overrides_follow_structured_content_contract():
    providers = [
        ArxivSearch(skip_auth=True),
        SciverseSearch(api_key='test'),
        SemanticScholarSearch(api_key='test'),
        StackOverflowSearch(key='test'),
        WikipediaSearch(skip_auth=True),
    ]
    item = {
        'title': 'Fallback',
        'url': '',
        'snippet': 'Snippet',
        'source': 'test',
        'extra': {},
    }

    for provider in providers:
        result = provider.get_content(item)
        assert isinstance(result, dict)
        assert set(result) == {'title', 'url', 'snippet', 'source', 'extra', 'content'}
