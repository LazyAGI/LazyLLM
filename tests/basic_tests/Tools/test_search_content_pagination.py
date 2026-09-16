from types import SimpleNamespace

import pytest

from lazyllm.tools.tools.search import (
    SearchBase, GoogleSearch, BingSearch, BochaSearch, TavilySearch, ArxivSearch,
    SemanticScholarSearch, WikipediaSearch, StackOverflowSearch, GoogleBooksSearch, TencentSearch,
)
from lazyllm.tools.tools.search import arxiv_search, wikipedia_search, stackoverflow_search


def response(payload):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)


@pytest.mark.parametrize('provider', [
    GoogleSearch, BingSearch, BochaSearch, TavilySearch, ArxivSearch, SemanticScholarSearch,
    WikipediaSearch, StackOverflowSearch, GoogleBooksSearch, TencentSearch,
])
def test_all_local_content_readers_use_shared_pagination(provider):
    assert provider.get_content is SearchBase.get_content
    assert provider.get_contents is SearchBase.get_contents


def test_webpage_pages_reconstruct_text_and_drop_repeated_content(monkeypatch):
    provider = TavilySearch(api_key='test')
    text = '中文😀abc' * 501
    monkeypatch.setattr(provider, '_fetch_content_text', lambda item: text)
    item = {'title': 'Page', 'url': 'https://example.test', 'snippet': 's' * 1000,
            'extra': {'raw_content': text, 'content': text, 'doc_id': 'doc'}}
    pages = []
    offset = 0
    while True:
        result = provider.get_content(item, offset=offset)
        read = result['extra']['content_read']
        assert len(result['content']) <= 700
        assert len(result['snippet']) == 700
        assert 'raw_content' not in result['extra'] and 'content' not in result['extra']
        assert result['extra']['doc_id'] == 'doc'
        pages.append(result['content'])
        if not read['more']:
            break
        assert read['next_offset'] > offset
        offset = read['next_offset']
    assert ''.join(pages) == text
    assert item['extra']['raw_content'] == text
    end = provider.get_content(item, offset=len(text))
    assert end['content'] == '' and end['extra']['content_read']['more'] is False
    assert provider.get_content(item, limit=len(text))['content'] == text[:700]
    batch = provider.get_contents([item, item], offset=700, limit=100)
    assert all(r['content'] == text[700:800] for r in batch)


def test_failed_webpage_does_not_claim_end(monkeypatch):
    provider = TavilySearch(api_key='test')
    monkeypatch.setattr(provider, '_fetch_content_text', lambda item: '')
    result = provider.get_content({'extra': {'content_read': {'more': False}}}, offset=100)
    read = result['extra']['content_read']
    assert read['fallback'] is True and read['offset'] is None
    assert 'more' not in read and 'next_offset' not in read


def test_arxiv_pages_are_explicitly_abstracts(monkeypatch):
    abstract = 'Abstract text. ' * 200
    xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><summary>' + abstract + '</summary></entry></feed>'
    monkeypatch.setattr(arxiv_search, 'httpx', SimpleNamespace(
        get=lambda *a, **k: SimpleNamespace(text=xml, raise_for_status=lambda: None)))
    provider = ArxivSearch()
    item = {'url': 'https://arxiv.org/abs/2401.12345'}
    first = provider.get_content(item)
    assert first['extra']['content_read']['content_type'] == 'abstract'
    assert first['extra']['content_read']['more'] is True
    assert provider.get_content(item, offset=700)['content'] == abstract.strip()[700:1400]


def test_semantic_scholar_abstract_and_fallback(monkeypatch):
    provider = SemanticScholarSearch(api_key='test')
    monkeypatch.setattr(provider, '_request', lambda *a, **k: response({'abstract': 'a' * 1500}))
    item = {'extra': {'paperId': 'paper'}}
    result = provider.get_content(item, offset=700)
    assert result['content'] == 'a' * 700
    assert result['extra']['content_read']['content_type'] == 'abstract'
    fallback = provider.get_content({'snippet': 'preview'})
    assert fallback['extra']['content_read']['fallback'] is True
    assert fallback['extra']['content_read']['content_type'] == 'search_preview'
    assert 'more' not in fallback['extra']['content_read']


def test_wikipedia_article_pagination(monkeypatch):
    payload = {'query': {'pages': {'1': {'extract': 'w' * 1500}}}}
    monkeypatch.setattr(wikipedia_search, 'httpx', SimpleNamespace(get=lambda *a, **k: response(payload)))
    result = WikipediaSearch().get_content({'extra': {'pageid': 1}}, offset=1400)
    assert result['content'] == 'w' * 100
    assert result['extra']['content_read']['more'] is False
    assert result['extra']['content_read']['content_type'] == 'encyclopedia_article'


def test_stackoverflow_question_and_answer_pagination(monkeypatch):
    def get(url, **kwargs):
        if '/questions/' in url:
            return response({'items': [{'body': '<p>' + 'q' * 900 + '</p>', 'accepted_answer_id': 2}]})
        return response({'items': [{'body': '<p>' + 'a' * 900 + '</p>'}]})
    monkeypatch.setattr(stackoverflow_search, 'httpx', SimpleNamespace(get=get))
    provider = StackOverflowSearch(key='test')
    result = provider.get_content({'url': 'https://stackoverflow.com/questions/1'}, offset=900)
    assert 'Accepted Answer' in result['content']
    assert len(result['content']) == 700
    assert result['extra']['content_read']['content_type'] == 'question_and_accepted_answer'
