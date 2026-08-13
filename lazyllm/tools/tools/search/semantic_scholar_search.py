from typing import Any, Dict, List, Optional

from lazyllm.common import ApiKeyHeaderStrategy

from .base import SearchBase, _make_content_result, _make_result


class SemanticScholarSearch(SearchBase):

    def __init__(self, api_key: Optional[str] = None,
                 timeout: int = 15, source_name: str = 'semantic_scholar'):
        super().__init__(
            source_name=source_name, api_key=api_key,
            auth_strategy=ApiKeyHeaderStrategy('x-api-key'),
        )
        self._timeout = timeout
        self._base = 'https://api.semanticscholar.org/graph/v1'

    def get_content(self, item: Dict[str, Any]) -> Dict[str, Any]:
        extra = item.get('extra') or {}
        paper_id = extra.get('paperId')
        if not paper_id:
            snippet = item.get('snippet', '')
            if snippet:
                return _make_content_result(item, snippet)
            return super().get_content(item)
        url = f'{self._base}/paper/{paper_id}'
        try:
            resp = self._request('GET', url, params={'fields': 'abstract'}, timeout=self._timeout)
            data = resp.json()
        except Exception:
            snippet = item.get('snippet') or ''
            return _make_content_result(item, snippet) if snippet else super().get_content(item)
        content = (data.get('abstract') or '').strip() or (item.get('snippet') or '')
        return _make_content_result(item, content)

    def search(self, query: str, limit: int = 10,
               fields: Optional[str] = None) -> List[dict]:
        url = f'{self._base}/paper/search'
        params = {
            'query': query,
            'limit': min(limit, 100),
            'fields': fields or 'title,url,abstract,authors,year,citationCount',
        }
        resp = self._request('GET', url, params=params, timeout=self._timeout)
        data = resp.json()
        items = data.get('data') or []
        out: List[dict] = []
        for it in items:
            title = it.get('title', '')
            url = it.get('url') or f'https://www.semanticscholar.org/paper/{it.get("paperId", "")}'
            snippet = it.get('abstract') or ''
            authors = it.get('authors')
            if authors:
                author_names = [a.get('name', '') for a in authors if isinstance(a, dict)]
                extra = {'authors': author_names, 'year': it.get('year'), 'citationCount': it.get('citationCount'),
                         'paperId': it.get('paperId')}
            else:
                extra = {'year': it.get('year'), 'citationCount': it.get('citationCount'),
                         'paperId': it.get('paperId')}
            out.append(_make_result(
                title=title,
                url=url,
                snippet=snippet,
                source=self.source_name,
                **{k: v for k, v in extra.items() if v is not None},
            ))
        return out
