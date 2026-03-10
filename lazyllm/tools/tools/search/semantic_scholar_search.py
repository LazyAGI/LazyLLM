from typing import List, Optional

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class SemanticScholarSearch(SearchBase):

    def __init__(self, api_key: Optional[str] = None,
                 timeout: int = 15, source_name: str = 'semantic_scholar'):
        super().__init__(source_name=source_name)
        self._api_key = api_key
        self._timeout = timeout
        self._base = 'https://api.semanticscholar.org/graph/v1'

    def search(self, query: str, limit: int = 10,
               fields: Optional[str] = None) -> List[dict]:
        url = f'{self._base}/paper/search'
        params = {
            'query': query,
            'limit': min(limit, 100),
            'fields': fields or 'title,url,abstract,authors,year,citationCount',
        }
        headers = {}
        if self._api_key:
            headers['x-api-key'] = self._api_key
        try:
            resp = httpx.get(url, params=params, headers=headers or None, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        items = data.get('data') or []
        out: List[dict] = []
        for it in items:
            title = it.get('title', '')
            url = it.get('url') or f'https://www.semanticscholar.org/paper/{it.get("paperId", "")}'
            snippet = it.get('abstract') or ''
            authors = it.get('authors')
            if authors:
                author_names = [a.get('name', '') for a in authors if isinstance(a, dict)]
                extra = {'authors': author_names, 'year': it.get('year'), 'citationCount': it.get('citationCount')}
            else:
                extra = {'year': it.get('year'), 'citationCount': it.get('citationCount')}
            out.append(_make_result(
                title=title,
                url=url,
                snippet=snippet,
                source=self.source_name,
                **{k: v for k, v in extra.items() if v is not None},
            ))
        return out
