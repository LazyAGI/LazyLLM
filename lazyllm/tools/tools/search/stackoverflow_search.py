from typing import List, Optional

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class StackOverflowSearch(SearchBase):

    def __init__(self, site: str = 'stackoverflow', key: Optional[str] = None,
                 timeout: int = 10, source_name: str = 'stackoverflow'):
        super().__init__(source_name=source_name)
        self._site = site
        self._key = key
        self._timeout = timeout

    def search(self, query: str, count: int = 10,
               sort: str = 'relevance') -> List[dict]:
        url = 'https://api.stackexchange.com/2.3/search/advanced'
        params = {
            'order': 'desc',
            'sort': sort,
            'q': query,
            'site': self._site,
            'pagesize': min(count, 100),
        }
        if self._key:
            params['key'] = self._key
        try:
            resp = httpx.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        items = data.get('items') or []
        out: List[dict] = []
        for it in items:
            link = it.get('link', '')
            title = it.get('title', '')
            snippet = it.get('body', '')[:500] if it.get('body') else ''
            if snippet and len(it.get('body', '')) > 500:
                snippet = snippet + '...'
            out.append(_make_result(
                title=title,
                url=link,
                snippet=snippet,
                source=self.source_name,
                score=it.get('score'),
                answer_count=it.get('answer_count'),
                is_answered=it.get('is_answered'),
            ))
        return out
