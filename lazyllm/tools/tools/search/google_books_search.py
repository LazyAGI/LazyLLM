from typing import List, Optional

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class GoogleBooksSearch(SearchBase):

    def __init__(self, api_key: Optional[str] = None,
                 timeout: int = 10, source_name: str = 'google_books'):
        super().__init__(source_name=source_name)
        self._api_key = api_key
        self._timeout = timeout
        self._url = 'https://www.googleapis.com/books/v1/volumes'

    def search(self, query: str, max_results: int = 10) -> List[dict]:
        params = {'q': query, 'maxResults': min(max_results, 40)}
        if self._api_key:
            params['key'] = self._api_key
        try:
            resp = httpx.get(self._url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        items = data.get('items') or []
        out: List[dict] = []
        for it in items:
            vol = it.get('volumeInfo') or {}
            title = vol.get('title', '')
            link = it.get('volumeInfo', {}).get('infoLink') or it.get('selfLink', '')
            snippet = vol.get('description', '') or vol.get('subtitle', '')
            if snippet and len(snippet) > 500:
                snippet = snippet[:500] + '...'
            authors = vol.get('authors', [])
            out.append(_make_result(
                title=title,
                url=link,
                snippet=snippet,
                source=self.source_name,
                authors=authors,
                publishedDate=vol.get('publishedDate'),
                pageCount=vol.get('pageCount'),
            ))
        return out
