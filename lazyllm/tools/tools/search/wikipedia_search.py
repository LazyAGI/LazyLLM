import re
from typing import Any, Dict, List
from urllib.parse import quote

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class WikipediaSearch(SearchBase):

    def __init__(self, base_url: str = 'https://en.wikipedia.org',
                 timeout: int = 10, source_name: str = 'wikipedia'):
        super().__init__(source_name=source_name)
        self._base_url = base_url.rstrip('/')
        self._api_url = f'{self._base_url}/w/api.php'
        self._timeout = timeout

    def get_content(self, item: Dict[str, Any]) -> str:
        extra = item.get('extra') or {}
        pageid = extra.get('pageid')
        if pageid is None:
            return super().get_content(item)
        params = {
            'action': 'query',
            'pageids': pageid,
            'prop': 'extracts',
            'explaintext': 1,
            'exintro': 0,
            'format': 'json',
        }
        try:
            resp = httpx.get(self._api_url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return super().get_content(item)
        pages = data.get('query', {}).get('pages') or {}
        page = pages.get(str(pageid)) or {}
        return (page.get('extract') or '').strip() or super().get_content(item)

    def search(self, query: str, limit: int = 10) -> List[dict]:
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srlimit': min(limit, 500),
            'format': 'json',
        }
        resp = httpx.get(self._api_url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        search = data.get('query', {}).get('search') or []
        out: List[dict] = []
        for it in search:
            title = it.get('title', '')
            snippet = it.get('snippet', '')
            if snippet:
                snippet = re.sub(r'<[^>]+>', '', snippet)
            page_url = f'{self._base_url}/wiki/{quote(title.replace(" ", "_"))}'
            out.append(_make_result(
                title=title,
                url=page_url,
                snippet=snippet,
                source=self.source_name,
                pageid=it.get('pageid'),
            ))
        return out
