from typing import List, Optional

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class BochaSearch(SearchBase):

    def __init__(self, api_key: str, base_url: str = 'https://api.bochaai.com',
                 timeout: int = 15, source_name: str = 'bocha'):
        super().__init__(source_name=source_name)
        self._api_key = api_key
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout

    def search(self, query: str, count: int = 10,
               freshness: Optional[str] = None,
               summary: bool = False) -> List[dict]:
        url = f'{self._base_url}/v1/web-search'
        headers = {'Authorization': f'Bearer {self._api_key}', 'Content-Type': 'application/json'}
        body = {'query': query, 'count': min(count, 20), 'summary': summary}
        if freshness:
            body['freshness'] = freshness
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        results = data.get('results') or data.get('data') or data.get('items') or []
        if isinstance(results, dict):
            results = results.get('value', results.get('results', [])) or []
        out: List[dict] = []
        for it in results:
            if isinstance(it, dict):
                title = it.get('title') or it.get('name') or ''
                url = it.get('url') or it.get('link') or ''
                snippet = it.get('snippet') or it.get('description') or it.get('summary') or ''
                out.append(_make_result(title=title, url=url, snippet=snippet, source=self.source_name))
        return out
