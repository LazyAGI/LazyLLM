from typing import Any, Dict, List, Optional

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class BochaSearch(SearchBase):

    def __init__(self, api_key: Optional[str] = None, base_url: str = 'https://api.bochaai.com',
                 timeout: int = 15, source_name: str = 'bocha', dynamic_auth: bool = False):
        if dynamic_auth and api_key is not None:
            raise ValueError('api_key must be None when dynamic_auth=True')
        super().__init__(source_name=source_name, api_key=api_key, dynamic_auth=dynamic_auth)
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout

    def search(self, query: str, topk: int = 5, include_content: bool = False,
               count: int = 10, freshness: Optional[str] = None,
               summary: bool = False) -> List[Dict[str, Any]]:
        limit = max(1, min(int(topk), 10))
        url = f'{self._base_url}/v1/web-search'
        headers = self.inject_auth_header({'Content-Type': 'application/json'})
        body = {'query': query, 'count': min(count, 20), 'summary': summary}
        if freshness:
            body['freshness'] = freshness
        resp = httpx.post(url, headers=headers, json=body, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
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
        items = out[:limit]
        if include_content:
            for item in items:
                item['content'] = self.get_content(item)
        return items
