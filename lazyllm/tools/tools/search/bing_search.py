from typing import Any, Dict, List, Optional

from lazyllm.common import ApiKeyHeaderStrategy
from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class BingSearch(SearchBase):

    def __init__(self, subscription_key: Optional[str] = None,
                 base_url: str = 'https://api.bing.microsoft.com/v7.0/search',
                 timeout: int = 10, source_name: str = 'bing', dynamic_auth: bool = False):
        if dynamic_auth and subscription_key is not None:
            raise ValueError('subscription_key must be None when dynamic_auth=True')
        super().__init__(
            source_name=source_name, api_key=subscription_key,
            auth_strategy=ApiKeyHeaderStrategy('Ocp-Apim-Subscription-Key'),
            dynamic_auth=dynamic_auth,
        )
        self._url = base_url
        self._timeout = timeout

    def search(self, query: str, topk: int = 5, include_content: bool = False,
               count: int = 10) -> List[Dict[str, Any]]:
        limit = max(1, min(int(topk), 10))
        headers = self.inject_auth_header()
        params = {'q': query, 'count': min(count, 50)}
        resp = httpx.get(
            self._url,
            headers=headers,
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get('_type') == 'ErrorResponse':
            return []
        web = data.get('webPages') or {}
        items = web.get('value') or []
        results = [
            _make_result(
                title=it.get('name', ''),
                url=it.get('url', ''),
                snippet=it.get('snippet', ''),
                source=self.source_name,
            )
            for it in items
        ][:limit]
        if include_content:
            for item in results:
                item['content'] = self.get_content(item)
        return results
