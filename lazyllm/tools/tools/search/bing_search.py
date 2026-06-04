from typing import List, Optional

from lazyllm.common import ApiKeyHeaderStrategy

from .base import SearchBase, _make_result


class BingSearch(SearchBase):

    def __init__(self, subscription_key: Optional[str] = None,
                 endpoint: Optional[str] = 'https://api.bing.microsoft.com/v7.0/search',
                 timeout: int = 10, source_name: str = 'bing'):
        super().__init__(
            source_name=source_name, api_key=subscription_key,
            auth_strategy=ApiKeyHeaderStrategy('Ocp-Apim-Subscription-Key'),
            dynamic_auth=(subscription_key is None),
        )
        self._url = endpoint
        self._timeout = timeout

    def search(self, query: str, count: int = 10) -> List[dict]:
        params = {'q': query, 'count': min(count, 50)}
        resp = self._request('GET', self._url, params=params, timeout=self._timeout)
        data = resp.json()
        if data.get('_type') == 'ErrorResponse':
            return []
        web = data.get('webPages') or {}
        items = web.get('value') or []
        return [
            _make_result(
                title=it.get('name', ''),
                url=it.get('url', ''),
                snippet=it.get('snippet', ''),
                source=self.source_name,
            )
            for it in items
        ]
