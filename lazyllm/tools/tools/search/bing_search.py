from typing import List, Optional

from lazyllm.thirdparty import httpx

from .base import SearchBase, _make_result


class BingSearch(SearchBase):

    def __init__(self, subscription_key: str, endpoint: Optional[str] = None,
                 timeout: int = 10, source_name: str = 'bing'):
        super().__init__(source_name=source_name)
        self._key = subscription_key
        self._url = endpoint or 'https://api.bing.microsoft.com/v7.0/search'
        self._timeout = timeout

    def search(self, query: str, count: int = 10) -> List[dict]:
        headers = {'Ocp-Apim-Subscription-Key': self._key}
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
        return [
            _make_result(
                title=it.get('name', ''),
                url=it.get('url', ''),
                snippet=it.get('snippet', ''),
                source=self.source_name,
            )
            for it in items
        ]
