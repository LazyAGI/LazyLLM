from lazyllm.tools.agent.toolsManager import fc_register
from typing import List, Optional

from lazyllm.common import ApiKeyHeaderStrategy

from .base import SearchBase, _make_result


class SerplySearch(SearchBase):

    def __init__(self, api_key: Optional[str] = None,
                 endpoint: str = 'https://api.serply.io/v1/search/',
                 timeout: int = 10, source_name: str = 'serply'):
        super().__init__(
            source_name=source_name, api_key=api_key,
            auth_strategy=ApiKeyHeaderStrategy('X-Api-Key'),
            dynamic_auth=(api_key is None),
        )
        self._url = endpoint
        self._timeout = timeout

    @fc_register(host_file='NONE')
    def search(self, query: str, num: int = 10) -> List[dict]:
        # One call fetches a single result page, and a page carries at most 10 organic
        # results, so num is clamped here rather than silently truncated by the API.
        params = {'q': query, 'num': min(num, 10)}
        resp = self._request('GET', self._url, params=params, timeout=self._timeout)
        data = resp.json()
        items = data.get('results') or []
        return [
            _make_result(
                title=it.get('title', ''),
                url=it.get('link', ''),
                snippet=it.get('description', ''),
                source=self.source_name,
            )
            for it in items
        ]
