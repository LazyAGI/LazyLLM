from lazyllm.tools.agent.toolsManager import fc_register
from typing import Dict, Any, List, Optional

from lazyllm.common import ApiKeyHeaderStrategy
from .base import SearchBase, _make_result


class KeenableSearch(SearchBase):

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = 'https://api.keenable.ai',
                 timeout: int = 30, source_name: str = 'keenable'):
        # The key is optional: without one the free public endpoint is used.
        super().__init__(
            source_name=source_name, api_key=api_key or '',
            auth_strategy=ApiKeyHeaderStrategy('X-API-Key'),
            skip_auth=True,
        )
        self._base_url = base_url.rstrip('/')
        self._timeout = timeout

    @fc_register(host_file='NONE')
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        path = '/v1/search' if self._get_token() else '/v1/search/public'
        resp = self._request(
            'POST', f'{self._base_url}{path}',
            json={'query': query, 'max_results': max_results},
            headers={'X-Keenable-Title': 'LazyLLM'},
            timeout=self._timeout,
        )
        data = resp.json()
        out: List[Dict[str, Any]] = []
        for it in data.get('results') or []:
            url = it.get('url') or ''
            if not url:
                continue
            # Prefer the short description; _make_result truncates long snippets.
            out.append(_make_result(
                title=it.get('title') or '',
                url=url,
                snippet=it.get('description') or it.get('snippet') or '',
                source=self.source_name,
            ))
        return out[:max_results]
