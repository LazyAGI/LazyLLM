from typing import Optional, Dict, Any, List

from lazyllm.common import QueryParamStrategy
from lazyllm.tools.tools import HttpTool

from .base import SearchBase, _make_result


class GoogleSearch(SearchBase):

    def __init__(self, custom_search_api_key: Optional[str] = None, search_engine_id: str = '',
                 base_url: str = 'https://customsearch.googleapis.com/customsearch/v1',
                 timeout: int = 10, proxies: Optional[Dict[str, str]] = None,
                 source_name: str = 'google', dynamic_auth: bool = False):
        if dynamic_auth and custom_search_api_key is not None:
            raise ValueError('custom_search_api_key must be None when dynamic_auth=True')
        super().__init__(
            source_name=source_name, api_key=custom_search_api_key,
            auth_strategy=QueryParamStrategy('key'),
            dynamic_auth=dynamic_auth,
        )
        params = {
            'key': '{{api_key}}',
            'cx': '{{search_engine_id}}',
            'q': '{{query}}',
            'dateRestrict': '{{date_restrict}}',
            'start': 0,
            'num': 10,
        }
        self._http = HttpTool(
            method='GET',
            url=base_url,
            params=params,
            timeout=timeout,
            proxies=proxies,
        )
        self._search_engine_id = search_engine_id

    def search(self, query: str, topk: int = 5, include_content: bool = False,
               date_restrict: str = '',
               search_engine_id: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = max(1, min(int(topk), 10))
        sid = search_engine_id or self._search_engine_id
        raw = self._http.forward(
            query=query,
            search_engine_id=sid,
            date_restrict=date_restrict,
            api_key=self.get_current_token(),
        )
        if not raw or not isinstance(raw, dict):
            return []
        items = raw.get('items') or []
        results = [
            _make_result(
                title=it.get('title', ''),
                url=it.get('link', ''),
                snippet=it.get('snippet', ''),
                source=self.source_name,
            )
            for it in items
        ][:limit]
        if include_content:
            for item in results:
                item['content'] = self.get_content(item)
        return results
