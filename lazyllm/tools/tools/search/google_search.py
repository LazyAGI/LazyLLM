from typing import Optional, Dict, Any, List

from lazyllm.tools.tools import HttpTool

from .base import SearchBase, _make_result


class GoogleSearch(SearchBase):

    def __init__(self, custom_search_api_key: str, search_engine_id: str,
                 timeout: int = 10, proxies: Optional[Dict[str, str]] = None,
                 source_name: str = 'google'):
        super().__init__(source_name=source_name)
        params = {
            'key': custom_search_api_key,
            'cx': '{{search_engine_id}}',
            'q': '{{query}}',
            'dateRestrict': '{{date_restrict}}',
            'start': 0,
            'num': 10,
        }
        self._http = HttpTool(
            method='GET',
            url='https://customsearch.googleapis.com/customsearch/v1',
            params=params,
            timeout=timeout,
            proxies=proxies,
        )
        self._search_engine_id = search_engine_id

    def search(self, query: str,
               date_restrict: str = 'm1',
               search_engine_id: Optional[str] = None,
               raise_on_error: bool = False) -> List[Dict[str, Any]]:
        sid = search_engine_id or self._search_engine_id
        try:
            raw = self._http.forward(
                query=query,
                search_engine_id=sid,
                date_restrict=date_restrict,
            )
        except Exception as err:
            return self._handle_error(err, raise_on_error=raise_on_error)
        if not raw or not isinstance(raw, dict):
            return []
        items = raw.get('items') or []
        return [
            _make_result(
                title=it.get('title', ''),
                url=it.get('link', ''),
                snippet=it.get('snippet', ''),
                source=self.source_name,
            )
            for it in items
        ]
