from lazyllm.tools.tools import HttpTool
from typing import Optional, Dict

class GoogleSearch(HttpTool):
    # @param proxies refer to https://www.python-httpx.org/advanced/proxies
    def __init__(self, custom_search_api_key: str, search_engine_id: str,
                 timeout=10, proxies: Optional[Dict] = None):
        # refer to https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?hl=zh-cn
        params = {
            'key': custom_search_api_key,
            'cx': '{{search_engine_id}}',
            'q': '{{query}}',
            'dateRestrict': '{{date_restrict}}',
            'start': 0,
            'num': 10,
        }
        super().__init__(method='GET', url='https://customsearch.googleapis.com/customsearch/v1',
                         params=params, timeout=timeout, proxies=proxies)
        self._search_engine_id = search_engine_id

    def forward(self, query: str, date_restrict: str = 'm1',
                search_engine_id: Optional[str] = None) -> Optional[Dict]:
        if not search_engine_id:
            search_engine_id = self._search_engine_id

        return super().forward(query=query, search_engine_id=search_engine_id,
                               date_restrict=date_restrict)
