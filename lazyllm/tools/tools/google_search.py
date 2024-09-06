from lazyllm.tools.tools import HttpTool
from typing import Optional, Dict

class GoogleSearch(HttpTool):
    # @param proxies refer to https://www.python-httpx.org/advanced/proxies
    def __init__(self, custom_search_api_key: str, search_engine_id: str,
                 timeout=10, proxies: Optional[Dict] = None,
                 post_process_code: Optional[str] = None):
        # refer to https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?hl=zh-cn
        params = {
            'key': custom_search_api_key,
            'cx': '{{search_engine_id}}',
            'q': '{{query}}',
            'dateRestrict': '{{date_restrict}}',
            'start': '{{start}}',
            'num': '{{num}}',
        }
        super().__init__(method='GET', url='https://customsearch.googleapis.com/customsearch/v1',
                         params=params, timeout=timeout, proxies=proxies,
                         post_process_code=post_process_code)
        self.search_engine_id = search_engine_id

    def forward(self, query: str, date_restrict: str = 'm1',
                start: int = 0, num: int = 10,
                search_engine_id: Optional[str] = None) -> Optional[Dict]:
        params = {
            'cx': search_engine_id if search_engine_id else self.search_engine_id,
            'q': query,
            'dateRestrict': date_restrict,
            'start': str(start),
            'num': str(num),
        }
        return super().forward(**params)


if __name__ == '__main__':
    key = '<custom search api key>',
    cx = '<search engine id>',
    proxies = {
        "http://": "<http_proxy>",
        "https://": "<https_proxy>",
    }
    params = {
        'query': '商汤科技',
        'date_restrict': 'm1',
        'start': 0,
        'num': 10,
    }
    google = GoogleSearch(key, cx, proxies=proxies)
    res = google(**params)
    print(res)
