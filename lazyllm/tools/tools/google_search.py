from lazyllm.tools.tools import HttpTool
from typing import Optional, Dict

class GoogleSearch(HttpTool):
    """
通过 Google 搜索指定的关键词。

Args:
    custom_search_api_key (str): 用户申请的 Google API key。
    search_engine_id (str): 用户创建的用于检索的搜索引擎 id。
    timeout (int): 搜索请求的超时时间，单位是秒，默认是 10。
    proxies (Dict[str, str], optional): 请求时所用的代理服务。格式参考 `https://www.python-httpx.org/advanced/proxies`。


Examples:

    from lazyllm.tools.tools import GoogleSearch

    key = '<your_google_search_api_key>'
    cx = '<your_search_engine_id>'

    google = GoogleSearch(custom_search_api_key=key, search_engine_id=cx)
    """
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
        """
执行搜索请求。

Args:
    query (str): 要检索的关键词。
    date_restrict (str): 要检索内容的时效性。默认检索一个月内的网页（`m1`）。参数格式可以参考 `https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?hl=zh-cn`。
    search_engine_id (str, optional): 用于检索的搜索引擎 id。如果该值为空，则使用构造函数中传入的值。


Examples:

    from lazyllm.tools.tools import GoogleSearch

    key = '<your_google_search_api_key>'
    cx = '<your_search_engine_id>'

    google = GoogleSearch(key, cx)
    res = google(query='商汤科技', date_restrict='m1')
    """
        if not search_engine_id:
            search_engine_id = self._search_engine_id

        return super().forward(query=query, search_engine_id=search_engine_id,
                               date_restrict=date_restrict)
