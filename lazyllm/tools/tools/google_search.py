from lazyllm.tools.tools import HttpTool
from typing import Optional, Dict

class GoogleSearch(HttpTool):
    """
Search for specified keywords through Google.

Args:
    custom_search_api_key (str): The Google API key applied by the user.
    search_engine_id (str): The ID of the search engine created by the user for retrieval.
    timeout (int): The timeout for the search request, in seconds, default is 10.
    proxies (Dict[str, str], optional): The proxy services used during the request. Format reference `https://www.python-httpx.org/advanced/proxies`.


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
Execute search request.

Args:
    query (str): Keywords to retrieve.
    date_restrict (str): Timeliness of the content to retrieve. Defaults to web pages within one month (m1). Refer to `https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?hl=zh-cn` for parameter format.
    search_engine_id (str, optional): Search engine ID for retrieval. If this value is empty, the value passed in the constructor is used.


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
