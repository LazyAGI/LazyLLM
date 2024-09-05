from lazyllm.module.module import ModuleBase
from lazyllm.tools.http_request import HttpRequest
from typing import Optional, Dict

class GoogleSearch(ModuleBase):

    # @param proxies refer to https://www.python-httpx.org/advanced/proxies
    def __init__(self, key, cx, timeout=10, proxies: Optional[Dict] = None):
        super().__init__()
        self.url = 'https://customsearch.googleapis.com/customsearch/v1'
        self.key = key
        self.cx = cx
        self.timeout = timeout
        self.proxies = proxies

    # @param params refer to https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list?hl=zh-cn
    def forward(self, params: Dict) -> Dict:
        new_params = params.copy()
        new_params.setdefault('key', self.key)
        new_params.setdefault('cx', self.cx)
        request = HttpRequest(method='GET', url=self.url, params=new_params,
                              timeout=self.timeout, proxies=self.proxies,
                              api_key='', headers=None, body=None)
        return request()


if __name__ == '__main__':
    key = '<custom search api key>',
    cx = '<search engine id>',
    proxies = {
        "http://": "<http_proxy>",
        "https://": "<https_proxy>",
    }
    params = {
        'c2coff': '1',
        'cr': 'countryCN',
        'lr': 'lang_zh-CN',
        'dateRestrict': 'm1',
        'start': 0,
        'num': 10,
        'q': '商汤科技',
    }
    google = GoogleSearch(key, cx, proxies=proxies)
    res = google(params=params)
    print(res)
