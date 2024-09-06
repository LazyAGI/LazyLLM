from lazyllm.common import compile_code
from lazyllm.tools.http_request import HttpRequest
from typing import Optional, Dict

class HttpTool(HttpRequest):
    def __init__(self, method: str, url: str,
                 api_key: Optional[str] = None,
                 headers: Optional[Dict] = None,
                 params: Optional[Dict] = None,
                 body: Optional[str] = None,
                 timeout: int = 10,
                 proxies: Optional[Dict] = None,
                 post_process_code: Optional[str] = None):
        super().__init__(method, url, api_key, headers, params, body)
        self._post_processor = compile_code(post_process_code) if post_process_code else None

    def forward(self, *args, **kwargs):
        res = super().forward(*args, **kwargs)
        return self._post_processor(res) if self._post_processor else res
