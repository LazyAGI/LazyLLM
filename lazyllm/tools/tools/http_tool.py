from lazyllm.common import compile_code
from lazyllm.tools.http_request import HttpRequest
from typing import Optional, Dict

class HttpTool(HttpRequest):
    def __init__(self,
                 method: Optional[str] = None,
                 url: Optional[str] = None,
                 params: Optional[Dict[str, str]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None,
                 timeout: int = 10,
                 proxies: Optional[Dict[str, str]] = None,
                 code_str: Optional[str] = None):
        super().__init__(method, url, '', headers, params, body)
        self._has_http = True if url else False
        self._code_str = compile_code(code_str) if code_str else None

    def forward(self, *args, **kwargs):
        if self._has_http:
            res = super().forward(*args, **kwargs)
            return self._code_str(res) if self._code_str else res
        elif self._code_str:
            return self._code_str(*args, **kwargs)
        return None
