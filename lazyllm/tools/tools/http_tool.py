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
                 name: str = None,
                 code_str: Optional[str] = None):
        super().__init__(method, url, '', headers, params, body)
        self._has_http = True if url else False
        self._compiled_code = compile_code(code_str) if code_str else None
        self._name = name

    def forward(self, *args, **kwargs):
        if self._has_http:
            res = super().forward(*args, **kwargs)
            return self._compiled_code(res) if self._compiled_code else res
        elif self._compiled_code:
            return self._compiled_code(*args, **kwargs)
        return None
